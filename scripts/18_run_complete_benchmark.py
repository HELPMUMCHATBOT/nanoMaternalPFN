#!/usr/bin/env python3
"""Complete, resumable frozen-cohort benchmark across 13 model configurations.

All datasets, context draws, and endpoints are preserved exactly. Model choices
and score reporting are exploratory because earlier holdouts informed v2 design.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from nanomaternalpfn.evaluation.baselines import get_baseline
from nanomaternalpfn.evaluation.calibration import multiclass_brier, ece
from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
from nanomaternalpfn.nanotabpfn_baseline import NanoTabPFNReferenceClassifier

DATASETS = (
    "uci_maternal_risk", "vat_gdm", "vat_delivery", "low_birth_weight",
    "cardiotocography", "pima_diabetes", "tpehg_preterm",
    "ctu_cord_acidemia",
)
METHODS = (
    "logistic", "random_forest", "xgboost", "lightgbm",
    "nanotabpfn_generic", "nanotabpfn_mixture",
    "tabpfn_v2_single", "tabpfn_v2_default",
    "tiny_generic", "tiny_mixture", "base_generic", "base_mixture",
    *V1_ARMS, "adaptive_tiny", "adaptive_base",
)
PFN_ARMS = ("tiny_generic", "tiny_mixture", "base_generic", "base_mixture")
V1_ARMS = tuple("v1_"+name for name in PFN_ARMS)
RAW_FIELDS = (
    "dataset", "method", "context", "seed", "accuracy", "brier", "ece15",
    "auroc", "fit_ms", "predict_ms", "peak_gpu_allocated_mb",
    "peak_gpu_reserved_mb", "max_process_rss_mb",
)
METRICS = ("auroc", "accuracy", "brier", "ece15")
SUMMARY_FIELDS = (
    "dataset", "method", "context", "metric", "n_valid_draws", "mean",
    "sd_between_context_draws", "mean_fit_ms", "mean_predict_ms",
    "mean_peak_gpu_allocated_mb", "mean_peak_gpu_reserved_mb",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    manifest = json.loads(path.read_text())
    npz_path = path.with_suffix(".npz")
    if _sha256(npz_path) != manifest["sha256"]:
        raise RuntimeError(f"Frozen dataset SHA256 mismatch: {path}")
    with np.load(npz_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"])
    test = np.asarray(manifest["test_idx"], dtype=np.int64)
    if not len(test):
        raise RuntimeError(f"No held-out test records: {path}")
    classes = np.unique(y)
    if len(classes) not in (2, 3):
        raise RuntimeError(f"Expected 2/3 classes: {path}")
    return manifest, X, y, test, classes


def make_estimator(method, root, device, seed, query_batch_size):
    ckpts = root / "checkpoints"
    if method in ("nanotabpfn_generic","nanotabpfn_mixture"):
        return NanoTabPFNReferenceClassifier(
            ckpts / f"{method}.best.inference.pt",
            device=device, query_batch_size=query_batch_size,
        )
    if method in V1_ARMS:
        return NanoMaternalPFNClassifier(
            ckpts / "v1" / f"{method[3:]}.best.inference.pt",
            device=device,query_batch_size=query_batch_size,
        )
    if method in PFN_ARMS:
        return NanoMaternalPFNClassifier(
            ckpts / "v2" / f"{method}.best.inference.pt",
            device=device, query_batch_size=query_batch_size,
        )
    if method.startswith("adaptive_"):
        tier = method.split("_", 1)[1]
        return AdaptiveNanoMaternalPFNClassifier(
            (
                ckpts / "v1" / f"{tier}_mixture.best.inference.pt",
                ckpts / "v2" / f"{tier}_mixture.best.inference.pt",
            ),
            query_batch_size=query_batch_size, device=device,
            cv_folds=4, calibrate=True,
        )
    return get_baseline(
        method, seed=seed, tabpfn_device=device
    )


def validate_probs(p, estimator, classes, n_rows, method):
    p = np.asarray(p, dtype=np.float64)
    local = np.asarray(estimator.classes_)
    aligned = np.zeros((n_rows, len(classes)), dtype=np.float64)
    if p.shape != (n_rows, len(local)):
        raise RuntimeError(
            f"{method}: returned {p.shape}, expected {(n_rows,len(local))}"
        )
    for j, label in enumerate(local):
        idx = np.where(classes == label)[0]
        if len(idx) != 1:
            raise RuntimeError(f"{method}: unexpected label {label}")
        aligned[:, idx[0]] = p[:, j]
    if not np.isfinite(aligned).all() or np.any(aligned < -1e-6):
        raise RuntimeError(f"{method}: invalid finite/probability values")
    if not np.allclose(aligned.sum(axis=1), 1, atol=1e-4):
        raise RuntimeError(f"{method}: probabilities not normalized")
    return np.clip(aligned, 0, 1)


def score_predictions(y_true, pred, classes):
    truth = np.searchsorted(classes, y_true)
    scores = {
        "accuracy": float(accuracy_score(truth, pred.argmax(axis=1))),
        "brier": multiclass_brier(truth, pred),
        "ece15": ece(truth, pred),
    }
    try:
        scores["auroc"] = float(roc_auc_score(
            truth,
            pred[:, 1] if len(classes) == 2 else pred,
            multi_class="ovr",
        ))
    except ValueError:
        scores["auroc"] = float("nan")
    return scores


def _rss_mb():
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / 1e6)
    except ImportError:
        return float("nan")


def summarize(raw_path, summary_path):
    groups = defaultdict(list)
    with raw_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["dataset"],row["method"],int(row["context"]))].append(row)
    entries=[]
    for (dataset,method,context), observations in sorted(groups.items()):
        for metric in METRICS:
            vals=np.array([float(r[metric]) for r in observations])
            finite=np.isfinite(vals)
            if not finite.any():
                continue
            def avg(field):
                x=np.array([float(r[field]) for r in observations])
                return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
            entries.append({
                "dataset":dataset, "method":method, "context":context,
                "metric":metric, "n_valid_draws":int(finite.sum()),
                "mean":float(vals[finite].mean()),
                "sd_between_context_draws":float(
                    vals[finite].std(ddof=1)
                ) if finite.sum()>1 else float("nan"),
                "mean_fit_ms":avg("fit_ms"),
                "mean_predict_ms":avg("predict_ms"),
                "mean_peak_gpu_allocated_mb":avg("peak_gpu_allocated_mb"),
                "mean_peak_gpu_reserved_mb":avg("peak_gpu_reserved_mb"),
            })
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    with summary_path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(entries)
    return len(entries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--contexts",nargs="+",type=int,default=[16,32,64,128,256,512])
    p.add_argument("--max-draws",type=int,default=None)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--query-batch-size",type=int,default=64)
    p.add_argument("--continue-on-error",action="store_true",help="Log method failures; incomplete run is not a finished benchmark")
    args=p.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("Requested CUDA but no NVIDIA GPU is available")
    if args.max_draws is not None and args.max_draws < 1:
        p.error("--max-draws must be positive")
    if args.query_batch_size < 1 or len(set(args.methods)) != len(args.methods):
        p.error("Invalid batch size or duplicate methods")
    if not args.contexts or min(args.contexts) < 1:
        p.error("Invalid contexts")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    random.seed(20261107)
    np.random.seed(20261107)
    torch.manual_seed(20261107)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config = {
        "datasets":args.datasets, "methods":args.methods,
        "contexts":args.contexts, "max_draws":args.max_draws,
        "device":args.device, "query_batch_size":args.query_batch_size,
        "frozen_splits":True, "exploratory_not_confirmatory":True,
        "paper_title":"nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction",
    }
    config_file=args.output_dir/"benchmark_config.json"
    if config_file.exists() and json.loads(config_file.read_text())!=config:
        p.error("Cannot resume a different benchmark configuration")
    config_file.write_text(json.dumps(config,indent=2)+"\n")
    raw_path=args.output_dir/"all_draws.csv"
    summary_path=args.output_dir/"summary.csv"
    completed=set()
    if raw_path.exists():
        with raw_path.open(newline="") as handle:
            reader=csv.DictReader(handle)
            if tuple(reader.fieldnames or ())!=RAW_FIELDS:
                p.error("Existing all_draws.csv has different schema")
            for item in reader:
                key=(item["dataset"],item["method"],int(item["context"]),int(item["seed"]))
                if key in completed:
                    p.error("Duplicate existing result: "+str(key))
                completed.add(key)
    failures=args.output_dir/"failures.jsonl"
    incomplete=0
    with raw_path.open("a",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=RAW_FIELDS)
        if handle.tell()==0:
            writer.writeheader()
        for dataset in args.datasets:
            path=args.input_dir/"manifests"/f"{dataset}.json"
            manifest,X,y,test,classes=load_manifest(path)
            for method in args.methods:
                # A model is loaded at most once per dataset/method; context CV
                # and fixed global torch state cannot inspect held-out labels.
                estimator=None
                for budget,draws in sorted(
                    manifest["context_draws"].items(),
                    key=lambda kv:int(kv[0]),
                ):
                    budget=int(budget)
                    if budget not in args.contexts:
                        continue
                    for seed,draw in enumerate(
                        draws[:args.max_draws] if args.max_draws else draws
                    ):
                        key=(dataset,method,budget,seed)
                        if key in completed:
                            continue
                        draw=np.asarray(draw,dtype=np.int64)
                        if len(np.unique(y[draw]))<2:
                            continue
                        if estimator is None or method in (
                            "logistic","random_forest","xgboost","lightgbm",
                            "tabpfn_v2_single","tabpfn_v2_default",
                        ):
                            estimator=make_estimator(
                                method,args.input_dir,args.device,seed,args.query_batch_size
                            )
                        try:
                            if args.device=="cuda":
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                            t0=time.perf_counter()
                            estimator.fit(X[draw],y[draw])
                            if args.device=="cuda":
                                torch.cuda.synchronize()
                            fit_ms=(time.perf_counter()-t0)*1000
                            t0=time.perf_counter()
                            probability=estimator.predict_proba(X[test])
                            if args.device=="cuda":
                                torch.cuda.synchronize()
                            predict_ms=(time.perf_counter()-t0)*1000
                            aligned=validate_probs(
                                probability,estimator,classes,len(test),method
                            )
                            metrics=score_predictions(y[test],aligned,classes)
                            if args.device=="cuda":
                                peak_alloc=float(torch.cuda.max_memory_allocated()/1e6)
                                peak_reserved=float(torch.cuda.max_memory_reserved()/1e6)
                            else:
                                peak_alloc=peak_reserved=float("nan")
                            writer.writerow({
                                "dataset":dataset,"method":method,
                                "context":budget,"seed":seed,**metrics,
                                "fit_ms":fit_ms,"predict_ms":predict_ms,
                                "peak_gpu_allocated_mb":peak_alloc,
                                "peak_gpu_reserved_mb":peak_reserved,
                                "max_process_rss_mb":_rss_mb(),
                            })
                            handle.flush()
                            os.fsync(handle.fileno())
                            completed.add(key)
                            print(f"{dataset}/{method} ctx={budget} "
                                  f"draw={seed+1}/{len(draws)} "
                                  f"AUROC={metrics['auroc']:.4f} "
                                  f"Brier={metrics['brier']:.4f} "
                                  f"fit={fit_ms:.1f}ms query={predict_ms:.1f}ms",
                                  flush=True)
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            incomplete+=1
                            with failures.open("a") as errors:
                                errors.write(json.dumps({
                                    "dataset":dataset,"method":method,
                                    "context":budget,"seed":seed,
                                    "error":repr(exc),
                                })+"\n")
                            print(f"FAILED {key}: {exc}",flush=True)
                    if raw_path.stat().st_size:
                        summarize(raw_path,summary_path)
                if method.startswith("tabpfn"):
                    del estimator
                    if args.device=="cuda":
                        torch.cuda.empty_cache()
    n_summary=summarize(raw_path,summary_path)
    runtime={
        "torch":torch.__version__,
        "numpy":np.__version__,
        "python":platform.python_version(),
        "cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tabpfn_version":None,
    }
    try:
        import importlib.metadata
        runtime["tabpfn_version"]=importlib.metadata.version("tabpfn")
        runtime["xgboost_version"]=importlib.metadata.version("xgboost")
        runtime["lightgbm_version"]=importlib.metadata.version("lightgbm")
    except Exception:
        pass
    (args.output_dir/"runtime_versions.json").write_text(
        json.dumps(runtime,indent=2)+"\n"
    )
    print(json.dumps({
        "observations":len(completed),"summary_rows":n_summary,
        "failures":incomplete,"output":str(args.output_dir)
    },indent=2),flush=True)
    if incomplete:
        raise RuntimeError("Benchmark incomplete; see failures.jsonl")


if __name__=="__main__":
    main()
