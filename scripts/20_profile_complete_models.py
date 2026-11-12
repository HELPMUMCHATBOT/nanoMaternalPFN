#!/usr/bin/env python3
"""Matched single-query fit/inference timing for all 13 methods on one GPU host.

Synthetic timing inputs only; no fake clinical performance scores. Timing
includes context setup for each estimator and excludes first-load/download
cost. PFNs condition on context rather than updating neural parameters.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
from nanomaternalpfn.nanotabpfn_baseline import NanoTabPFNReferenceClassifier
from nanomaternalpfn.evaluation.baselines import get_baseline

ARMS = ("tiny_generic","tiny_mixture","base_generic","base_mixture")
V1_ARMS = tuple("v1_"+name for name in ARMS)
METHODS = ("logistic","random_forest","xgboost","lightgbm",
           "nanotabpfn_generic","nanotabpfn_mixture","tabpfn_v2_single","tabpfn_v2_default",
           *ARMS,*V1_ARMS,"adaptive_tiny","adaptive_base")


def estimator(method,root,device):
    ckpt=root/"checkpoints"
    if method in ("nanotabpfn_generic","nanotabpfn_mixture"):
        return NanoTabPFNReferenceClassifier(
            ckpt/f"{method}.best.inference.pt",
            device=device, query_batch_size=1,
        )
    if method in V1_ARMS:
        return NanoMaternalPFNClassifier(
            ckpt/"v1"/f"{method[3:]}.best.inference.pt",
            device=device,query_batch_size=1,
        )
    if method in ARMS:
        return NanoMaternalPFNClassifier(
            ckpt/"v2"/f"{method}.best.inference.pt",
            device=device,query_batch_size=1,
        )
    if method.startswith("adaptive_"):
        tier=method.split("_",1)[1]
        return AdaptiveNanoMaternalPFNClassifier(
            [ckpt/"v1"/f"{tier}_mixture.best.inference.pt",
             ckpt/"v2"/f"{tier}_mixture.best.inference.pt"],
            device=device,query_batch_size=1,
        )
    return get_baseline(method, seed=23, tabpfn_device=device)


def measure(estimator,X,y,query,device,repeats):
    estimator.fit(X,y)
    estimator.predict_proba(query)
    fit=[], predict=[]
    for _ in range(repeats):
        if device=="cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        estimator.fit(X,y)
        if device=="cuda":
            torch.cuda.synchronize()
        fit.append(1000*(time.perf_counter()-begin))
        begin=time.perf_counter()
        estimator.predict_proba(query)
        if device=="cuda":
            torch.cuda.synchronize()
        predict.append(1000*(time.perf_counter()-begin))
    return {
        "fit_p50_ms":statistics.median(fit),
        "query_p50_ms":statistics.median(predict),
        "fit_plus_query_p50_ms":statistics.median([a+b for a,b in zip(fit,predict)]),
        "query_p99_ms":float(np.percentile(predict,99)),
        "gpu_peak_reserved_mb":float(torch.cuda.max_memory_reserved()/1e6)
            if device=="cuda" else None,
        "gpu_peak_allocated_mb":float(torch.cuda.max_memory_allocated()/1e6)
            if device=="cuda" else None,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--contexts",nargs="+",type=int,default=[16,64,128])
    p.add_argument("--features",type=int,default=6)
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    if args.repeats<1 or args.features<1 or min(args.contexts)<2:
        p.error("Invalid repeats/features/contexts")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    entries=[]
    for method in METHODS:
        model=None
        for n in args.contexts:
            rng=np.random.default_rng(312+n)
            X=rng.standard_normal((n,args.features)).astype(np.float32)
            query=rng.standard_normal((1,args.features)).astype(np.float32)
            y=(np.arange(n)%2).astype(np.int64)
            X[1,0]=np.nan
            # Cold checkpoint load and downloads excluded from warm timing;
            # all methods run identical 1-query synthetic dimensions.
            if model is None or method in (
                "logistic","random_forest","xgboost","lightgbm",
                "tabpfn_v2_single","tabpfn_v2_default",
            ):
                model=estimator(method,args.input_dir,args.device)
            timing=measure(model,X,y,query,args.device,args.repeats)
            row={
                "method":method,"context":n,"features":args.features,
                "device":args.device, "repeats":args.repeats,
                "query_rows":1,
                "input_type":"matched synthetic timing only",
                "excludes_cold_load_and_download":True,
                **timing,
            }
            entries.append(row)
            args.output.write_text(json.dumps({
                "platform":platform.platform(),
                "gpu":torch.cuda.get_device_name(0) if args.device=="cuda" else None,
                "rows":entries,
            },indent=2)+"\n")
            print(f"{method} ctx={n}: query p50={timing['query_p50_ms']:.2f}ms "
                  f"fit p50={timing['fit_p50_ms']:.2f}ms",flush=True)
        del model
        if args.device=="cuda":
            torch.cuda.empty_cache()
    print(f"SAVED {len(entries)} comparable latency entries: {args.output}")


if __name__=="__main__":
    main()

