#!/usr/bin/env python3
"""Frozen-context real-data benchmarks with optional paper-completion methods.

--resume preserves completed context draws. Existing v1/v2 summaries should
remain untouched: use a new output directory for additional comparators.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from nanomaternalpfn.evaluation.transfer_curves import evaluate

DEFAULT_DATASETS = (
    "uci_maternal_risk", "vat_gdm", "vat_delivery", "low_birth_weight",
    "cardiotocography", "pima_diabetes", "tpehg_preterm",
    "ctu_cord_acidemia",
)
ARMS = ("tiny_generic", "tiny_mixture", "base_generic", "base_mixture")
ORIGINAL = ("logistic", "random_forest") + ARMS
PAPER = ("xgboost", "lightgbm", "tabpfn_v2_single", "tabpfn_v2_default",
         "adaptive_tiny", "adaptive_base")
RESEARCH_ADAPTIVE = tuple(
    f"adaptive_{tier}_{kind}" for tier in ("tiny", "base")
    for kind in ("three", "four")
)
SUMMARY_FIELDS = ("dataset", "method", "context", "metric",
                  "n_valid_draws", "mean", "sd_between_context_draws")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--v1-checkpoint-dir", type=Path,
                   default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/results/downstream_pilot"))
    p.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS,
                   default=list(DEFAULT_DATASETS))
    p.add_argument("--methods", nargs="+", choices=ORIGINAL + PAPER + RESEARCH_ADAPTIVE,
                   default=list(ORIGINAL))
    p.add_argument("--full", action="store_true")
    p.add_argument("--max-draws", type=int, default=None)
    p.add_argument("--contexts", nargs="+", type=int, default=None)
    p.add_argument("--query-batch-size", type=int, default=8)
    p.add_argument("--tabpfn-device", default="cpu",
                   choices=("cpu", "mps", "cuda", "auto"))
    p.add_argument("--symmetrize-classes", action="store_true")
    p.add_argument("--calibrate-context", action="store_true")
    p.add_argument("--no-adaptive-calibration", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--adaptive-weighting", choices=("softmax", "winner", "uniform"), default="softmax")
    args = p.parse_args()
    if args.query_batch_size < 1 or (
        args.max_draws is not None and args.max_draws < 1
    ):
        p.error("batch size and maximum draws must be positive")
    if len(set(args.methods)) != len(args.methods):
        p.error("duplicate method")
    if args.contexts is not None and (
        len(set(args.contexts)) != len(args.contexts)
        or min(args.contexts) < 1
    ):
        p.error("invalid context budgets")
    budgets = args.contexts if args.contexts is not None else (
        None if args.full else [16]
    )
    max_draws = args.max_draws if args.max_draws is not None else (
        None if args.full else 1
    )
    checkpoints = {
        arm: args.checkpoint_dir / f"{arm}.best.inference.pt" for arm in ARMS
    }
    adaptive = {
        f"adaptive_{tier}": (
            args.v1_checkpoint_dir / f"{tier}_mixture.best.inference.pt",
            args.checkpoint_dir / f"{tier}_mixture.best.inference.pt",
        ) for tier in ("tiny", "base")
    }
    for tier in ("tiny", "base"):
        adaptive[f"adaptive_{tier}_three"] = (
            args.v1_checkpoint_dir / f"{tier}_mixture.best.inference.pt",
            args.checkpoint_dir / f"{tier}_mixture.best.inference.pt",
            args.checkpoint_dir / f"{tier}_generic.best.inference.pt",
        )
        adaptive[f"adaptive_{tier}_four"] = (
            *adaptive[f"adaptive_{tier}_three"],
            args.v1_checkpoint_dir / f"{tier}_generic.best.inference.pt",
        )
    for method in args.methods:
        for filename in (
            adaptive[method] if method in adaptive else
            (checkpoints[method],) if method in checkpoints else ()
        ):
            if not filename.is_file():
                p.error(f"Missing checkpoint for {method}: {filename}")
    for name in args.datasets:
        manifest = args.manifest_dir / f"{name}.json"
        if not manifest.is_file() or not manifest.with_suffix(".npz").is_file():
            p.error(f"Missing frozen real dataset: {manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "study_config.json"
    config = {
        "datasets":args.datasets, "methods":args.methods, "full":args.full,
        "contexts":budgets, "max_draws":max_draws,
        "query_batch_size":args.query_batch_size,
        "tabpfn_device":args.tabpfn_device,
        "tabpfn_version":"ModelVersion.V2",
        "tabpfn_one_pass_n_estimators":1,
        "tabpfn_default_n_estimators":"auto",
        "symmetrize_classes":args.symmetrize_classes,
        "calibrate_context":args.calibrate_context,
        "adaptive_calibrate":not args.no_adaptive_calibration,
        "adaptive_weighting": args.adaptive_weighting,
        "checkpoints":{
            m:str(checkpoints[m]) for m in args.methods if m in checkpoints
        },
        "adaptive_checkpoints":{
            m:[str(v) for v in adaptive[m]]
            for m in args.methods if m in adaptive
        },
        "not_confirmatory":True,
        "shared_cohorts":[["vat_gdm","vat_delivery"]],
        "nonmaternal_control":"pima_diabetes",
    }
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            p.error("Existing study configuration differs")
        if not args.resume:
            p.error("Results already exist; use --resume or fresh output-dir")
    else:
        config_path.write_text(json.dumps(config, indent=2)+"\n")
    summaries = []
    for name in args.datasets:
        manifest = args.manifest_dir / f"{name}.json"
        for method in args.methods:
            mode = (
                "pfn" if method in checkpoints else
                "adaptive" if method in adaptive else method
            )
            output = args.output_dir / f"{name}.{method}.csv"
            print(f"START {name} / {method}", flush=True)
            rows = evaluate(
                manifest,
                checkpoint=checkpoints.get(method),
                methods=(mode,),
                output=output,
                max_draws=max_draws,
                contexts=budgets,
                query_batch_size=args.query_batch_size,
                pfn_kwargs={
                    "symmetrize_classes":args.symmetrize_classes,
                    "calibrate_context":args.calibrate_context,
                },
                adaptive_checkpoints=adaptive.get(method),
                adaptive_kwargs={
                    "calibrate":not args.no_adaptive_calibration,
                    "weighting":args.adaptive_weighting,
                },
                baseline_kwargs={"tabpfn_device":args.tabpfn_device},
                resume=args.resume,
            )
            for metric in ("auroc","accuracy","brier","ece15"):
                groups = defaultdict(list)
                for row in rows:
                    if np.isfinite(row[metric]):
                        groups[int(row["context"])].append(float(row[metric]))
                for budget, vals in sorted(groups.items()):
                    summaries.append({
                        "dataset":name, "method":method, "context":budget,
                        "metric":metric, "n_valid_draws":len(vals),
                        "mean":float(np.mean(vals)),
                        "sd_between_context_draws":(
                            float(np.std(vals,ddof=1)) if len(vals)>1 else float("nan")
                        ),
                    })
            with (args.output_dir / "summary.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(summaries)
            print(f"DONE {name}/{method}: {len(rows)} draws", flush=True)
    print(f"Completed {len(args.datasets)} real dataset tasks. "
          f"Summary: {args.output_dir / 'summary.csv'}")


if __name__=="__main__":
    main()
