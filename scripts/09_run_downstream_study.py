#!/usr/bin/env python3
"""Evaluate all four frozen nanoMaternalPFN models on the SAME real holdouts.

Default is a small CPU pilot; use --full for 20 prespecified draws per budget.
No training occurs. No synthetic data are substituted for missing real data.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from nanomaternalpfn.evaluation.transfer_curves import evaluate

DEFAULT_DATASETS = (
    "uci_maternal_risk", "vat_gdm", "vat_delivery", "low_birth_weight",
    "cardiotocography", "pima_diabetes", "tpehg_preterm",
    "ctu_cord_acidemia",
)
ARMS = (
    ("tiny", "generic"), ("tiny", "mixture"),
    ("base", "generic"), ("base", "mixture"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("artifacts/results/aws_full_pretrain"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/results/downstream_pilot"))
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS,
                        default=list(DEFAULT_DATASETS))
    parser.add_argument("--full", action="store_true",
                        help="Run all context budgets and 20 draws per budget")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--symmetrize-classes", action="store_true")
    parser.add_argument("--calibrate-context", action="store_true")
    args = parser.parse_args()
    if args.query_batch_size < 1:
        parser.error("query-batch-size must be positive")

    checkpoint_paths = {
        (tier, prior): args.checkpoint_dir / f"{tier}_{prior}.best.inference.pt"
        for tier, prior in ARMS
    }
    for arm, path in checkpoint_paths.items():
        if not path.is_file():
            parser.error(f"Missing pretrained checkpoint for {arm}: {path}")
    for name in args.datasets:
        path = args.manifest_dir / f"{name}.json"
        if not path.is_file() or not path.with_suffix(".npz").is_file():
            parser.error(f"Missing frozen real dataset: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "study_config.json"
    config = {
        "full": args.full, "pilot_only": not args.full,
        "datasets": args.datasets,
        "max_draws": None if args.full else 1,
        "contexts": None if args.full else [16],
        "query_batch_size": args.query_batch_size,
        "symmetrize_classes":args.symmetrize_classes,
        "calibrate_context":args.calibrate_context,
        "checkpoints": {f"{t}_{p}": str(path) for (t,p),path in checkpoint_paths.items()},
        "synthetic_validation_is_not_clinical_validation": True,
        "observational_test_sets_are_not_for_model_selection": True,
        "shared_cohorts": [["vat_gdm", "vat_delivery"]],
        "nonmaternal_control": "pima_diabetes",
        "proxy_task": "uci_maternal_risk",
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Output directory already belongs to another study configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    summaries = []
    for name in args.datasets:
        manifest = args.manifest_dir / f"{name}.json"
        # Fit classical baselines once per dataset, not four times.
        methods = [
            ("logistic", None, ("logistic",)),
            ("random_forest", None, ("random_forest",)),
        ]
        methods += [
            (f"{tier}_{prior}", checkpoint_paths[(tier, prior)], ("pfn",))
            for tier, prior in ARMS
        ]
        for label, ckpt, meth in methods:
            csv_path = args.output_dir / f"{name}.{label}.csv"
            print(f"START {name} / {label}", flush=True)
            rows = evaluate(
                manifest, checkpoint=ckpt, methods=meth,
                output=csv_path, query_batch_size=args.query_batch_size,
                max_draws=None if args.full else 1,
                contexts=None if args.full else [16],
                pfn_kwargs={"symmetrize_classes":args.symmetrize_classes,
                            "calibrate_context":args.calibrate_context},
            )
            for metric in ("auroc", "accuracy", "brier", "ece15"):
                import numpy as np
                by_context = defaultdict(list)
                for row in rows:
                    value = row[metric]
                    if np.isfinite(value):
                        by_context[int(row["context"])].append(value)
                for budget, values in sorted(by_context.items()):
                    summaries.append({
                        "dataset": name, "method": label, "context": budget,
                        "metric": metric, "n_valid_draws": len(values),
                        "mean": float(np.mean(values)),
                        "sd_between_context_draws": (
                            float(np.std(values, ddof=1)) if len(values)>1 else float("nan")
                        ),
                    })
            with (args.output_dir / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=("dataset","method","context","metric",
                                    "n_valid_draws","mean","sd_between_context_draws")
                )
                writer.writeheader()
                writer.writerows(summaries)
            print(f"DONE {name} / {label}: {len(rows)} evaluated draws", flush=True)
    print(f"Completed {len(args.datasets)} real dataset tasks. "
          f"Summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
