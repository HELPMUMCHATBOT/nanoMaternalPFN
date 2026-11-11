#!/usr/bin/env python3
"""Exploratory, matched-draw comparison of original vs symmetrized AWS PFNs.

WARNING: The original downstream test scores have already been examined.
Results on the same test sets are POST-HOC and cannot validate model
selection or constitute a new unbiased clinical claim. Use --split train
for a diagnostic independent of held-out TEST labels. The original
downstream_full/ results and original checkpoints are never modified.

Examples:
    python scripts/11_compare_symmetrized_inference.py
    python scripts/11_compare_symmetrized_inference.py --split test --datasets uci_maternal_risk low_birth_weight
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
from nanomaternalpfn.symmetrized_classifier import (
    SymmetrizedNanoMaternalPFNClassifier, context_prior_adjustment,
)
from nanomaternalpfn.evaluation.calibration import multiclass_brier, ece


def score(truth, probabilities, classes):
    predicted = classes[np.argmax(probabilities, axis=1)]
    integer_truth = np.searchsorted(classes, truth)
    try:
        auroc = float(roc_auc_score(
            integer_truth,
            probabilities[:, 1] if len(classes) == 2 else probabilities,
            multi_class="ovr",
        ))
    except ValueError:
        auroc = float("nan")
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(truth, predicted)),
        "brier": multiclass_brier(integer_truth, probabilities),
        "ece15": ece(integer_truth, probabilities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/results/aws_full_pretrain"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/symmetrization_training_diagnostic"))
    parser.add_argument("--datasets", nargs="+", default=["uci_maternal_risk", "low_birth_weight"])
    parser.add_argument("--models", nargs="+", default=["tiny_mixture", "base_mixture"])
    parser.add_argument("--contexts", type=int, nargs="+", default=[64])
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--queries", type=int, default=48, help="training-split diagnostic only")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-permutations", type=int, default=1)
    parser.add_argument("--context-calibration-alpha", type=float, default=0.0,
                        help="optional transductive adjustment, 0 disables; report separately")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()
    if args.draws < 1 or args.batch_size < 1 or args.queries < 1:
        parser.error("draws, queries, batch-size must be positive")
    if args.feature_permutations < 1:
        parser.error("feature-permutations must be positive")
    if not 0 <= args.context_calibration_alpha <= 1:
        parser.error("context-calibration-alpha must lie in [0,1]")
    # Never overwrite the prespecified study output (even accidentally).
    baseline = Path("artifacts/results/downstream_full").resolve()
    if args.output_dir.resolve() == baseline or baseline in args.output_dir.resolve().parents:
        parser.error("Refusing to write inside the prespecified downstream_full output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))

    config = {
        "exploratory": True,
        "split": args.split,
        "heldout_warning": "POST-HOC; original heldout metrics already known" if args.split == "test" else None,
        "datasets": args.datasets, "models": args.models,
        "contexts": args.contexts, "draws": args.draws,
        "queries": args.queries if args.split == "train" else "all heldout test rows",
        "feature_permutations": args.feature_permutations,
        "class_permutations": "all 2! or 3!",
        "calibration_alpha": args.context_calibration_alpha,
        "transductive_adjustment": bool(args.context_calibration_alpha),
        "brier_definition": "multiclass one-hot sum of squared errors (binary scores multiplied by 2 vs ordinary binary Brier)",
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Different configuration already present in output-dir; use a fresh output path")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    fields = ("dataset", "model", "context", "draw", "split", "variant",
              "n_queries", "elapsed_seconds", "auroc", "accuracy", "brier", "ece15")
    output = args.output_dir / "per_draw.csv"
    with output.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        for name in args.datasets:
            manifest_path = args.manifest_dir / (name + ".json")
            manifest = json.loads(manifest_path.read_text())
            npz_path = manifest_path.with_suffix(".npz")
            if hashlib.sha256(npz_path.read_bytes()).hexdigest() != manifest["sha256"]:
                raise RuntimeError(f"{name}: frozen NPZ integrity check failed")
            with np.load(npz_path, allow_pickle=False) as d:
                X, y = d["X"].copy(), d["y"].astype(np.int64).copy()
                train, test = d["train_idx"].copy(), d["test_idx"].copy()
            classes = np.unique(y)
            for model_name in args.models:
                path = args.checkpoint_dir / (model_name + ".best.inference.pt")
                raw = NanoMaternalPFNClassifier(path, query_batch_size=args.batch_size)
                sym = SymmetrizedNanoMaternalPFNClassifier(
                    path, query_batch_size=args.batch_size,
                    n_feature_permutations=args.feature_permutations,
                )
                for context in args.contexts:
                    draws = manifest["context_draws"].get(str(context), [])[:args.draws]
                    for i, context_rows in enumerate(draws):
                        indices = np.asarray(context_rows, dtype=np.int64)
                        if len(np.unique(y[indices])) != len(classes):
                            print(f"SKIP {name} ctx={context} draw={i} incomplete classes", flush=True)
                            continue
                        if args.split == "test":
                            query = test
                        else:
                            remaining = train[~np.isin(train, indices)]
                            # Use exact feature hashes to exclude matching context profiles,
                            # including NaN positions.
                            def profile_key(row):
                                return np.nan_to_num(row, nan=9.87654321e19).tobytes()
                            used = {profile_key(row) for row in X[indices]}
                            query = np.array([
                                idx for idx in remaining if profile_key(X[idx]) not in used
                            ][:args.queries], dtype=np.int64)
                        if not len(query):
                            continue
                        for label, estimator in (("original", raw), ("symmetrized", sym)):
                            estimator.fit(X[indices], y[indices])
                            before = time.perf_counter()
                            probs = estimator.predict_proba(X[query])
                            elapsed = time.perf_counter() - before
                            predictions = [(label, probs, elapsed)]
                            if label == "symmetrized" and args.context_calibration_alpha > 0:
                                predictions.append((
                                    "symmetrized_transductive_prior",
                                    context_prior_adjustment(
                                        probs, estimator.y_,
                                        alpha=args.context_calibration_alpha,
                                    ), elapsed,
                                ))
                            for variant, pp, seconds in predictions:
                                row = {
                                    "dataset": name, "model": model_name, "context": context,
                                    "draw": i, "split": args.split, "variant": variant,
                                    "n_queries": len(query), "elapsed_seconds": round(seconds, 4),
                                    **score(y[query], pp, classes),
                                }
                                writer.writerow(row)
                                dest.flush()
                                print(f"{name} {model_name} {variant} ctx={context} draw={i}"
                                      f" auroc={row['auroc']:.4f} brier={row['brier']:.4f}"
                                      f" ece15={row['ece15']:.4f} sec={seconds:.2f}", flush=True)
    print("Saved exploratory per-draw output:", output)


if __name__ == "__main__":
    main()
