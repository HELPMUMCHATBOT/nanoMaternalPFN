"""Leakage-safe low-data context sweeps using frozen held-out indices.

The same context draws and fixed test rows are reused for all methods.
No preprocessing or model weights are fitted on the test set.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from .baselines import get_baseline
from .calibration import multiclass_brier, ece
from ..classifier import NanoMaternalPFNClassifier
from ..adaptive_classifier import AdaptiveNanoMaternalPFNClassifier


FIELDS = ("method", "context", "seed", "accuracy", "brier", "ece15", "auroc")


def evaluate(
    manifest_path, checkpoint=None, methods=("logistic", "random_forest"),
    output="artifacts/results/transfer.csv", query_batch_size=16,
    max_draws=None, contexts=None, pfn_kwargs=None,
    adaptive_checkpoints=None, adaptive_kwargs=None, baseline_kwargs=None,
    resume=False,
):
    """Evaluate all applicable frozen contexts against one fixed test pool.

    max_draws is a pilot-only cap; omit it for all 20 prespecified draws.
    3-class AUROC can be undefined if the held-out pool lacks a class; record NaN.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    npz = manifest_path.with_suffix(".npz")
    if hashlib.sha256(npz.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("frozen dataset SHA256 mismatch")
    data = np.load(npz, allow_pickle=False)
    X, y = data["X"], data["y"]
    test = np.asarray(manifest["test_idx"], dtype=int)
    classes = np.unique(y)
    if len(classes) not in (2, 3):
        raise ValueError("supports 2 or 3 classes")
    if len(test) == 0:
        raise ValueError("empty frozen test set")
    if max_draws is not None and max_draws < 1:
        raise ValueError("max_draws must be positive")
    selected_contexts = {int(c) for c in contexts} if contexts is not None else None
    results = []
    skipped = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if resume and output.exists():
        with output.open(newline="") as previous:
            reader = csv.DictReader(previous)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected resume CSV schema: {output}")
            for old in reader:
                row = {}
                for field in FIELDS:
                    row[field] = (int(old[field]) if field in ("context", "seed")
                                  else float(old[field]) if field in
                                  ("auroc", "accuracy", "brier", "ece15")
                                  else old[field])
                key = (row["method"], row["context"], row["seed"])
                if key in existing_keys:
                    raise ValueError(f"Duplicate row in {output}: {key}")
                existing_keys.add(key)
                results.append(row)
    mode = "a" if resume and output.exists() else "w"
    with output.open(mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for method in methods:
            if method == "pfn" and not checkpoint:
                raise ValueError("pretrained checkpoint required for PFN comparison")
            if method == "adaptive" and not adaptive_checkpoints:
                raise ValueError("adaptive method needs 2-8 checkpoints")
            pfn = (NanoMaternalPFNClassifier(
                checkpoint, query_batch_size=query_batch_size,
                **(pfn_kwargs or {})
            ) if method == "pfn" else None)
            adaptive = (AdaptiveNanoMaternalPFNClassifier(
                adaptive_checkpoints, query_batch_size=query_batch_size,
                **(adaptive_kwargs or {})
            ) if method == "adaptive" else None)
            for budget, all_draws in sorted(
                manifest["context_draws"].items(), key=lambda kv: int(kv[0])
            ):
                if selected_contexts is not None and int(budget) not in selected_contexts:
                    continue
                draws = all_draws[:max_draws] if max_draws else all_draws
                for seed, draw in enumerate(draws):
                    if (method, int(budget), seed) in existing_keys:
                        continue
                    draw = np.asarray(draw, dtype=int)
                    if len(np.unique(y[draw])) < 2:
                        skipped.append({
                            "method": method, "context": int(budget),
                            "seed": seed, "reason": "single_context_class",
                        })
                        continue
                    estimator = (pfn if pfn is not None else
                                 adaptive if adaptive is not None else
                                 get_baseline(method, seed, **(baseline_kwargs or {})))
                    estimator.fit(X[draw], y[draw])
                    if adaptive is not None:
                        diag_path = output.with_suffix(".adaptive_diagnostics.jsonl")
                        with diag_path.open("a") as handle:
                            handle.write(json.dumps({
                                "dataset": manifest_path.stem,
                                "context": int(budget), "seed": seed,
                                **estimator.diagnostics_,
                            }, sort_keys=True) + "\n")
                    pred = estimator.predict_proba(X[test])
                    aligned = np.zeros((len(test), len(classes)))
                    for j, c in enumerate(estimator.classes_):
                        aligned[:, np.where(classes == c)[0][0]] = pred[:, j]
                    # Classification probabilities must be finite and normalized.
                    if not np.isfinite(aligned).all() or not np.allclose(
                        aligned.sum(axis=1), 1.0, atol=1e-5
                    ):
                        raise RuntimeError(f"Invalid predicted probabilities: {method}")
                    truth = np.searchsorted(classes, y[test])
                    scores = {
                        "method": method, "context": int(budget), "seed": seed,
                        "accuracy": float(accuracy_score(
                            y[test], classes[aligned.argmax(axis=1)]
                        )),
                        "brier": multiclass_brier(truth, aligned),
                        "ece15": ece(truth, aligned),
                    }
                    try:
                        scores["auroc"] = float(roc_auc_score(
                            truth, aligned[:, 1] if len(classes) == 2 else aligned,
                            multi_class="ovr",
                        ))
                    except ValueError:
                        scores["auroc"] = float("nan")
                    writer.writerow(scores)
                    file.flush()  # A partial run retains its completed observations.
                    results.append(scores)
                    print(f"{manifest_path.stem} {method} context={budget} "
                          f"draw={seed + 1}/{len(draws)} "
                          f"auroc={scores['auroc']:.4f} "
                          f"brier={scores['brier']:.4f}", flush=True)
    if skipped:
        output.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=2))
    return results
