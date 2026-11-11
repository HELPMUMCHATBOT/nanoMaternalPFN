#!/usr/bin/env python3
"""Read-only, training-split-only diagnostics of pretrained PFN checkpoints.

Usage from nanoMaternalPFN root:
    python scripts/10_diagnose_prior_failures.py

Never edits frozen manifests, original checkpoints or downstream_full results.
This script DOES NOT use held-out test labels for choosing a class mapping.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from nanomaternalpfn.model import ModelConfig, NanoMaternalPFN

ARMS = ("tiny_generic", "tiny_mixture", "base_generic", "base_mixture")
RISK_ORDER = ("low risk", "mid risk", "high risk")


def semantic_mapping(classes):
    """Map original categorical IDs into prespecified ascending risk IDs."""
    if {x.lower() for x in classes} != set(RISK_ORDER):
        raise ValueError("Semantic mapping only defined for the three UCI risk levels")
    return np.array([RISK_ORDER.index(x.lower()) for x in classes], dtype=np.int64)


def restore_columns(predicted, original_to_new):
    """Original class j lives in new output column original_to_new[j]."""
    return predicted[:, original_to_new]


def batched_predict(model, context_x, context_y, query_x, n_classes, batch_size):
    rows = []
    yy = torch.as_tensor(context_y[None], dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(query_x), batch_size):
            chunk = query_x[start : start + batch_size]
            all_x = np.concatenate([context_x, chunk], axis=0).astype(np.float32)
            xx = torch.from_numpy(all_x[None])
            p = model.predict_proba(xx, yy, n_classes=n_classes)[0].numpy()
            rows.append(p)
    return np.concatenate(rows)


def maybe_auc(y, p):
    try:
        if len(np.unique(y)) != p.shape[1]:
            return None
        auc = roc_auc_score(
            y, p[:, 1] if p.shape[1] == 2 else p,
            multi_class="ovr",
        )
        return round(float(auc), 6)
    except ValueError:
        return None


def check_manifest(manifest_dir, name):
    path = manifest_dir / (name + ".json")
    manifest = json.loads(path.read_text())
    npz = path.with_suffix(".npz")
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Frozen split checksum mismatch for " + name)
    with np.load(npz, allow_pickle=False) as arrays:
        X, y = arrays["X"].copy(), arrays["y"].copy()
        train = arrays["train_idx"].copy()
    return manifest, X, y.astype(np.int64), train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--output", type=Path, default=Path("artifacts/results/diagnostics/prior_failures.json"))
    p.add_argument("--datasets", nargs="+", default=["uci_maternal_risk", "low_birth_weight"])
    p.add_argument("--contexts", type=int, nargs="+", default=[16, 64])
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--queries", type=int, default=48)
    p.add_argument("--query-batch-size", type=int, default=8)
    a = p.parse_args()
    if min(a.draws, a.queries, a.query_batch_size) < 1:
        p.error("draws, queries, query-batch-size must all be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    portfolio = json.loads((a.manifest_dir / "open_portfolio_report.json").read_text())
    models = {}
    for arm in ARMS:
        file = a.checkpoint_dir / f"{arm}.best.inference.pt"
        cp = torch.load(file, map_location="cpu", weights_only=True)
        model = NanoMaternalPFN(ModelConfig(**cp["config"]))
        model.load_state_dict(cp["state_dict"], strict=True)
        models[arm] = (model.eval(), cp.get("step"))

    results = {
        "scope": "training-only exploratory diagnosis; no held-out test labels used",
        "models": {arm: {"checkpoint_step": step} for arm, (_, step) in models.items()},
        "results": [],
    }
    for dataset in a.datasets:
        manifest, X, y, train = check_manifest(a.manifest_dir, dataset)
        classes = portfolio["dataset_results"][dataset]["source"]["classes"]
        n_classes = len(np.unique(y))
        risk_map = semantic_mapping(classes) if dataset == "uci_maternal_risk" else None
        for budget in a.contexts:
            draws = manifest["context_draws"].get(str(budget), [])[: a.draws]
            for draw_idx, draw in enumerate(draws):
                ctx = np.asarray(draw, dtype=np.int64)
                # Training rows outside the labeled context, never test rows.
                # Avoid evaluating a duplicate feature profile of a context row.
                remaining = train[~np.isin(train, ctx)]
                ctx_x = X[ctx]
                is_duplicate = np.any(
                    np.all(np.isclose(X[remaining, None, :], ctx_x[None, :, :],
                                      rtol=0, atol=0, equal_nan=True), axis=2),
                    axis=1,
                )
                queries = remaining[~is_duplicate][: a.queries]
                if not len(queries):
                    continue
                cx, cy, qx, qy = X[ctx], y[ctx], X[queries], y[queries]
                for arm, (model, _) in models.items():
                    p0 = batched_predict(model, cx, cy, qx, n_classes, a.query_batch_size)
                    permutation_errors = []
                    for perm in itertools.permutations(range(n_classes)):
                        if perm == tuple(range(n_classes)):
                            continue
                        mapping = np.asarray(perm, dtype=np.int64)
                        pp = batched_predict(
                            model, cx, mapping[cy], qx, n_classes, a.query_batch_size
                        )
                        permutation_errors.append(
                            float(np.mean(np.abs(restore_columns(pp, mapping) - p0)))
                        )
                    row = {
                        "dataset": dataset, "method": arm,
                        "context": budget, "draw": draw_idx,
                        "n_training_only_queries": len(queries),
                        "prob_std_per_class": np.std(p0, axis=0).round(8).tolist(),
                        "mean_prob_std": round(float(np.std(p0, axis=0).mean()), 8),
                        "mean_probabilities": np.mean(p0, axis=0).round(6).tolist(),
                        "average_permutation_equivariance_error": round(float(np.mean(permutation_errors)), 6),
                        "original_training_only_auc": maybe_auc(qy, p0),
                    }
                    if risk_map is not None:
                        p_semantic = batched_predict(
                            model, cx, risk_map[cy], qx,
                            n_classes, a.query_batch_size,
                        )
                        unpermuted = restore_columns(p_semantic, risk_map)
                        row["original_classes"] = classes
                        row["semantic_original_to_new"] = risk_map.tolist()
                        row["semantic_training_only_auc"] = maybe_auc(qy, unpermuted)
                        row["semantic_probability_shift_mae"] = round(
                            float(np.mean(np.abs(unpermuted - p0))), 6
                        )
                    results["results"].append(row)
                    print(
                        f"{dataset} {arm} ctx={budget} draw={draw_idx} "
                        f"pstd={row['mean_prob_std']:.6f} "
                        f"perm_mae={row['average_permutation_equivariance_error']:.4f} "
                        f"auc={row['original_training_only_auc']} "
                        f"semantic_auc={row.get('semantic_training_only_auc')}",
                        flush=True,
                    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2) + "\n")
    print("WROTE", a.output)


if __name__ == "__main__":
    main()
