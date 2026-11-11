"""Optional class/feature symmetrized inference. Does not modify model weights.

Averaging over all C! label permutations guarantees permutation equivariance
(up to floating point error) for fixed, label-independent feature shuffles.
A finite sample of feature shuffles is NOT guaranteed feature-permutation
invariant. Context-prior adjustment is exploratory and uses unlabeled query
features, so it is transductive; it is disabled by default.
"""
from itertools import permutations

import numpy as np
import torch

from .model import ModelConfig, NanoMaternalPFN


def context_prior_adjustment(probabilities, labels, alpha=0.5, smoothing=1.0):
    """Optional unlabeled-query, transductive probability adjustment.

    Adjusts class marginals toward a smoothed labeled-context prevalence.
    No query/test labels are used. alpha=0 preserves probabilities exactly.
    It is NOT guaranteed to improve calibration and should be reported
    separately from nontransductive baselines.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or not len(probs) or probs.shape[1] not in (2, 3):
        raise ValueError("probabilities must be a nonempty [queries, 2|3] array")
    if len(y) == 0 or np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("context labels do not match probability columns")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("invalid probabilities")
    if not (0 <= alpha <= 1) or smoothing <= 0:
        raise ValueError("alpha must be in [0,1] and smoothing must be positive")
    if alpha == 0:
        return probs.copy()

    n_classes = probs.shape[1]
    freq = np.bincount(y, minlength=n_classes)
    prior = (freq + smoothing) / (len(y) + smoothing * n_classes)
    marginal = np.maximum(probs.mean(axis=0), 1e-8)
    logits = np.log(np.maximum(probs, 1e-8))
    logits += alpha * (np.log(prior) - np.log(marginal))
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


class SymmetrizedNanoMaternalPFNClassifier:
    """Fit context and average all C! class-index permutations.

    Query batches are small to avoid quadratic row-attention memory.
    No model retraining or held-out label tuning occurs.
    """
    def __init__(self, checkpoint, *, device="cpu", query_batch_size=8,
                 n_feature_permutations=1, feature_seed=42):
        if query_batch_size < 1 or n_feature_permutations < 1:
            raise ValueError("query batch size and feature permutations must be positive")
        self.checkpoint = checkpoint
        self.device = device
        self.query_batch_size = int(query_batch_size)
        self.n_feature_permutations = int(n_feature_permutations)
        self.feature_seed = int(feature_seed)
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X and y shapes mismatch")
        self.classes_ = np.unique(y)
        if len(self.classes_) not in (2, 3):
            raise ValueError("context must contain all 2 or 3 target classes")
        self.y_ = np.searchsorted(self.classes_, y)
        self.X_ = X
        if self.model is None:
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            model = NanoMaternalPFN(ModelConfig(**ckpt["config"]))
            model.load_state_dict(ckpt["state_dict"], strict=True)
            self.model = model.to(self.device).eval()
        return self

    def _feature_permutations(self):
        d = self.X_.shape[1]
        rng = np.random.default_rng(self.feature_seed)
        return [np.arange(d)] + [
            rng.permutation(d) for _ in range(self.n_feature_permutations - 1)
        ]

    @torch.inference_mode()
    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.X_.shape[1]:
            raise ValueError("feature shapes mismatch")
        n_classes = len(self.classes_)
        if len(x) == 0:
            return np.empty((0, n_classes), dtype=np.float32)
        accum = np.zeros((len(x), n_classes), dtype=np.float64)
        all_class_maps = [np.array(p, dtype=np.int64) for p in permutations(range(n_classes))]
        feature_maps = self._feature_permutations()
        for col_perm in feature_maps:
            context_columns = self.X_[:, col_perm]
            for class_map in all_class_maps:
                y_ctx = torch.as_tensor(class_map[self.y_][None], device=self.device)
                for start in range(0, len(x), self.query_batch_size):
                    stop = min(start + self.query_batch_size, len(x))
                    query_columns = x[start:stop, col_perm]
                    xx = torch.as_tensor(np.concatenate(
                        [context_columns, query_columns], axis=0
                    )[None], device=self.device)
                    predicted = self.model.predict_proba(
                        xx, y_ctx, n_classes=n_classes
                    )[0].cpu().numpy()
                    # Original class j is output column class_map[j].
                    accum[start:stop] += predicted[:, class_map]
        accum /= len(feature_maps) * len(all_class_maps)
        if not np.isfinite(accum).all():
            raise RuntimeError("nonfinite symmetrized probabilities")
        return accum.astype(np.float32)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
