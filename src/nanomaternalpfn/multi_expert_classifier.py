"""Context-only multi-expert PFN: no test/query labels in model selection.

This generalization of the existing two-checkpoint adaptive wrapper is opt-in.
The trained checkpoint weights are never updated. Context-only cross-fitting
sets convex expert weights using log loss, not held-out test AUROC.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from .adaptive_classifier import temperature_scale
from .classifier import NanoMaternalPFNClassifier


class ContextOnlyMultiExpertPFN:
    """Select or average >=2 frozen PFN experts using labeled-context CV."""

    def __init__(self, checkpoints, query_batch_size=8, cv_folds=4,
                 weight_sharpness=6.0, calibrate=True, estimator_factory=None,
                 device="cpu"):
        if len(checkpoints) < 2:
            raise ValueError("at least two pretrained checkpoints required")
        if cv_folds < 2 or weight_sharpness <= 0:
            raise ValueError("cv_folds>=2 and weight_sharpness>0 required")
        self.checkpoints = tuple(Path(x) for x in checkpoints)
        self.query_batch_size = int(query_batch_size)
        self.cv_folds = int(cv_folds)
        self.weight_sharpness = float(weight_sharpness)
        self.calibrate = bool(calibrate)
        self.estimator_factory = estimator_factory
        self.device = device
        self.experts_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("invalid labeled context shape")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes not in (2, 3):
            raise ValueError("only 2 or 3 observed context classes supported")
        n = len(self.checkpoints)
        if self.experts_ is None:
            factory = self.estimator_factory or (
                lambda p: NanoMaternalPFNClassifier(
                    p, device=self.device, query_batch_size=self.query_batch_size)
            )
            self.experts_ = [factory(p) for p in self.checkpoints]
        self.weights_ = np.full(n, 1.0 / n, dtype=np.float64)
        self.temperature_ = 1.0
        self.context_cv_nll_ = [None] * n
        self.context_cv_folds_ = 0
        self.cv_status_ = "insufficient_minority_labels"
        folds = min(self.cv_folds, int(np.bincount(encoded_y).min()))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True,
                                 random_state=20261111)
            out_of_fold = np.full((n, len(y), n_classes), np.nan, dtype=np.float64)
            for train, valid in cv.split(X, encoded_y):
                for j, expert in enumerate(self.experts_):
                    expert.fit(X[train], y[train])
                    if not np.array_equal(expert.classes_, self.classes_):
                        raise RuntimeError("expert changed context class order")
                    probs = np.asarray(expert.predict_proba(X[valid]), dtype=float)
                    if probs.shape != (len(valid), n_classes):
                        raise ValueError("invalid context CV probability shape")
                    if (not np.isfinite(probs).all()
                        or np.any(probs < 0)
                        or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
                        raise ValueError("invalid context CV probabilities")
                    out_of_fold[j, valid] = probs
            if np.isfinite(out_of_fold).all():
                clipped = np.clip(out_of_fold, 1e-7, 1.0)
                losses = -np.log(
                    clipped[:, np.arange(len(y)), encoded_y]
                ).mean(axis=1)
                scores = -self.weight_sharpness * (losses - losses.min())
                scores -= scores.max()
                weights = np.exp(scores)
                self.weights_ = weights / weights.sum()
                self.context_cv_nll_ = [float(v) for v in losses]
                self.context_cv_folds_ = folds
                self.cv_status_ = "cross_fitted"
                if self.calibrate:
                    mixed = np.einsum("k,knc->nc", self.weights_, clipped)

                    def objective(log_temp):
                        probabilities = temperature_scale(mixed, np.exp(log_temp))
                        selected = probabilities[np.arange(len(y)), encoded_y]
                        return (float(-np.log(np.clip(selected, 1e-7, 1)).mean())
                                + 0.025 * log_temp ** 2)

                    result = minimize_scalar(objective,
                        bounds=(np.log(0.5), np.log(3.0)), method="bounded")
                    if result.success and np.isfinite(result.fun):
                        self.temperature_ = float(np.exp(result.x))
            else:
                self.cv_status_ = "invalid_cross_fit"
        # Re-condition experts using the entire labeled context after CV.
        for expert in self.experts_:
            expert.fit(X, y)
            if not np.array_equal(expert.classes_, self.classes_):
                raise RuntimeError("expert changed final class order")
        self.diagnostics_ = {
            "context_only": True,
            "n_experts": n,
            "expert_checkpoints": [str(x) for x in self.checkpoints],
            "expert_weights": [float(x) for x in self.weights_],
            "expert_cv_nll": self.context_cv_nll_,
            "cv_folds": self.context_cv_folds_,
            "cv_status": self.cv_status_,
            "temperature": self.temperature_,
            "n_context": len(y),
            "exploratory": True,
        }
        return self

    def predict_proba(self, X):
        if self.experts_ is None or not hasattr(self, "diagnostics_"):
            raise ValueError("call fit before predict_proba")
        parts = [np.asarray(expert.predict_proba(X), dtype=float)
                 for expert in self.experts_]
        if len({part.shape for part in parts}) != 1:
            raise ValueError("expert prediction shapes differ")
        mixed = np.einsum("k,knc->nc", self.weights_, np.stack(parts))
        if self.calibrate and self.cv_status_ == "cross_fitted":
            mixed = temperature_scale(mixed, self.temperature_)
        if (not np.isfinite(mixed).all()
            or not np.allclose(mixed.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("invalid adaptive prediction probabilities")
        return mixed.astype(np.float32)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
