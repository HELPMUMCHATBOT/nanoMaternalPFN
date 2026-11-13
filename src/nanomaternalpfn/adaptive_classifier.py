"""Context-only weighted ensemble of two or more frozen PFN checkpoints.

Expert selection and calibration use ONLY labeled context folds. Query records
or query labels never participate in selection. Ensemble accuracy cannot be
attributed to the standalone inference footprint of one checkpoint.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from .classifier import NanoMaternalPFNClassifier


def temperature_scale(probs, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class AdaptiveNanoMaternalPFNClassifier:
    """Compare 2-8 frozen experts by stratified context-only cross-validation.

    weighting='softmax' yields smooth CV-log-loss weights; 'winner' selects one
    expert; 'uniform' is a control. The same frozen task and split can evaluate
    every rule. In small contexts, CV may not be possible; weights then revert
    to uniform rather than using test labels.
    """
    def __init__(
        self, checkpoints, query_batch_size=8, cv_folds=4,
        weight_sharpness=6.0, calibrate=True, estimator_factory=None,
        device="cpu", weighting="softmax",
    ):
        if not 2 <= len(checkpoints) <= 8:
            raise ValueError("expected 2 to 8 pretrained expert checkpoints")
        if cv_folds < 2 or weight_sharpness <= 0:
            raise ValueError("cv_folds >=2 and weight_sharpness >0 required")
        if weighting not in ("softmax", "winner", "uniform"):
            raise ValueError("weighting must be softmax, winner, or uniform")
        self.checkpoints = tuple(Path(p) for p in checkpoints)
        self.query_batch_size = query_batch_size
        self.cv_folds = int(cv_folds)
        self.weight_sharpness = float(weight_sharpness)
        self.calibrate = bool(calibrate)
        self.estimator_factory = estimator_factory
        self.device = device
        self.weighting = weighting
        self.experts_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("labeled context shape mismatch")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes not in (2, 3):
            raise ValueError("expected 2 or 3 context classes")
        n_experts = len(self.checkpoints)
        if self.experts_ is None:
            factory = self.estimator_factory or (
                lambda p: NanoMaternalPFNClassifier(
                    p, device=self.device,
                    query_batch_size=self.query_batch_size,
                )
            )
            self.experts_ = [factory(p) for p in self.checkpoints]
        self.weights_ = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)
        self.temperature_ = 1.0
        self.context_cv_nll_ = [None] * n_experts
        self.context_cv_folds_ = 0
        self.cv_status_ = "insufficient_minority_labels"
        folds = min(self.cv_folds, int(np.bincount(encoded_y).min()))
        if folds >= 2:
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=20261111,
            )
            oof = np.full((n_experts, len(y), n_classes), np.nan, dtype=np.float64)
            for train, valid in splitter.split(X, encoded_y):
                if len(np.unique(y[train])) != n_classes:
                    continue
                for j, expert in enumerate(self.experts_):
                    expert.fit(X[train], y[train])
                    if not np.array_equal(expert.classes_, self.classes_):
                        raise RuntimeError("class mapping changed inside context CV")
                    prediction = np.asarray(expert.predict_proba(X[valid]))
                    if prediction.shape != (len(valid), n_classes):
                        raise RuntimeError("expert returned wrong probability shape")
                    oof[j, valid] = prediction
            if np.isfinite(oof).all() and np.all(oof >= 0):
                clipped = np.clip(oof, 1e-7, 1.0)
                clipped /= clipped.sum(axis=2, keepdims=True)
                nll = -np.log(clipped[:, np.arange(len(y)), encoded_y]).mean(axis=1)
                if self.weighting == "softmax":
                    logits = -self.weight_sharpness * (nll - nll.min())
                    weights = np.exp(logits - logits.max())
                    self.weights_ = weights / weights.sum()
                elif self.weighting == "winner":
                    self.weights_ = np.eye(n_experts)[np.argmin(nll)]
                self.context_cv_nll_ = [float(v) for v in nll]
                self.context_cv_folds_ = folds
                self.cv_status_ = "cross_fitted"
                if self.calibrate:
                    oof_mix = np.einsum("k,knc->nc", self.weights_, clipped)
                    def objective(log_t):
                        calibrated = temperature_scale(oof_mix, np.exp(log_t))
                        chosen = calibrated[np.arange(len(y)), encoded_y]
                        return float(
                            -np.log(np.clip(chosen, 1e-7, 1.0)).mean()
                            + 0.025 * log_t ** 2
                        )
                    opt = minimize_scalar(
                        objective, bounds=(np.log(.5), np.log(3.)),
                        method="bounded",
                    )
                    if opt.success and np.isfinite(opt.fun):
                        self.temperature_ = float(np.exp(opt.x))
            else:
                self.cv_status_ = "invalid_cross_fit"
        # Restore every expert to the complete labeled context, never query data.
        for expert in self.experts_:
            expert.fit(X, y)
        self.diagnostics_ = {
            "context_only": True,
            "cv_status": self.cv_status_,
            "cv_folds": self.context_cv_folds_,
            "weighting": self.weighting,
            "expert_paths": [str(p) for p in self.checkpoints],
            "expert_weights": [float(v) for v in self.weights_],
            "expert_cv_nll": self.context_cv_nll_,
            "temperature": self.temperature_,
            "n_context": len(y),
        }
        if n_experts == 2:
            # Keep the original paper's two-expert diagnostics compatible.
            self.diagnostics_.update({
                "v1_weight": float(self.weights_[0]),
                "v2_weight": float(self.weights_[1]),
                "v1_cv_nll": self.context_cv_nll_[0],
                "v2_cv_nll": self.context_cv_nll_[1],
            })
        return self

    def predict_proba(self, X):
        if self.experts_ is None or not hasattr(self, "diagnostics_"):
            raise ValueError("call fit before predict_proba")
        stacked = np.stack([m.predict_proba(X) for m in self.experts_])
        p = np.einsum("k,knc->nc", self.weights_, stacked)
        if self.calibrate and self.cv_status_ == "cross_fitted":
            p = temperature_scale(p, self.temperature_)
        return p.astype(np.float32)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
