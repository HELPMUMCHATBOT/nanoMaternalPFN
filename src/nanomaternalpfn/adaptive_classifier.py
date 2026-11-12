"""Context-only ensemble of pretrained v1 and v2 maternal PFN experts.

No test/query data or labels determine weights or temperature. This is a
two-checkpoint ensemble, not a single tiny PFN with unchanged latency/RAM.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from .classifier import NanoMaternalPFNClassifier


def temperature_scale(probs, temperature):
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class AdaptiveNanoMaternalPFNClassifier:
    def __init__(self, checkpoints, query_batch_size=8, cv_folds=4,
                 weight_sharpness=6.0, calibrate=True, estimator_factory=None,
                 device='cpu'):
        if len(checkpoints) != 2:
            raise ValueError("expected exactly v1 and v2 checkpoints")
        if cv_folds < 2 or weight_sharpness <= 0:
            raise ValueError("cv_folds >=2 and weight_sharpness >0 required")
        self.checkpoints = tuple(Path(p) for p in checkpoints)
        self.query_batch_size = query_batch_size
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
            raise ValueError("labeled context shape mismatch")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes not in (2, 3):
            raise ValueError("expected 2 or 3 context classes")
        if self.experts_ is None:
            factory = self.estimator_factory or (
                lambda p: NanoMaternalPFNClassifier(
                    p, device=self.device, query_batch_size=self.query_batch_size
                )
            )
            self.experts_ = [factory(p) for p in self.checkpoints]
        self.weights_ = np.array([0.5, 0.5], dtype=np.float64)
        self.temperature_ = 1.0
        self.context_cv_nll_ = [None, None]
        self.context_cv_folds_ = 0
        self.cv_status_ = "insufficient_minority_labels"
        folds = min(self.cv_folds, int(np.bincount(encoded_y).min()))
        if folds >= 2:
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=20261111
            )
            oof = np.full((2, len(y), n_classes), np.nan, dtype=np.float64)
            for train, valid in splitter.split(X, encoded_y):
                if len(np.unique(y[train])) != n_classes:
                    continue
                for j, expert in enumerate(self.experts_):
                    expert.fit(X[train], y[train])
                    if not np.array_equal(expert.classes_, self.classes_):
                        raise RuntimeError("class mapping changed inside context CV")
                    oof[j, valid] = expert.predict_proba(X[valid])
            if np.isfinite(oof).all():
                clipped = np.clip(oof, 1e-7, 1.0)
                nll = -np.log(clipped[:, np.arange(len(y)), encoded_y]).mean(axis=1)
                logits = -self.weight_sharpness * (nll - nll.min())
                w = np.exp(logits - logits.max())
                self.weights_ = w / w.sum()
                self.context_cv_nll_ = [float(v) for v in nll]
                self.context_cv_folds_ = folds
                self.cv_status_ = "cross_fitted"
                if self.calibrate:
                    oof_mix = np.einsum("k,knc->nc", self.weights_, clipped)
                    def objective(log_t):
                        calibrated = temperature_scale(oof_mix, np.exp(log_t))
                        chosen = calibrated[np.arange(len(y)), encoded_y]
                        return float(-np.log(np.clip(chosen, 1e-7, 1.0)).mean()
                                     + 0.025 * log_t ** 2)
                    opt = minimize_scalar(
                        objective, bounds=(np.log(.5), np.log(3.)),
                        method="bounded",
                    )
                    if opt.success and np.isfinite(opt.fun):
                        self.temperature_ = float(np.exp(opt.x))
            else:
                self.cv_status_ = "invalid_cross_fit"
        # Experts condition on all and only labeled context rows for prediction.
        for expert in self.experts_:
            expert.fit(X, y)
        self.diagnostics_ = {
            "context_only": True,
            "cv_status": self.cv_status_,
            "cv_folds": self.context_cv_folds_,
            "v1_weight": float(self.weights_[0]),
            "v2_weight": float(self.weights_[1]),
            "v1_cv_nll": self.context_cv_nll_[0],
            "v2_cv_nll": self.context_cv_nll_[1],
            "temperature": self.temperature_,
            "n_context": len(y),
        }
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
