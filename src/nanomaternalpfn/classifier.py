"""Frozen pretrained PFN inference with optional batched class-symmetry and inductive context CV calibration.

v1 default stays unchanged. v2 defaults to single-pass unless explicitly requested
otherwise: data augmentation encourages, but cannot guarantee, class equivariance.
Any temperature/odds parameters are fitted only on the labeled context rows.
"""
from itertools import permutations
import numpy as np
import torch
from .model import NanoMaternalPFN, ModelConfig

class NanoMaternalPFNClassifier:
    def __init__(self, checkpoint, device="cpu", query_batch_size=8,
                 symmetrize_classes=False, calibrate_context=False,
                 calibration_folds=4):
        if query_batch_size is not None and query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        if calibration_folds < 2:
            raise ValueError("calibration_folds must be >=2")
        self.checkpoint = checkpoint
        self.device = device
        self.query_batch_size = query_batch_size
        self.symmetrize_classes = bool(symmetrize_classes)
        self.calibrate_context = bool(calibrate_context)
        self.calibration_folds = int(calibration_folds)
        self.model = None
        self.temperature_ = 1.0
        self.prior_strength_ = 0.0
        self.training_variant_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X and y shapes mismatch")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2 or len(self.classes_) > 3:
            raise ValueError("supports 2 to 3 classes")
        self.y_ = np.searchsorted(self.classes_, y)
        self.X_ = X
        if self.model is None:
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.training_variant_ = ckpt.get("variant", "v1")
            model = NanoMaternalPFN(ModelConfig(**ckpt["config"]))
            model.load_state_dict(ckpt["state_dict"], strict=True)
            self.model = model.to(self.device).eval()
        self.temperature_, self.prior_strength_ = 1.0, 0.0
        if self.calibrate_context:
            self._fit_context_calibration()
        return self

    def _predict_uncalibrated(self, x, ctx_x, ctx_y):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != ctx_x.shape[1]:
            raise ValueError("features mismatch")
        classes = len(self.classes_)
        if len(x) == 0:
            return np.empty((0, classes), dtype=np.float64)
        output = []
        chunks = self.query_batch_size or len(x)
        maps = [np.asarray(p, dtype=np.int64) for p in permutations(range(classes))] if self.symmetrize_classes else [np.arange(classes)]
        maps_tensor = torch.as_tensor(np.stack(maps), device=self.device)
        labels = torch.as_tensor(ctx_y[None], device=self.device).long()
        permuted_labels = maps_tensor[:, labels.squeeze(0)]
        with torch.inference_mode():
            for start in range(0, len(x), chunks):
                xx = np.concatenate([ctx_x, x[start:start + chunks]], axis=0)
                xx = torch.as_tensor(xx[None], device=self.device)
                xx = xx.expand(len(maps), -1, -1)
                p = self.model.predict_proba(xx, permuted_labels, n_classes=classes).cpu().numpy()
                # map original class j back from permuted class maps[k,j]
                aligned = np.stack([p[k][:, m] for k,m in enumerate(maps)], axis=0)
                output.append(aligned.mean(axis=0))
        return np.concatenate(output, axis=0).astype(np.float64)

    def _fit_context_calibration(self):
        """Four-fold cross-fitting, entirely inside the labeled context.

        Each validation fold is predicted by the same frozen pretrained
        model, conditioned only on its three training folds. The learned
        temperature and prevalence-odds weight are shared across queries.
        """
        from sklearn.model_selection import StratifiedKFold
        from scipy.optimize import minimize
        n_classes = len(self.classes_)
        counts = np.bincount(self.y_, minlength=n_classes)
        folds = min(self.calibration_folds, int(counts.min()))
        # Skip rather than leak data or invent calibration when rare classes
        # prevent folds containing all classes.
        if folds < 2 or len(self.y_) < 8:
            return
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20261111)
        probabilities, labels = [], []
        for train, valid in splitter.split(self.X_, self.y_):
            if len(np.unique(self.y_[train])) != n_classes:
                continue
            probabilities.append(self._predict_uncalibrated(
                self.X_[valid], self.X_[train], self.y_[train]
            ))
            labels.append(self.y_[valid])
        if not probabilities:
            return
        probs = np.clip(np.concatenate(probabilities), 1e-7, 1.0)
        truth = np.concatenate(labels)
        prior = (counts + 1.0) / (len(self.y_) + n_classes)
        prior_direction = np.log(prior) - np.log(1. / n_classes)
        # 2 free scalars, but mildly regularized around identity to prevent
        # unstable context-small calibrators. Optimize multiclass log loss.
        def objective(params):
            log_t, beta = params
            z = np.log(probs) / np.exp(log_t) + beta * prior_direction
            z -= z.max(axis=1, keepdims=True)
            ex = np.exp(z)
            log_p = z - np.log(ex.sum(axis=1, keepdims=True))
            return float(-log_p[np.arange(len(truth)),truth].mean()
                         + .01 * log_t ** 2 + .01 * beta ** 2)
        opt = minimize(objective, np.array([0.,0.]), method="L-BFGS-B",
                       bounds=[(-1.6,1.6),(-2.,2.)])
        if opt.success and np.isfinite(opt.x).all():
            self.temperature_ = float(np.exp(opt.x[0]))
            self.prior_strength_ = float(opt.x[1])

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        probs = self._predict_uncalibrated(x, self.X_, self.y_)
        if not len(probs):
            return probs
        if self.calibrate_context:
            c = len(self.classes_)
            prior = (np.bincount(self.y_,minlength=c) + 1.) / (len(self.y_) + c)
            z = np.log(np.clip(probs, 1e-7, 1.)) / self.temperature_
            z += self.prior_strength_ * (np.log(prior) - np.log(1. / c))
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            probs = p / p.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
