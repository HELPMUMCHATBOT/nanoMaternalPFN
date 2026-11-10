"""Sklearn-like inference wrapper for pretrained research checkpoints."""
import numpy as np
import torch
from .model import NanoMaternalPFN, ModelConfig


class NanoMaternalPFNClassifier:
    def __init__(self, checkpoint, device="cpu", query_batch_size=None):
        self.checkpoint = checkpoint
        self.device = device
        if query_batch_size is not None and query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.query_batch_size = query_batch_size
        self.model = None

    def fit(self, X, y):
        """Set a labeled context; model weights remain frozen across contexts."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y shapes mismatch")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2 or len(self.classes_) > 3:
            raise ValueError("supports 2 to 3 classes")
        self.y_ = np.searchsorted(self.classes_, y)
        self.X_ = X
        # Avoid re-reading multi-megabyte weights for every context draw.
        if self.model is None:
            checkpoint = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            self.model = NanoMaternalPFN(ModelConfig(**checkpoint["config"]))
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model.to(self.device).eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.X_.shape[1]:
            raise ValueError("features mismatch")
        if len(x) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        # Each query attends to the labeled context, not to other queries.
        # Small chunks bound peak memory for large real-world test pools.
        chunk_size = self.query_batch_size or len(x)
        y_context = torch.as_tensor(self.y_[None], device=self.device)
        results = []
        for start in range(0, len(x), chunk_size):
            query = x[start:start + chunk_size]
            combined = np.concatenate([self.X_, query], axis=0)
            xx = torch.as_tensor(combined[None], device=self.device)
            probabilities = self.model.predict_proba(
                xx, y_context, n_classes=len(self.classes_)
            )[0]
            results.append(probabilities.cpu().numpy())
        return np.concatenate(results, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
