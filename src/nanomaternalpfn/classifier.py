"""Sklearn-like inference wrapper for pretrained research checkpoints."""
from dataclasses import dataclass
import numpy as np
import torch
from .model import NanoMaternalPFN, ModelConfig

class NanoMaternalPFNClassifier:
    def __init__(self, checkpoint, device='cpu'):
        self.checkpoint = checkpoint
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError('X and y shapes mismatch')
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2 or len(self.classes_) > 3:
            raise ValueError('supports 2 to 3 classes')
        self.y_ = np.searchsorted(self.classes_, y)
        self.X_ = X
        checkpoint = torch.load(self.checkpoint,map_location=self.device,weights_only=True)
        self.model = NanoMaternalPFN(ModelConfig(**checkpoint['config']))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.X_.shape[1]:
            raise ValueError('features mismatch')
        combined = np.concatenate([self.X_,x],axis=0)
        xx = torch.as_tensor(combined[None],device=self.device)
        yy = torch.as_tensor(self.y_[None],device=self.device)
        return self.model.predict_proba(xx,yy,n_classes=len(self.classes_))[0].cpu().numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
