"""Correctness-first query wrapper; no claim of projected KV reuse."""
import torch

class ContextReference:
    def __init__(self, model, context_x, context_y):
        self.model=model.eval()
        self.context_x=context_x
        self.context_y=context_y

    @torch.no_grad()
    def predict(self, query_x, n_classes=None):
        return self.model(torch.cat([self.context_x,query_x],dim=1),self.context_y,n_classes=n_classes)
