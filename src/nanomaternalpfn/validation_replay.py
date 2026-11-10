"""Deterministic held-out synthetic validation for replay-pretrained PFNs.

These tasks are newly generated synthetic tables with a seed distinct from the
training replay. The same validation task draws are used at every evaluation
checkpoint. This is NOT clinical validation or a real-data test set.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate_replay(
    model, replay, *, contexts, device="cpu", batch_size=8,
    batches_per_context=4, query_rows=8, draw_seed=20271008,
):
    """Return context-stratified cross entropy on fixed held-out task draws.

    A fresh RNG with the same seed for each call ensures identical validation
    batches across checkpoints, without consuming the training sampler RNG.
    Each eligible context receives equal weight in the reported overall loss.
    """
    if not contexts or batch_size < 1 or batches_per_context < 1 or query_rows < 1:
        raise ValueError("Invalid synthetic validation configuration")
    rng = np.random.default_rng(draw_seed)
    was_training = model.training
    model.eval()
    losses_by_context = {}
    try:
        for context in contexts:
            context = int(context)
            losses = []
            for _ in range(batches_per_context):
                x, y, n_ctx, n_classes = replay.sample_batch(
                    rng, batch_size=batch_size, context_budgets=(context,),
                    query_rows=query_rows,
                )
                if n_ctx != context:
                    raise RuntimeError("Validation sampler returned an unexpected context")
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                logits = model(x, y[:, :n_ctx], n_classes=n_classes)
                loss = F.cross_entropy(logits.flatten(0, 1), y[:, n_ctx:].flatten())
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("Non-finite held-out synthetic validation loss")
                losses.append(float(loss.item()))
            losses_by_context[str(context)] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return {
        "loss": float(sum(losses_by_context.values()) / len(losses_by_context)),
        "loss_by_context": losses_by_context,
        "batches_per_context": batches_per_context,
        "batch_size": batch_size,
        "task_presentations": len(contexts) * batches_per_context * batch_size,
        "draw_seed": draw_seed,
        "synthetic_only": True,
    }
