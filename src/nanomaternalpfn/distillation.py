"""C4: Synthetic-only PFN ensemble-to-Tiny distillation.

Training targets use ground-truth synthetic labels plus frozen teacher soft
probabilities. No real clinical rows or query labels enter expert selection.
Each saved checkpoint records teacher hashes and measured optimizer steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import ModelConfig, NanoMaternalPFN
from .priors.hdf5_replay import HDF5TaskReplay, file_sha256
from .train_replay import TIERS
from .validation_replay import evaluate_replay


def load_teacher(path, device="cpu"):
    ckpt = torch.load(path, weights_only=True, map_location=device)
    model = NanoMaternalPFN(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def distill(
    replay_path, teacher_paths, output, *, steps=2500, batch_size=32,
    microbatch_size=8, max_context=512, query_rows=8,
    alpha=.5, temperature=2.0, lr=5e-4, seed=20261113,
    device="cpu", validation_path=None, validate_every=250,
    feature_encoder_type="gated", normalization="robust",
):
    if not teacher_paths or any(not Path(p).is_file() for p in teacher_paths):
        raise ValueError("all teacher checkpoints must exist")
    if not 0 <= alpha <= 1 or temperature <= 0 or steps < 1:
        raise ValueError("invalid distillation hyperparameters")
    if batch_size < 1 or microbatch_size < 1 or batch_size % microbatch_size:
        raise ValueError("microbatch must divide effective batch size")
    if validation_path is not None and (
        Path(validation_path).resolve() == Path(replay_path).resolve()
        or file_sha256(validation_path) == file_sha256(replay_path)
    ):
        raise ValueError("synthetic validation replay must be disjoint from training")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    teachers = [load_teacher(p, device) for p in teacher_paths]
    cfg = replace(
        TIERS["tiny"], architecture="pre_ln_v2",
        feature_encoder_type=feature_encoder_type, normalization=normalization,
    )
    student = NanoMaternalPFN(cfg).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    budgets = [c for c in (16,32,64,128,256,512) if c <= max_context]
    if not budgets:
        raise ValueError("max_context must be >=16")
    history = []
    best_loss = float("inf")
    best_step = None
    from contextlib import nullcontext
    val_cm = HDF5TaskReplay(validation_path) if validation_path else nullcontext(None)
    with val_cm as validation, HDF5TaskReplay(replay_path) as replay:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(batch_size // microbatch_size):
                x, y, n_ctx, classes = replay.sample_batch(
                    rng, batch_size=microbatch_size,
                    context_budgets=budgets, query_rows=query_rows,
                )
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                with torch.no_grad():
                    teacher_probs = torch.stack([
                        torch.softmax(t(x, y[:, :n_ctx], n_classes=classes) / temperature, -1)
                        for t in teachers
                    ]).mean(0)
                logits = student(x, y[:, :n_ctx], n_classes=classes)
                truth = y[:, n_ctx:]
                supervised = F.cross_entropy(
                    logits.reshape(-1, classes), truth.reshape(-1)
                )
                softened = F.log_softmax(logits / temperature, dim=-1)
                transfer = F.kl_div(
                    softened, teacher_probs, reduction="batchmean"
                ) / truth.shape[1] * temperature ** 2
                loss = alpha * supervised + (1-alpha) * transfer
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite distillation loss")
                (loss * microbatch_size / batch_size).backward()
                micro_losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            measured_loss = float(np.mean(micro_losses))
            if validation is not None and (
                step % validate_every == 0 or step == steps
            ):
                summary = evaluate_replay(
                    student, validation, contexts=budgets, device=device,
                    batch_size=2, batches_per_context=2,
                    query_rows=query_rows, draw_seed=seed + 100,
                )
                history.append({"step":step, "loss":summary["loss"],
                                "training_loss":measured_loss})
                if summary["loss"] < best_loss:
                    best_loss = summary["loss"]
                    best_step = step
                    best_path = output.with_name(output.stem+".best.inference.pt")
                    torch.save({"state_dict":student.state_dict(),
                                "config":asdict(cfg), "step":step,
                                "synthetic_only":True, "distilled":True}, best_path)
            if step == steps:
                torch.save({"state_dict":student.state_dict(),
                            "config":asdict(cfg), "step":step,
                            "synthetic_only":True, "distilled":True}, output)
    report = {
        "steps_completed":steps, "effective_batch_size":batch_size,
        "task_presentations":steps * batch_size,
        "student_parameters":student.parameter_count(),
        "teacher_checkpoint_sha256":[file_sha256(p) for p in teacher_paths],
        "training_replay_sha256":file_sha256(replay_path),
        "validation_replay_sha256":file_sha256(validation_path) if validation_path else None,
        "validation_history":history, "best_validation_step":best_step,
        "alpha":alpha, "temperature":temperature, "synthetic_only":True,
        "clinical_auroc_measured":False,
    }
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    return report
