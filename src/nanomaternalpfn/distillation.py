"""C4: Synthetic-only, resumable PFN-ensemble-to-Tiny distillation.

Never equate a best-so-far partial checkpoint with a completed student.
Resume verifies exact teacher/replay hashes and all training hyperparameters.
Only actually executed optimizer steps are reported.
"""
from __future__ import annotations

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
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    model = NanoMaternalPFN(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def distill(
    replay_path, teacher_paths, output, *, steps=2500, batch_size=32,
    microbatch_size=8, max_context=512, query_rows=8,
    alpha=.5, temperature=2.0, lr=5e-4, seed=20261113,
    device="cpu", validation_path=None, validate_every=250,
    feature_encoder_type="gated", normalization="robust",
    resume=False, checkpoint_every=25,
):
    if not teacher_paths or any(not Path(p).is_file() for p in teacher_paths):
        raise ValueError("all teacher checkpoints must exist")
    if not 0 <= alpha <= 1 or temperature <= 0 or steps < 1 or lr <= 0:
        raise ValueError("invalid distillation hyperparameters")
    if batch_size < 1 or microbatch_size < 1 or batch_size % microbatch_size:
        raise ValueError("microbatch_size must divide effective batch size")
    if checkpoint_every < 1 or validate_every < 1 or query_rows < 1:
        raise ValueError("checkpoint, validation intervals and query rows must be positive")
    if max_context < 16:
        raise ValueError("max_context must be >=16")
    if validation_path is not None and (
        Path(validation_path).resolve() == Path(replay_path).resolve()
        or file_sha256(validation_path) == file_sha256(replay_path)
    ):
        raise ValueError("synthetic validation replay must be disjoint from training")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA accelerator is unavailable")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output
    progress_path = output.with_name(output.stem + ".training.pt")
    best_path = output.with_name(output.stem + ".best.inference.pt")
    report_path = output.with_suffix(".json")
    teacher_hashes = [file_sha256(p) for p in teacher_paths]
    replay_hash = file_sha256(replay_path)
    validation_hash = file_sha256(validation_path) if validation_path else None
    cfg = replace(
        TIERS["tiny"], architecture="pre_ln_v2",
        feature_encoder_type=feature_encoder_type, normalization=normalization,
    )
    budgets = [c for c in (16,32,64,128,256,512) if c <= max_context]
    settings = {
        "schema_version":2,
        "config":asdict(cfg),
        "teacher_checkpoint_sha256":teacher_hashes,
        "training_replay_sha256":replay_hash,
        "validation_replay_sha256":validation_hash,
        "steps_requested":int(steps),
        "batch_size":int(batch_size),
        "microbatch_size":int(microbatch_size),
        "max_context":int(max_context),
        "query_rows":int(query_rows),
        "alpha":float(alpha),
        "temperature":float(temperature),
        "lr":float(lr),
        "seed":int(seed),
        "validate_every":int(validate_every),
        "checkpoint_every":int(checkpoint_every),
    }
    if resume and final_checkpoint.exists() and report_path.exists():
        old = json.loads(report_path.read_text())
        if old.get("steps_completed") == steps and old.get("settings") == settings:
            if validation_hash and not best_path.is_file():
                raise RuntimeError("completed student missing its selected inference checkpoint")
            return old
    if not resume and (final_checkpoint.exists() or progress_path.exists()):
        raise FileExistsError("student output already exists; use resume=True")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    teachers = [load_teacher(p, device) for p in teacher_paths]
    student = NanoMaternalPFN(cfg).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    history = []
    best_loss = float("inf")
    best_step = None
    start = 0
    if resume and progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        if saved.get("settings") != settings:
            raise ValueError("cannot resume distillation with different teachers, archives or hyperparameters")
        student.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["np_rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        history = list(saved["validation_history"])
        best_loss = float(saved["best_validation_loss"])
        best_step = saved["best_validation_step"]
        start = int(saved["step"])
        if start < 0 or start > steps:
            raise ValueError("invalid saved optimizer step")
        if best_step is not None and not best_path.is_file():
            raise RuntimeError("validation-best weights missing from resumed distillation")
    elif resume and final_checkpoint.exists():
        raise RuntimeError("student checkpoint exists without matching completed report or training state")

    from contextlib import nullcontext
    val_cm = HDF5TaskReplay(validation_path) if validation_path else nullcontext(None)
    with val_cm as validation, HDF5TaskReplay(replay_path) as replay:
        for step in range(start+1, steps+1):
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
                    soft_targets = torch.stack([
                        torch.softmax(t(x, y[:, :n_ctx], n_classes=classes) / temperature, -1)
                        for t in teachers
                    ]).mean(0)
                logits = student(x, y[:, :n_ctx], n_classes=classes)
                truth = y[:, n_ctx:]
                supervised = F.cross_entropy(logits.reshape(-1, classes), truth.reshape(-1))
                softened = F.log_softmax(logits / temperature, dim=-1)
                transfer = F.kl_div(
                    softened, soft_targets, reduction="batchmean"
                ) / truth.shape[1] * temperature ** 2
                loss = alpha * supervised + (1-alpha) * transfer
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite distillation loss")
                (loss * microbatch_size / batch_size).backward()
                micro_losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            measured_loss = float(np.mean(micro_losses))
            if validation is not None and (step % validate_every == 0 or step == steps):
                metrics = evaluate_replay(
                    student, validation, contexts=budgets, device=device,
                    batch_size=2, batches_per_context=2,
                    query_rows=query_rows, draw_seed=seed + 100,
                )
                history.append({
                    "step":step, "loss":metrics["loss"], "training_loss":measured_loss,
                })
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    best_step = step
                    _atomic_save({
                        "state_dict":student.state_dict(), "config":asdict(cfg),
                        "step":step, "synthetic_only":True, "distilled":True,
                    }, best_path)
            if step % checkpoint_every == 0 or step == steps:
                _atomic_save({
                    "settings":settings, "state_dict":student.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "np_rng_state":rng.bit_generator.state,
                    "torch_rng_state":torch.get_rng_state(),
                    "cuda_rng_states":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "step":step, "validation_history":history,
                    "best_validation_loss":best_loss,
                    "best_validation_step":best_step,
                }, progress_path)

    if not progress_path.exists():
        raise RuntimeError("distillation optimizer progress not saved")
    _atomic_save({
        "state_dict":student.state_dict(), "config":asdict(cfg),
        "step":steps, "synthetic_only":True, "distilled":True,
    }, final_checkpoint)
    report = {
        "steps_completed":steps, "effective_batch_size":batch_size,
        "task_presentations":steps*batch_size,
        "student_parameters":student.parameter_count(),
        "teacher_checkpoint_sha256":teacher_hashes,
        "training_replay_sha256":replay_hash,
        "validation_replay_sha256":validation_hash,
        "validation_history":history, "best_validation_step":best_step,
        "alpha":alpha, "temperature":temperature,
        "synthetic_only":True, "clinical_auroc_measured":False,
        "settings":settings, "training_state_path":str(progress_path),
    }
    tmp = report_path.with_name(report_path.name+".tmp")
    tmp.write_text(json.dumps(report,indent=2)+"\n")
    tmp.replace(report_path)
    return report
