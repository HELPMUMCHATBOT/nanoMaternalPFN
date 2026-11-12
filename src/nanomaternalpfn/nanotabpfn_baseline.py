"""Upstream nanoTabPFN reference architecture with context-only preprocessing.

Code for NanoTabPFNModel is imported verbatim from the pinned upstream
automl/nanoTabPFN source. We retrain its 3-output model on our generic v2
synthetic prior with the same 5000-step/32-task budget as nanoMaternalPFN.
This is NOT an official pretrained nanoTabPFN checkpoint.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from .third_party.nanotabpfn.model import NanoTabPFNModel
from .priors.hdf5_replay import create_replay, HDF5TaskReplay


@dataclass
class NanoTabConfig:
    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 3
    num_outputs: int = 3


def new_nanotab(config=None):
    config = config or NanoTabConfig()
    if isinstance(config, dict):
        config = NanoTabConfig(**config)
    return NanoTabPFNModel(**asdict(config))


def impute_from_context(x: torch.Tensor, n_context: int) -> torch.Tensor:
    """Impute non-finite entries using only labeled-context statistics.

    Input x: batch × (labeled context + query) × features.
    No query values can influence context means.
    """
    context = x[:, :n_context, :]
    valid = torch.isfinite(context)
    means = torch.where(valid, context, 0.0).sum(
        dim=1, keepdim=True
    ) / valid.sum(dim=1, keepdim=True).clamp(min=1)
    return torch.where(torch.isfinite(x), x, means)


class NanoTabPFNReferenceClassifier:
    """Frozen upstream architecture, context-conditioned 2/3-class inference."""
    def __init__(self, checkpoint, device="cpu", query_batch_size=8):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.query_batch_size = int(query_batch_size)
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
            raise ValueError("context rows and labels mismatch")
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        if len(self.classes_) not in (2, 3):
            raise ValueError("nanoTabPFN supports exactly 2/3 observed classes")
        if self.model is None:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            if not state.get("synthetic_only") or state.get("prior") not in ("generic_v2","mixture_v2"):
                raise ValueError("Expected matched generic-v2 nanoTabPFN checkpoint")
            self.model = new_nanotab(state["config"])
            self.model.load_state_dict(state["state_dict"], strict=True)
            self.model.to(self.device).eval()
        self.X_ = X
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("fit context before inference")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError("query feature count mismatch")
        outputs = []
        n = len(self.X_)
        for i in range(0, len(X), self.query_batch_size):
            chunk = X[i:i+self.query_batch_size]
            joined = torch.from_numpy(
                np.concatenate((self.X_, chunk), axis=0)[None]
            ).to(self.device)
            clean = impute_from_context(joined, n)
            labels = torch.from_numpy(self.y_[None].astype(np.float32)).to(
                self.device
            )
            with torch.inference_mode():
                logits = self.model(
                    (clean, labels), train_test_split_index=n
                )[..., :len(self.classes_)]
                outputs.append(torch.softmax(logits, dim=-1)[0].cpu().numpy())
        if not outputs:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _batch_loss(model, replay, rng, *, batch_size, context_budgets,
                device, query_rows=8):
    x, y, nctx, nc = replay.sample_batch(
        rng, batch_size=batch_size,
        context_budgets=context_budgets, query_rows=query_rows
    )
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    xt = impute_from_context(xt, nctx)
    out = model((xt, yt[:, :nctx].float()),
                train_test_split_index=nctx)[..., :nc]
    return F.cross_entropy(out.flatten(0,1), yt[:, nctx:].flatten())


@torch.inference_mode()
def validate(model, replay, device, seed, *, max_context):
    model.eval()
    rng = np.random.default_rng(seed)
    all_losses = []
    for context in (16, 32, 64, 128, 256, 512):
        if context > max_context:
            continue
        for _ in range(2):
            loss = _batch_loss(
                model, replay, rng, batch_size=4,
                context_budgets=(context,), device=device
            )
            all_losses.append(float(loss.item()))
    model.train()
    if not all_losses:
        raise RuntimeError("Empty held-out synthetic validation")
    return float(np.mean(all_losses))


def train_matched_nanotab(
    workspace, checkpoint_dir, *, steps=5000,
    batch_size=32, microbatch_size=8,
    num_tables=200000, val_tables=512,
    device="cuda", seed=20261107, val_seed=20271008,
    validate_every=250, max_context=512, prior="generic",
):
    """Match our v2 generic-prior task and optimizer budgets on upstream model."""
    if prior not in ("generic","mixture"):
        raise ValueError("Unsupported nanoTabPFN prior")
    if batch_size % microbatch_size or val_seed == seed:
        raise ValueError("invalid batch accumulation or validation seed")
    if max_context < 16 or steps < 1:
        raise ValueError("invalid context or training steps")
    workspace, checkpoint_dir = Path(workspace), Path(checkpoint_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_file = workspace / f"nanotab_{prior}_v2_train.h5"
    val_file = workspace / f"nanotab_{prior}_v2_validation.h5"
    replay_args = dict(
        prior=prior, min_rows=32, max_rows=576,
        min_features=5, max_features=16, variant="v2",
    )
    train_provenance = create_replay(
        train_file, num_tables=num_tables, seed=seed, **replay_args
    )
    val_provenance = create_replay(
        val_file, num_tables=val_tables, seed=val_seed, **replay_args
    )
    if not train_provenance["fully_generated"] or not val_provenance["fully_generated"]:
        raise RuntimeError("Synthetic archives incomplete")
    manifest = {
        "train_prior":train_provenance, "validation_prior":val_provenance,
        "source":"automl/nanoTabPFN@6071eec58a1a3225854480125001d5a62e279f6d",
        "architecture":"upstream-nanotabpfn", "prior":prior,
        "optimizer":"AdamW fixed LR 5e-4 (Tiny v2 matched)",
        "lr_peak":0.0005, "seed":seed, "validation_seed":val_seed,
        "steps":steps, "batch_size":batch_size,
        "microbatch_size":microbatch_size,
        "max_context":max_context, "query_rows":8,
        "validate_every":validate_every, "synthetic_only":True,
    }
    (checkpoint_dir/f"training_protocol_{prior}.json").write_text(
        json.dumps(manifest, indent=2)+"\n"
    )
    config=NanoTabConfig()
    torch.manual_seed(seed)
    np_rng=np.random.default_rng(seed)
    random.seed(seed)
    model=new_nanotab(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
    latest=checkpoint_dir/f"nanotabpfn_{prior}.train.pt"
    best=checkpoint_dir/f"nanotabpfn_{prior}.best.inference.pt"
    history=[]
    start=0
    best_val=float("inf")
    if latest.exists():
        state=torch.load(latest,map_location="cpu",weights_only=False)
        if state["protocol"] != {
            k:manifest[k] for k in (
                "seed","steps","batch_size","microbatch_size",
                "max_context","validate_every",
            )
        }:
            raise RuntimeError("Refusing incompatible nanoTabPFN training resume")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        np_rng.bit_generator.state=state["np_rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        start=int(state["step"])
        best_val=float(state["best_validation_loss"])
        history=state["validation_history"]
        print(f"Resuming nanoTabPFN from step {start}/{steps}",flush=True)
    budgets=tuple(c for c in (16,32,64,128,256,512) if c<=max_context)
    if start < steps:
        model.train()
        with HDF5TaskReplay(train_file) as replay, HDF5TaskReplay(val_file) as val:
            for step in range(start+1, steps+1):
                # Match the Tiny nanoMaternalPFN v2 AdamW peak LR and fixed schedule.
                for group in optimizer.param_groups:
                    group["lr"]=5e-4
                optimizer.zero_grad(set_to_none=True)
                total=0.
                for _ in range(batch_size//microbatch_size):
                    loss=_batch_loss(
                        model,replay,np_rng,batch_size=microbatch_size,
                        context_budgets=budgets,device=device,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite nanoTabPFN loss step {step}")
                    (loss/(batch_size//microbatch_size)).backward()
                    total+=float(loss.item())/(batch_size//microbatch_size)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step % validate_every == 0 or step == steps:
                    score=validate(
                        model,val,device,seed=val_seed,
                        max_context=max_context
                    )
                    history.append({
                        "step":step,"train_loss":total,
                        "heldout_synthetic_loss":score,
                    })
                    if score<best_val:
                        best_val=score
                        torch.save({
                            "config":asdict(config),
                            "state_dict":model.state_dict(),
                            "step":step, "validation_loss":score,
                            "prior":f"{prior}_v2", "synthetic_only":True,
                            "upstream_sha":"6071eec58a1a3225854480125001d5a62e279f6d",
                        },best.with_suffix(".tmp"))
                        best.with_suffix(".tmp").replace(best)
                    state={
                        "protocol":{k:manifest[k] for k in (
                            "seed","steps","batch_size","microbatch_size",
                            "max_context","validate_every",
                        )},
                        "config":asdict(config),"state_dict":model.state_dict(),
                        "optimizer":optimizer.state_dict(),"step":step,
                        "validation_history":history,
                        "best_validation_loss":best_val,
                        "np_rng_state":np_rng.bit_generator.state,
                        "torch_rng_state":torch.get_rng_state(),
                        "cuda_rng_states":torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None,
                    }
                    tmp=latest.with_suffix(".tmp")
                    torch.save(state,tmp)
                    tmp.replace(latest)
                    print(f"[nanoTabPFN] step {step}/{steps} train_loss={total:.5f} "
                          f"val_loss={score:.5f} best={best_val:.5f}",flush=True)
    if not best.exists():
        raise FileNotFoundError(f"Expected best checkpoint: {best}")
    summary={
        "checkpoint":str(best), "final_step":steps,
        "best_heldout_synthetic_loss":best_val,
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes":best.stat().st_size,
        "generic_prior_only":True,"training_protocol":manifest,
        "validation_history":history,
    }
    (checkpoint_dir/f"training_results_{prior}.json").write_text(
        json.dumps(summary,indent=2)+"\n"
    )
    return summary
