"""Resumable mixed-context pretraining from synthetic HDF5 task replay."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import ModelConfig, NanoMaternalPFN
from .priors.hdf5_replay import HDF5TaskReplay, file_sha256
from .validation_replay import evaluate_replay

TIERS={
    "tiny":ModelConfig(),
    "small":ModelConfig(128,4,4,256),
    "base":ModelConfig(192,6,6,384),
}


def train_replay(
    replay_path, output, *, tier="tiny", steps=2500, batch_size=8,
    lr=5e-4, seed=20261107, device="cpu", save_every=100,
    resume=False, max_context=512, query_rows=8, microbatch_size=None,
    validation_path=None, validate_every=250, validation_batches_per_context=2,
    validation_batch_size=8, validation_seed=20271008,
):
    """Train on a completed HDF5 replay; 'steps' is total desired steps.

    An abbreviated smoke checkpoint MUST NOT be described as research
    pretraining or clinical validation. Does not access real datasets.
    """
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if tier not in TIERS:
        raise ValueError("Unknown model tier")
    if steps<1 or batch_size<1 or query_rows<1:
        raise ValueError("steps, batch_size and query_rows must be positive")
    # batch_size is the EFFECTIVE number of tables per optimizer step.
    # Keep each CUDA forward/backward small, then accumulate gradients.
    microbatch_size = batch_size if microbatch_size is None else int(microbatch_size)
    if microbatch_size<1 or microbatch_size>batch_size or batch_size%microbatch_size:
        raise ValueError("microbatch_size must be a positive divisor of batch_size")
    accumulation_steps = batch_size // microbatch_size
    if validation_path is not None and (validate_every < 1 or validation_batches_per_context < 1 or validation_batch_size < 1):
        raise ValueError("Validation interval, batch size and batch count must be positive")
    validation_hash = file_sha256(validation_path) if validation_path is not None else None
    budgets=[b for b in (16,32,64,128,256,512) if b<=max_context]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA accelerator is unavailable")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    random.seed(seed)
    cfg=TIERS[tier]
    model=NanoMaternalPFN(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    start=0
    losses=[]
    validation_history=[]
    best_validation_loss=float("inf")
    best_validation_step=None
    replay_hash=file_sha256(replay_path)
    if resume:
        if not output.exists():
            raise FileNotFoundError(output)
        checkpoint=torch.load(output,map_location="cpu",weights_only=False)
        if checkpoint["replay_sha256"]!=replay_hash or checkpoint["config"]!=asdict(cfg):
            raise ValueError("Cannot resume on different prior archive/model architecture")
        for field,value in (("seed",seed),("batch_size",batch_size),("lr",lr),
                            ("max_context",max_context),("query_rows",query_rows)):
            if checkpoint[field]!=value:
                raise ValueError(f"Cannot resume with a different {field}")
        # Old checkpoints without this field used one full batch per update.
        prior_microbatch = checkpoint.get("microbatch_size", checkpoint["batch_size"])
        if prior_microbatch != microbatch_size:
            raise ValueError("Cannot resume with a different microbatch_size")
        if checkpoint.get("validation_sha256") != validation_hash:
            raise ValueError("Cannot resume with a different synthetic validation archive")
        if validation_hash is not None:
            for field,value in (("validate_every",validate_every),
                                ("validation_batches_per_context",validation_batches_per_context),
                                ("validation_batch_size",validation_batch_size),
                                ("validation_seed",validation_seed)):
                if checkpoint.get(field)!=value:
                    raise ValueError(f"Cannot resume with a different {field}")
        validation_history=list(checkpoint.get("validation_history", []))
        best_validation_loss=float(checkpoint.get("best_validation_loss", float("inf")))
        best_validation_step=checkpoint.get("best_validation_step")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start=int(checkpoint["step"])
        rng.bit_generator.state=checkpoint["np_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        losses=list(checkpoint["losses"])
    model.train()
    # Hold-out archive is independent of the training archive, with a distinct
    # deterministic task-generation seed. It is never sampled for optimizer updates.
    from contextlib import nullcontext
    validation_cm = HDF5TaskReplay(validation_path) if validation_path is not None else nullcontext(None)
    with validation_cm as validation_replay, HDF5TaskReplay(replay_path) as replay:
        for step in range(start+1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro_step in range(accumulation_steps):
                x,y,n_ctx,n_classes=replay.sample_batch(
                    rng,batch_size=microbatch_size,
                    context_budgets=budgets,
                    query_rows=query_rows)
                x=torch.from_numpy(x).to(device)
                y=torch.from_numpy(y).to(device)
                logits=model(x,y[:,:n_ctx],n_classes=n_classes)
                loss=F.cross_entropy(logits.flatten(0,1),y[:,n_ctx:].flatten())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at optimizer step {step}, microbatch {micro_step+1}")
                (loss / accumulation_steps).backward()
                step_loss += float(loss.detach().cpu()) / accumulation_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(step_loss)
            if validation_replay is not None and (step % validate_every == 0 or step == steps):
                metrics=evaluate_replay(
                    model, validation_replay, contexts=budgets, device=device,
                    batch_size=validation_batch_size,
                    batches_per_context=validation_batches_per_context,
                    query_rows=query_rows, draw_seed=validation_seed)
                metrics["step"]=step
                metrics["train_loss_last"]=step_loss
                validation_history.append(metrics)
                if metrics["loss"] < best_validation_loss:
                    best_validation_loss=metrics["loss"]
                    best_validation_step=step
                    # Separate best weights: selecting weights on a synthetic
                    # holdout is NOT evidence of clinical model quality.
                    best_path=output.with_name(output.stem+".best.inference.pt")
                    best_tmp=best_path.with_name(best_path.name+".tmp")
                    torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),
                                "step":step,"validation_loss":best_validation_loss,
                                "synthetic_only":True},best_tmp)
                    best_tmp.replace(best_path)
                history_path=output.with_name(output.stem+".validation.json")
                history_tmp=history_path.with_name(history_path.name+".tmp")
                history_tmp.write_text(json.dumps(validation_history,indent=2)+"\n")
                history_tmp.replace(history_path)
                print(f"[{tier}] step={step} train_loss={step_loss:.5f} "
                      f"heldout_synthetic_loss={metrics['loss']:.5f} "
                      f"best_step={best_validation_step}",flush=True)
            if step%save_every==0 or step%validate_every==0 or step==steps:
                state={
                    "state_dict":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "config":asdict(cfg),"tier":tier,
                    "seed":seed,"lr":lr,"batch_size":batch_size,
                    "microbatch_size":microbatch_size,
                    "gradient_accumulation_steps":accumulation_steps,
                    "task_presentations":step*batch_size,
                    "max_context":max_context,"query_rows":query_rows,
                    "step":step,"losses":losses,"prior":replay.settings,
                    "validation_sha256":validation_hash,
                    "validate_every":validate_every if validation_hash else None,
                    "validation_batches_per_context":validation_batches_per_context if validation_hash else None,
                    "validation_batch_size":validation_batch_size if validation_hash else None,
                    "validation_seed":validation_seed if validation_hash else None,
                    "validation_history":validation_history,
                    "best_validation_loss":best_validation_loss,
                    "best_validation_step":best_validation_step,
                    "replay_sha256":replay_hash,"synthetic_only":True,
                    "np_rng_state":rng.bit_generator.state,
                    "torch_rng_state":torch.get_rng_state(),
                    "python_rng_state":random.getstate(),
                    "environment":{"torch":torch.__version__,"numpy":np.__version__,
                                   "device":str(device)},
                }
                tmp=output.with_name(output.name+".tmp")
                torch.save(state,tmp)
                tmp.replace(output)
                # Separate weights-only checkpoint for safe inference loading.
                inference=output.with_name(output.stem+".inference.pt")
                torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),
                            "step":step,"synthetic_only":True,
                            "effective_batch_size":batch_size,
                            "microbatch_size":microbatch_size},inference)
                if validation_hash is not None and step%validate_every==0:
                    snapshot=output.with_name(f"{output.stem}.step{step:05d}.inference.pt")
                    torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),
                                "step":step,"synthetic_only":True,
                                "effective_batch_size":batch_size},snapshot)
    report={
        "tier":tier,"step":max(start,steps),"parameter_count":model.parameter_count(),
        "effective_batch_size":batch_size,"microbatch_size":microbatch_size,
        "gradient_accumulation_steps":accumulation_steps,
        "task_presentations":max(start,steps)*batch_size,
        "loss_last":losses[-1] if losses else None,
        "validation_sha256":validation_hash,
        "validation_history":validation_history,
        "best_validation_step":best_validation_step,
        "best_validation_loss":best_validation_loss if best_validation_step is not None else None,
        "best_inference_checkpoint":str(output.with_name(output.stem+".best.inference.pt")) if best_validation_step is not None else None,
        "checkpoint":str(output),"inference_checkpoint":str(output.with_name(output.stem+".inference.pt")),"replay_sha256":replay_hash,
        "synthetic_only":True,"research_validated":False,
        "device":str(device),
    }
    output.with_suffix(".json").write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description="Train or resume PFN from complete synthetic HDF5")
    p.add_argument("--replay",required=True)
    p.add_argument("--output",default="artifacts/checkpoints/tiny_mixture.pt")
    p.add_argument("--tier",choices=TIERS,default="tiny")
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--batch-size",type=int,default=8,help="Effective tasks per optimizer step")
    p.add_argument("--microbatch-size",type=int,default=None,
                   help="Tasks per forward/backward; defaults to effective batch size")
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-context",type=int,default=512)
    p.add_argument("--query-rows",type=int,default=8)
    p.add_argument("--validation-replay",default=None)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--validation-batches-per-context",type=int,default=2)
    a=p.parse_args()
    report=train_replay(a.replay,a.output,tier=a.tier,steps=a.steps,
                        batch_size=a.batch_size,lr=a.lr,seed=a.seed,
                        device=a.device,save_every=a.save_every,
                        resume=a.resume,max_context=a.max_context,
                        query_rows=a.query_rows,microbatch_size=a.microbatch_size,
                        validation_path=a.validation_replay,
                        validate_every=a.validate_every,
                        validation_batches_per_context=a.validation_batches_per_context)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
