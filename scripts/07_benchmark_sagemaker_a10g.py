#!/usr/bin/env python3
"""GPU pretraining throughput probe for SageMaker A10G (no performance claims).

Benchmark two model sizes and representative context lengths, with the *same
MHA, optimizer, loss and backward path* as real train_replay.py. It times only
GPU steps, so wall-time estimates must include HDF5 reading and validation.
"""
import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from nanomaternalpfn.model import NanoMaternalPFN
from nanomaternalpfn.train_replay import TIERS


def probe(tier,ctx,batch,features,query_rows,warmup,timed,device):
    torch.manual_seed(20261107)
    model=NanoMaternalPFN(TIERS[tier]).to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
    x=torch.randn(batch,ctx+query_rows,features,device=device)
    x[:,::7,0]=float('nan')
    y=torch.randint(0,3,(batch,ctx+query_rows),device=device)
    measured=[]
    oom=False
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(warmup+timed):
            if device.type=="cuda":
                torch.cuda.synchronize()
            start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits=model(x,y[:,:ctx],n_classes=3)
            loss=F.cross_entropy(logits.flatten(0,1),y[:,ctx:].flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if device.type=="cuda":
                torch.cuda.synchronize()
            if i>=warmup:
                measured.append(time.perf_counter()-start)
    except torch.cuda.OutOfMemoryError:
        oom=True
    finally:
        del model,opt,x,y
        if device.type=="cuda":
            torch.cuda.empty_cache()
    if oom:
        return {"tier":tier,"ctx":ctx,"batch":batch,"oom":True,
                "recommendation":"Retry at --batch-size 4 or 2; full study settings must record changed effective batch."}
    median=statistics.median(measured)
    return {"tier":tier,"ctx":ctx,"batch":batch,"oom":False,
            "median_step_s":round(median,5),"p90_step_s":round(sorted(measured)[int(.9*(len(measured)-1))],5),
            "projected_2500_steps_h":round(2500*median/3600,2),
            "peak_cuda_memory_gib":round(torch.cuda.max_memory_allocated()/2**30,2) if device.type=="cuda" else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tiers",nargs="+",default=["tiny","base"],choices=["tiny","small","base"])
    p.add_argument("--contexts",nargs="+",type=int,default=[16,128,512])
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--features",type=int,default=16)
    p.add_argument("--query-rows",type=int,default=8)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--timed",type=int,default=10)
    p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--output",default="artifacts/results/a10g_throughput.json")
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; run on ml.g5.2xlarge")
    device=torch.device("cuda")
    results=[probe(tier,ctx,a.batch_size,a.features,a.query_rows,a.warmup,a.timed,device)
             for tier in a.tiers for ctx in a.contexts]
    report={"device_name":torch.cuda.get_device_name(),"torch_version":torch.__version__,
            "gpu_count":torch.cuda.device_count(),"estimated_steps_per_model":a.steps,
            "workload":results,"note":"GPU kernel-only timing; does not include disk, HDF5, model startup, upload or benchmark overhead."}
    import pathlib
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print("Estimate full 2x2 compute from budget-weighted step timing, not by summing all budgets.")
if __name__=="__main__":main()
