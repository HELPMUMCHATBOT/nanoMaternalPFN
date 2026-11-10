#!/usr/bin/env python3
"""Run 2x2 prior × model-size study with real optimizer steps and provenance.

This script never claims a model is trained unless the requested training
run has actually completed. Requires a GPU for the intended full study.
"""
import argparse
import json
from pathlib import Path
import torch
from nanomaternalpfn.priors.hdf5_replay import create_replay
from nanomaternalpfn.train_replay import train_replay

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="artifacts")
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--tiers",nargs="+",choices=["tiny","base","small"],default=["tiny","base"])
    p.add_argument("--priors",nargs="+",choices=["generic","mixture"],default=["generic","mixture"])
    p.add_argument("--smoke",action="store_true",help="Small functional test; never a study result")
    a=p.parse_args()
    if a.smoke:
        a.num_tables=20
        a.steps=2
        a.batch_size=2
    out=Path(a.out)
    results=[]
    for prior in a.priors:
        path=out/"priors_h5"/f"{prior}_{a.num_tables}.h5"
        prior_result=create_replay(path,num_tables=a.num_tables,prior=prior,
                                   min_rows=24 if a.smoke else 32,
                                   max_rows=48 if a.smoke else 576,
                                   min_features=5,max_features=8 if a.smoke else 16)
        if not prior_result["fully_generated"]:
            raise RuntimeError("Synthetic prior dump is partial")
        for tier in a.tiers:
            checkpoint=out/"checkpoints"/f"{tier}_{prior}.pt"
            report=train_replay(path,checkpoint,tier=tier,steps=a.steps,
                                batch_size=a.batch_size,device=a.device,
                                max_context=16 if a.smoke else 512,
                                save_every=max(1,a.steps//5),
                                resume=a.resume and checkpoint.exists())
            report["prior"]=prior
            report["smoke_only"]=bool(a.smoke)
            results.append(report)
    out.joinpath("results").mkdir(parents=True,exist_ok=True)
    out.joinpath("results","phase12_training_runs.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
