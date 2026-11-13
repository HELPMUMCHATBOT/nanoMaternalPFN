#!/usr/bin/env python3
"""Complete C2/C3/C5/C6 factorial training with measured budget checkpoints.

Runs one identical training trajectory to the maximum budget per architecture
and synthetic prior setting, saving real checkpoints at each prespecified
step milestone. This avoids redundant model retraining for four budgets.
No AUROC is guessed and no clinical data participate in model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

from nanomaternalpfn.priors.hdf5_replay import create_replay, file_sha256
from nanomaternalpfn.train_replay import train_replay


def experiment_id(variant, maternal_fraction, steps, encoder, norm, tier):
    data = dict(variant=variant,maternal_fraction=maternal_fraction,
                steps=steps,encoder=encoder,norm=norm,tier=tier)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:14]


def run(args):
    budgets = sorted(set(args.step_budgets))
    if not budgets or min(budgets) < 1 or len(budgets) != len(args.step_budgets):
        raise ValueError("step budgets must be positive and unique")
    if not args.maternal_fractions or any(not 0<=p<=1 for p in args.maternal_fractions):
        raise ValueError("maternal fractions must be in [0,1]")
    if len(set(args.maternal_fractions)) != len(args.maternal_fractions):
        raise ValueError("duplicate maternal mixture weight")
    if args.train_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if args.batch_size<1 or args.microbatch_size<1 or args.batch_size % args.microbatch_size:
        raise ValueError("microbatch must divide effective batch")
    if args.validate_every<1 or args.num_tables<2 or args.validation_tables<2:
        raise ValueError("invalid validation or replay parameters")
    if any(b % args.validate_every for b in budgets):
        raise ValueError("all budget milestones must be divisible by validate_every")

    root = Path(args.output)
    trajectories = list(itertools.product(
        args.variants,args.maternal_fractions,args.encoders,
        args.normalizations,args.tiers,
    ))
    specs = []
    for variant,fraction,encoder,norm,tier in trajectories:
        for steps in budgets:
            specs.append(dict(
                id=experiment_id(variant,fraction,steps,encoder,norm,tier),
                variant=variant,maternal_fraction=float(fraction),
                steps=steps,encoder=encoder,normalization=norm,tier=tier,
            ))
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("duplicate experiment IDs")
    root.mkdir(parents=True,exist_ok=True)
    plan = dict(
        schema_version=2, experiments=specs, execute=args.execute,
        synthetic_only=True, clinical_results_measured=False,
        train_seed=args.train_seed,validation_seed=args.validation_seed,
        num_tables=args.num_tables,validation_tables=args.validation_tables,
        device=args.device, effective_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        training_trajectories=len(trajectories),
        optimizer_updates_per_trajectory=max(budgets),
        total_optimizer_updates=len(trajectories)*max(budgets),
        milestone_budgets=budgets,
    )
    plan_file=root/"research_plan.json"
    if args.resume and plan_file.exists():
        prior=json.loads(plan_file.read_text())
        for name in ("schema_version","experiments","train_seed","validation_seed",
                     "num_tables","validation_tables","effective_batch_size",
                     "microbatch_size","milestone_budgets"):
            if prior.get(name)!=plan.get(name):
                raise ValueError("cannot resume a different research plan: "+name)
    plan_file.write_text(json.dumps(plan,indent=2)+"\n")
    if not args.execute:
        return {
            "planned_configurations":len(specs),
            "shared_training_trajectories":len(trajectories),
            "completed":0,"plan":str(plan_file),
            "status":"not_executed",
        }

    result_file=root/"measured_results.json"
    results=json.loads(result_file.read_text()) if args.resume and result_file.exists() else []
    completed={r["id"] for r in results}
    if len(completed)!=len(results) or not completed.issubset({s["id"] for s in specs}):
        raise ValueError("invalid existing result IDs for resume")
    for variant,fraction,encoder,norm,tier in trajectories:
        items=[s for s in specs if s["variant"]==variant
               and s["maternal_fraction"]==fraction
               and s["encoder"]==encoder and s["normalization"]==norm
               and s["tier"]==tier]
        if all(s["id"] in completed for s in items):
            continue
        prefix=f"p{round(fraction*100):03d}_{variant}"
        archive=root/"priors"/f"{prefix}_train.h5"
        validation=root/"priors"/f"{prefix}_validation.h5"
        common=dict(prior="mixture",variant=variant,
                    maternal_probability=fraction,
                    min_rows=24 if args.smoke else 32,
                    max_rows=48 if args.smoke else 576,
                    max_features=8 if args.smoke else 16)
        for path,num,seed in (
            (archive,args.num_tables,args.train_seed),
            (validation,args.validation_tables,args.validation_seed),
        ):
            res=create_replay(path,num_tables=num,seed=seed,**common)
            if not res["fully_generated"]:
                raise RuntimeError("incomplete prior dump")
        if file_sha256(archive)==file_sha256(validation):
            raise RuntimeError("training and validation archives identical")
        trajectory=experiment_id(variant,fraction,max(budgets),encoder,norm,tier)
        output=root/"checkpoints"/(trajectory+".pt")
        if output.exists() and not args.resume:
            raise FileExistsError("existing checkpoint; refusing overwrite: "+str(output))
        started=time.perf_counter()
        report=train_replay(
            archive,output,tier=tier,steps=max(budgets),
            batch_size=args.batch_size,microbatch_size=args.microbatch_size,
            device=args.device,variant=variant,
            feature_encoder_type=encoder,normalization=norm,
            max_context=16 if args.smoke else args.max_context,
            query_rows=4 if args.smoke else 8,
            save_every=args.validate_every,validate_every=args.validate_every,
            validation_path=validation,validation_batches_per_context=1 if args.smoke else 2,
            validation_batch_size=2 if args.smoke else 8,
            seed=args.train_seed,resume=args.resume and output.exists(),
        )
        elapsed=time.perf_counter()-started
        records={int(h["step"]):h for h in report["validation_history"]}
        for spec in items:
            if spec["id"] in completed:
                continue
            budget=spec["steps"]
            snapshot=output.with_name(f"{output.stem}.step{budget:05d}.inference.pt")
            if not snapshot.is_file():
                raise RuntimeError("missing measured budget snapshot: "+str(snapshot))
            if budget not in records:
                raise RuntimeError("missing measured validation for budget "+str(budget))
            values=[(k,r["loss"]) for k,r in records.items() if k<=budget]
            best_step,best_loss=min(values,key=lambda item:item[1])
            entry={
                **spec,
                "training_trajectory_id":trajectory,
                "trajectory_total_elapsed_seconds_measured":elapsed,
                "completed_optimizer_steps":budget,
                "task_presentations":budget*args.batch_size,
                "train_loss_last":records[budget]["train_loss_last"],
                "heldout_synthetic_loss":records[budget]["loss"],
                "heldout_synthetic_best_loss":best_loss,
                "heldout_synthetic_best_step":best_step,
                "training_replay_sha256":report["replay_sha256"],
                "validation_replay_sha256":report["validation_sha256"],
                "inference_checkpoint":str(snapshot),
                "clinical_auroc_measured":False,
                "synthetic_only":True,
            }
            results.append(entry)
            completed.add(spec["id"])
            result_file.write_text(json.dumps(results,indent=2)+"\n")
            print(json.dumps(entry,sort_keys=True),flush=True)
    if len(results)!=len(specs):
        raise RuntimeError("C1-C6 factorial study incomplete")
    return {
        "planned_configurations":len(specs),
        "shared_training_trajectories":len(trajectories),
        "completed":len(results),
        "results":str(result_file),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="artifacts/results/c1_c6_research")
    p.add_argument("--variants",nargs="+",choices=("v2","v3_robust"),default=["v2","v3_robust"])
    p.add_argument("--maternal-fractions",nargs="+",type=float,default=[0,.3,.5,.7,1.])
    p.add_argument("--step-budgets",nargs="+",type=int,default=[2500,5000,10000,25000])
    p.add_argument("--encoders",nargs="+",choices=("linear","gated"),default=["linear","gated"])
    p.add_argument("--normalizations",nargs="+",choices=("standard","robust"),default=["standard","robust"])
    p.add_argument("--tiers",nargs="+",choices=("tiny","base","small"),default=["tiny"])
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--train-seed",type=int,default=20261107)
    p.add_argument("--validation-seed",type=int,default=20271008)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    p.add_argument("--max-context",type=int,default=512)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.smoke and not a.execute:
        p.error("--smoke requires --execute")
    print(json.dumps(run(a),indent=2))


if __name__=="__main__":
    main()
