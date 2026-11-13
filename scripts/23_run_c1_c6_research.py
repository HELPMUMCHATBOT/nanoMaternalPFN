#!/usr/bin/env python3
"""Reproducible C2/C3/C5/C6 factorial experiment runner.

Use --execute to generate real synthetic replay archives and perform actual
optimizer updates. Never substitutes a forecast for measured training results.
Clinical evaluation remains a separate frozen-manifest benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

from nanomaternalpfn.priors.hdf5_replay import create_replay, file_sha256
from nanomaternalpfn.train_replay import train_replay


def experiment_id(variant, maternal_fraction, steps, encoder, norm, tier):
    spec = dict(
        variant=variant, maternal_fraction=maternal_fraction, steps=steps,
        encoder=encoder, norm=norm, tier=tier,
    )
    key = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:14]


def run(args):
    if not args.step_budgets or any(s < 1 for s in args.step_budgets):
        raise ValueError("step budgets must be positive")
    if not args.maternal_fractions or any(not 0 <= p <= 1 for p in args.maternal_fractions):
        raise ValueError("maternal fractions must be in [0,1]")
    if args.train_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if args.batch_size < 1 or args.microbatch_size < 1 or args.batch_size % args.microbatch_size:
        raise ValueError("microbatch must divide batch size")
    if args.validate_every < 1 or args.num_tables < 2 or args.validation_tables < 2:
        raise ValueError("invalid validation or replay configuration")
    root = Path(args.output)
    specs = []
    for variant, fraction, steps, encoder, norm, tier in itertools.product(
        args.variants, args.maternal_fractions, args.step_budgets,
        args.encoders, args.normalizations, args.tiers,
    ):
        ident = experiment_id(variant, fraction, steps, encoder, norm, tier)
        specs.append(dict(
            id=ident, variant=variant, maternal_fraction=float(fraction),
            steps=steps, encoder=encoder, normalization=norm, tier=tier,
        ))
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("duplicate experiment configurations")
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(
        experiments=specs, execute=args.execute, synthetic_only=True,
        clinical_results_measured=False, train_seed=args.train_seed,
        validation_seed=args.validation_seed, num_tables=args.num_tables,
        validation_tables=args.validation_tables, device=args.device,
        effective_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
    )
    (root / "research_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        return dict(planned=len(specs), completed=0, plan=str(root / "research_plan.json"),
                    statement="DRY RUN. No optimizer steps were executed.")
    results = []
    for spec in specs:
        prefix = f"p{round(spec['maternal_fraction'] * 100):03d}_{spec['variant']}"
        archive = root / "priors" / f"{prefix}_train.h5"
        validation = root / "priors" / f"{prefix}_validation.h5"
        common = dict(
            prior="mixture", variant=spec["variant"],
            maternal_probability=spec["maternal_fraction"],
            min_rows=24 if args.smoke else 32,
            max_rows=48 if args.smoke else 576,
            max_features=8 if args.smoke else 16,
        )
        for path, num, seed in (
            (archive, args.num_tables, args.train_seed),
            (validation, args.validation_tables, args.validation_seed),
        ):
            generated = create_replay(path, num_tables=num, seed=seed, **common)
            if not generated["fully_generated"]:
                raise RuntimeError("partial replay is not valid for training")
        if file_sha256(archive) == file_sha256(validation):
            raise RuntimeError("training replay equals validation replay")
        output = root / "checkpoints" / (spec["id"] + ".pt")
        if output.exists():
            raise FileExistsError(
                f"{output} already exists; use another output root to avoid overwriting results"
            )
        started = time.perf_counter()
        report = train_replay(
            archive, output, tier=spec["tier"], steps=spec["steps"],
            batch_size=args.batch_size, microbatch_size=args.microbatch_size,
            device=args.device, variant=spec["variant"],
            feature_encoder_type=spec["encoder"],
            normalization=spec["normalization"],
            max_context=16 if args.smoke else args.max_context,
            query_rows=4 if args.smoke else 8,
            save_every=args.validate_every, validate_every=args.validate_every,
            validation_path=validation, validation_batches_per_context=1 if args.smoke else 2,
            validation_batch_size=2 if args.smoke else 8,
            seed=args.train_seed,
        )
        entry = {
            **spec, "elapsed_seconds_measured":time.perf_counter() - started,
            "completed_optimizer_steps":report["step"],
            "task_presentations":report["task_presentations"],
            "train_loss_last":report["loss_last"],
            "heldout_synthetic_best_loss":report["best_validation_loss"],
            "heldout_synthetic_best_step":report["best_validation_step"],
            "training_replay_sha256":report["replay_sha256"],
            "validation_replay_sha256":report["validation_sha256"],
            "inference_checkpoint":report["best_inference_checkpoint"],
            "clinical_auroc_measured":False,
            "synthetic_only":True,
        }
        results.append(entry)
        (root / "measured_results.json").write_text(json.dumps(results,indent=2) + "\n")
        print(json.dumps(entry,sort_keys=True),flush=True)
    return dict(planned=len(specs), completed=len(results),
                results=str(root / "measured_results.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--smoke",action="store_true",help="Use small contexts; never clinical evidence")
    args = p.parse_args()
    if args.smoke and not args.execute:
        p.error("--smoke requires --execute")
    print(json.dumps(run(args),indent=2))


if __name__ == "__main__":
    main()
