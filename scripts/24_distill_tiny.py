#!/usr/bin/env python3
"""Run a measured synthetic-only teacher-to-Tiny distillation (C4)."""
import argparse
import json
from pathlib import Path
from nanomaternalpfn.distillation import distill


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--validation-replay",type=Path,required=True)
    p.add_argument("--teachers",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    p.add_argument("--temperature",type=float,default=2.)
    p.add_argument("--alpha",type=float,default=.5)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--max-context",type=int,default=512)
    p.add_argument("--query-rows",type=int,default=8)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--feature-encoder-type",choices=("linear","gated"),default="gated")
    p.add_argument("--normalization",choices=("standard","robust"),default="robust")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    print(json.dumps(distill(
        a.replay,a.teachers,a.output,steps=a.steps,
        validation_path=a.validation_replay,
        batch_size=a.batch_size,microbatch_size=a.microbatch_size,
        temperature=a.temperature,alpha=a.alpha,lr=a.lr,
        max_context=a.max_context,query_rows=a.query_rows,
        validate_every=a.validate_every,
        feature_encoder_type=a.feature_encoder_type,
        normalization=a.normalization,device=a.device,
    ),indent=2))


if __name__ == "__main__":
    main()
