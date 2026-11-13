#!/usr/bin/env python3
"""Dedicated SageMaker entrypoint for measured C1-C6 synthetic research.

No clinical benchmark is run in this job. Checkpoints are persisted through
SageMaker's checkpoint directory and measurement reports through model output.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variants",default="v2,v3_robust")
    p.add_argument("--maternal-fractions",default="0,.3,.5,.7,1")
    p.add_argument("--step-budgets",default="2500,5000,10000,25000")
    p.add_argument("--encoders",default="linear,gated")
    p.add_argument("--normalizations",default="standard,robust")
    p.add_argument("--tiers",default="tiny")
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--max-context",type=int,default=512)
    p.add_argument("--train-seed",type=int,default=20261107)
    p.add_argument("--validation-seed",type=int,default=20271008)
    a=p.parse_args()

    output=Path(os.getenv("SM_MODEL_DIR","artifacts/sagemaker/c1_c6_model"))
    checkpoint_dir=Path("/opt/ml/checkpoints/c1_c6" if os.getenv("SM_MODEL_DIR")
                       else "artifacts/sagemaker/c1_c6_checkpoints")
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    command=[
        sys.executable,str(ROOT/"scripts/23_run_c1_c6_research.py"),
        "--execute","--resume","--device","cuda",
        "--output",str(checkpoint_dir),
        "--num-tables",str(a.num_tables),
        "--validation-tables",str(a.validation_tables),
        "--batch-size",str(a.batch_size),
        "--microbatch-size",str(a.microbatch_size),
        "--validate-every",str(a.validate_every),
        "--max-context",str(a.max_context),
        "--train-seed",str(a.train_seed),
        "--validation-seed",str(a.validation_seed),
    ]
    for flag,value in (
        ("variants",a.variants),("maternal-fractions",a.maternal_fractions),
        ("step-budgets",a.step_budgets),("encoders",a.encoders),
        ("normalizations",a.normalizations),("tiers",a.tiers),
    ):
        tokens=[x.strip() for x in value.split(",") if x.strip()]
        if not tokens:
            raise ValueError(f"empty sweep dimension: {flag}")
        command.extend(["--"+flag,*tokens])
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(filter(None,(
        str(ROOT/"src"),env.get("PYTHONPATH",""),
    )))
    subprocess.run(command,check=True,env=env)
    for name in ("research_plan.json","measured_results.json"):
        source=checkpoint_dir/name
        if source.exists():
            shutil.copy2(source,output/name)
    report={
        "synthetic_only":True,"clinical_auroc_measured":False,
        "status":"completed","checkpoint_dir":str(checkpoint_dir),
    }
    (output/"job_complete.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
