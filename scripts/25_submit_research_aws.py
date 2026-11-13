#!/usr/bin/env python3
"""Preview or submit measured C1-C6 synthetic pretraining to AWS SageMaker.

Defaults are dry run, requiring --submit to incur GPU charges. The checkpoint
prefix is dedicated to C1-C6 and cannot overwrite prior paper v1/v2 S3 keys.
"""
import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def stage(root,destination):
    root=Path(root); destination=Path(destination)
    shutil.copy2(root/"sagemaker_research.py",destination/"sagemaker_research.py")
    shutil.copy2(root/"requirements.txt",destination/"requirements.txt")
    shutil.copytree(
        root/"src"/"nanomaternalpfn",destination/"src"/"nanomaternalpfn",
        ignore=shutil.ignore_patterns("__pycache__","*.pyc",".DS_Store"),
    )
    (destination/"scripts").mkdir()
    shutil.copy2(root/"scripts"/"23_run_c1_c6_research.py",
                 destination/"scripts"/"23_run_c1_c6_research.py")
    total=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if total>10*1024*1024:
        raise RuntimeError("source package unexpectedly exceeds 10 MiB")
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit",action="store_true")
    p.add_argument("--region",default=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
    p.add_argument("--bucket",required=True)
    p.add_argument("--prefix",default="nanomaternalpfn/c1-c6-research")
    p.add_argument("--role-arn",required=True)
    p.add_argument("--instance-type",default="ml.g5.2xlarge")
    p.add_argument("--max-run-seconds",type=int,default=86400)
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    p.add_argument("--step-budgets",default="2500,5000,10000,25000")
    p.add_argument("--maternal-fractions",default="0,.3,.5,.7,1")
    p.add_argument("--variants",default="v2,v3_robust")
    p.add_argument("--encoders",default="linear,gated")
    p.add_argument("--normalizations",default="standard,robust")
    p.add_argument("--tiers",default="tiny")
    a=p.parse_args()
    if a.max_run_seconds<3600 or a.batch_size<1 or a.microbatch_size<1:
        p.error("invalid run limit or batch size")
    if a.batch_size%a.microbatch_size:
        p.error("microbatch size must divide effective batch size")
    if not a.bucket.strip() or not a.role_arn.strip():
        p.error("bucket and role ARN required")
    prefix=a.prefix.strip("/")
    if not prefix.startswith("nanomaternalpfn/c1-c6-"):
        p.error("S3 output prefix must start with nanomaternalpfn/c1-c6-")
    uri=f"s3://{a.bucket}/{prefix}"
    grids={
        "variant":a.variants.split(","),
        "maternal_fraction":a.maternal_fractions.split(","),
        "step_budget":a.step_budgets.split(","),
        "encoder":a.encoders.split(","),
        "normalization":a.normalizations.split(","),
        "tier":a.tiers.split(","),
    }
    count=1
    for dimension in grids.values():
        count*=len(dimension)
    print(json.dumps({
        "billable_job_submission":a.submit,
        "region":a.region,"instance_type":a.instance_type,
        "max_run_seconds":a.max_run_seconds,"s3_uri":uri,
        "experiment_count":count,
        "effective_batch_size":a.batch_size,
        "synthetic_only":True,"clinical_auroc_measured":False,
        "note":"Full matrix may require multiple checkpoint-resumed GPU jobs.",
    },indent=2))
    if not a.submit:
        print("DRY RUN: no AWS resources created. Add --submit to launch.")
        return

    try:
        import boto3
        from sagemaker.pytorch import PyTorch
        from sagemaker.session import Session
    except ImportError as exc:
        raise SystemExit("Install: python -m pip install 'sagemaker>=2.230,<3' boto3") from exc
    boto=boto3.Session(region_name=a.region)
    boto.client("sts").get_caller_identity()
    boto.client("s3").head_bucket(Bucket=a.bucket)
    session=Session(boto_session=boto)
    root=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="nanomaternalpfn-c1-c6-") as temp:
        stage(root,Path(temp))
        model=PyTorch(
            entry_point="sagemaker_research.py",source_dir=temp,
            role=a.role_arn,framework_version="2.5.1",py_version="py311",
            instance_count=1,instance_type=a.instance_type,
            volume_size=100,max_run=a.max_run_seconds,
            output_path=uri+"/outputs",
            checkpoint_s3_uri=uri+"/checkpoints",
            sagemaker_session=session,disable_profiler=True,
            hyperparameters={
                "variants":a.variants,
                "maternal-fractions":a.maternal_fractions,
                "step-budgets":a.step_budgets,
                "encoders":a.encoders,
                "normalizations":a.normalizations,
                "tiers":a.tiers,
                "num-tables":a.num_tables,
                "validation-tables":a.validation_tables,
                "batch-size":a.batch_size,
                "microbatch-size":a.microbatch_size,
            },
        )
        job_name="nanomaternalpfn-c1c6-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        model.fit(wait=False,logs=False,job_name=job_name)
        print(json.dumps({
            "submitted_job":model.latest_training_job.name,
            "output_prefix":uri+"/outputs",
            "checkpoint_prefix":uri+"/checkpoints",
        },indent=2))


if __name__=="__main__":
    main()
