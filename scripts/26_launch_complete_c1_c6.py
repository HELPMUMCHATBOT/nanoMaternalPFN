#!/usr/bin/env python3
"""ONE COMMAND to submit the complete C1-C6 study to SageMaker.

Runs the synthetic factorial training matrix, common-probe selection, measured
C4 distillation, and eight-task frozen clinical comparisons in one GPU job.
Not a pilot and not a dry-run: invocation submits the billable job.
Local AWS credentials are needed; they are never embedded or uploaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def load_bundle_helper():
    path=ROOT/"scripts"/"19_submit_full_aws_benchmark.py"
    spec=importlib.util.spec_from_file_location("paper_bundle_helpers",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_source(target):
    target=Path(target)
    shutil.copy2(ROOT/"sagemaker_research.py",target/"sagemaker_research.py")
    shutil.copy2(ROOT/"requirements_benchmark.txt",target/"requirements.txt")
    shutil.copytree(
        ROOT/"src"/"nanomaternalpfn",
        target/"src"/"nanomaternalpfn",
        ignore=shutil.ignore_patterns("__pycache__","*.pyc",".DS_Store"),
    )
    scripts=target/"scripts"
    scripts.mkdir(parents=True,exist_ok=True)
    for filename in (
        "18_run_complete_benchmark.py","23_run_c1_c6_research.py",
        "24_distill_tiny.py",
    ):
        shutil.copy2(ROOT/"scripts"/filename,scripts/filename)
    total=sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    if total>10*1024*1024:
        raise RuntimeError("unexpected SageMaker source bundle size")
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-name",default=None,help="Optional fixed job name for automatic continuation")
    p.add_argument("--region",default="us-east-1")
    p.add_argument("--bucket",default="sagemaker-us-east-1-355119164071")
    p.add_argument("--role-arn",default="arn:aws:iam::355119164071:role/SageMakerExecutionRole")
    p.add_argument("--prefix",default="nanomaternalpfn/c1-c6-complete")
    p.add_argument("--manifest-dir",type=Path,default=Path("artifacts/manifests"))
    p.add_argument("--v1-dir",type=Path,default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--v2-dir",type=Path,default=Path("artifacts/results/aws_v2_pretrain"))
    p.add_argument("--instance-type",default="ml.g5.2xlarge")
    p.add_argument("--volume-gb",type=int,default=150)
    p.add_argument("--max-run-seconds",type=int,default=432000,
                   help="Training-job limit in this AWS account is 432000 seconds (five days)")
    p.add_argument("--step-budgets",default="2500,5000,10000,25000")
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--distill-steps",type=int,default=25000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    a=p.parse_args()
    if a.volume_gb<100 or not 3600<=a.max_run_seconds<=432000:
        p.error("volume must be >=100 GiB and max-run-seconds between 3600 and the account quota of 432000")
    if a.batch_size<1 or a.microbatch_size<1 or a.batch_size%a.microbatch_size:
        p.error("invalid effective/microbatch size")
    if not a.prefix.startswith("nanomaternalpfn/c1-c6-"):
        p.error("output prefix must be isolated from original paper outputs")
    budgets=[int(x.strip()) for x in a.step_budgets.split(",")]
    if not budgets or any(b<1 for b in budgets):
        p.error("step budgets must be positive")
    bundle=load_bundle_helper()
    local_files=bundle.frozen_inputs(
        ROOT,a.manifest_dir,a.v1_dir,a.v2_dir
    )
    print(json.dumps({
        "submitting":"COMPLETE BILLABLE C1-C6 SAGEMAKER STUDY",
        "clinical_tasks":len(bundle.DATASETS),
        "synthetic_training_arms":2*5*len(budgets)*2*2,
        "distillation_steps":a.distill_steps,
        "eight_task_benchmark":"all contexts and all frozen context draws",
        "max_run_seconds":a.max_run_seconds,
        "instance_type":a.instance_type,
        "notes":"Full factorial can be extremely expensive. This is not a pilot or a dry-run.",
    },indent=2),flush=True)
    try:
        import boto3
        from sagemaker.pytorch import PyTorch
        from sagemaker.session import Session
    except ImportError as exc:
        raise SystemExit("Install AWS SDK: python -m pip install 'sagemaker>=2.230,<3' boto3") from exc

    boto=boto3.Session(region_name=a.region)
    account=boto.client("sts").get_caller_identity()["Account"]
    if account!="355119164071":
        raise RuntimeError(f"unexpected AWS account: {account}")
    boto.client("s3").head_bucket(Bucket=a.bucket)
    with tempfile.TemporaryDirectory(prefix="nanomaternalpfn-c1-c6-full-") as scratch:
        temp=Path(scratch)
        source=temp/"source"
        source.mkdir()
        stage_source(source)
        metadata=bundle.create_input_bundle(local_files,temp/"research_input.tar.gz")
        key=f"{a.prefix.strip('/')}/inputs/{metadata['sha256']}.tar.gz"
        s3=boto.client("s3")
        s3.upload_file(str(temp/"research_input.tar.gz"),a.bucket,key)
        input_uri=f"s3://{a.bucket}/{key}"
        s3.put_object(
            Bucket=a.bucket,
            Key=f"{a.prefix.strip('/')}/inputs/input_manifest.json",
            Body=json.dumps(metadata,indent=2).encode("utf-8"),
        )
        out_uri=f"s3://{a.bucket}/{a.prefix.strip('/')}"
        job=PyTorch(
            entry_point="sagemaker_research.py",
            source_dir=str(source),
            role=a.role_arn,framework_version="2.5.1",py_version="py311",
            instance_count=1,instance_type=a.instance_type,
            volume_size=a.volume_gb,max_run=a.max_run_seconds,
            output_path=out_uri+"/outputs",
            checkpoint_s3_uri=out_uri+"/checkpoints",
            sagemaker_session=Session(boto_session=boto),
            disable_profiler=True,
            environment={"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"},
            hyperparameters={
                "variants":"v2,v3_robust",
                "maternal-fractions":"0,.3,.5,.7,1",
                "step-budgets":a.step_budgets,
                "encoders":"linear,gated",
                "normalizations":"standard,robust",
                "tiers":"tiny",
                "num-tables":a.num_tables,
                "validation-tables":a.validation_tables,
                "batch-size":a.batch_size,
                "microbatch-size":a.microbatch_size,
                "distill-steps":a.distill_steps,
            },
        )
        name=a.job_name or ("nanomaternalpfn-allc1c6-"+
                         datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        if len(name)>63 or not name.startswith("nanomaternalpfn-allc1c6-"):
            raise ValueError("invalid research job name")
        print("Submitting full GPU study:",name,flush=True)
        job.fit(inputs={"research_input":input_uri},wait=False,logs=False,job_name=name)
        print(json.dumps({
            "submitted_job":job.latest_training_job.name,
            "input_s3":input_uri,
            "checkpoint_s3":out_uri+"/checkpoints",
            "outputs_s3":out_uri+"/outputs",
            "status_command":f"aws sagemaker describe-training-job --training-job-name {name} --query TrainingJobStatus --output text",
        },indent=2),flush=True)


if __name__=="__main__":
    main()
