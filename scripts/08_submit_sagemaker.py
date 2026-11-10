#!/usr/bin/env python3
"""Submit dedicated nanoMaternalPFN jobs to AWS SageMaker, benchmark first.

Run from your Mac with AWS_PROFILE=helpmum-training.
Default is dry-run (no billable job). --submit explicitly launches one job.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("benchmark","train"),default="benchmark")
    p.add_argument("--submit",action="store_true",help="Actually create a billable SageMaker GPU training job")
    p.add_argument("--region",default=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
    p.add_argument("--bucket",default="sagemaker-us-east-1-355119164071")
    p.add_argument("--prefix",default="nanomaternalpfn/phase2")
    p.add_argument("--role-arn",default="arn:aws:iam::355119164071:role/SageMakerExecutionRole")
    p.add_argument("--steps",type=int,default=2500)
    p.add_argument("--batch-size",type=int,default=32,help="Effective batch size for full training")
    p.add_argument("--microbatch-size",type=int,default=8,help="Per-forward batch size for accumulated training")
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--max-run-seconds",type=int,default=86400)
    args=p.parse_args()
    if args.batch_size<1 or args.steps<1 or args.num_tables<1:
        p.error("batch-size, steps and num-tables must be positive")
    if args.microbatch_size<1 or args.batch_size%args.microbatch_size:
        p.error("microbatch-size must be a positive divisor of batch-size")
    uri=f"s3://{args.bucket}/{args.prefix.strip('/')}"
    print(json.dumps({
        "mode":args.mode,"will_launch_billable_gpu":args.submit,
        "instance_type":"ml.g5.2xlarge","region":args.region,
        "role_arn":args.role_arn,"s3_prefix":uri,
        "steps_per_model":args.steps,"num_tables_per_prior":args.num_tables,
        "effective_batch_size":args.batch_size,
        "microbatch_size":args.microbatch_size,
        "gradient_accumulation_steps":args.batch_size//args.microbatch_size,
        "task_presentations_per_model":args.steps*args.batch_size,
        "checkpoint_s3_uri":uri+"/checkpoints",
        "note":"SageMaker writes final model and results to output prefix. No prior archives are uploaded automatically."
    },indent=2))
    if not args.submit:
        print("DRY RUN ONLY: add --submit to actually launch (billable GPU).")
        return
    try:
        import boto3
        from sagemaker.pytorch import PyTorch
        from sagemaker.session import Session
    except ImportError as exc:
        raise SystemExit("Install AWS SDK on your Mac: python3 -m pip install 'sagemaker>=2.230,<3' boto3") from exc
    boto=boto3.Session(region_name=args.region)
    # Quick identity and bucket checks before an expensive GPU allocation.
    sts=boto.client("sts")
    account=sts.get_caller_identity()["Account"]
    if account!="355119164071":
        raise SystemExit(f"Refusing to submit under unexpected account: {account}")
    boto.client("s3").head_bucket(Bucket=args.bucket)
    session=Session(boto_session=boto)
    kwargs=dict(
        entry_point="sagemaker_train.py",
        source_dir=str(Path(__file__).resolve().parent.parent),
        role=args.role_arn,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.2xlarge",
        volume_size=40,
        max_run=args.max_run_seconds,
        output_path=uri+"/outputs",
        checkpoint_s3_uri=uri+"/checkpoints",
        sagemaker_session=session,
        disable_profiler=True,
        hyperparameters={
            "mode":args.mode,
            "steps":args.steps,
            "batch-size":args.batch_size if args.mode=="train" else args.microbatch_size,
            "microbatch-size":args.microbatch_size,
            "num-tables":args.num_tables,
            "device":"cuda",
        },
        environment={"OMP_NUM_THREADS":"4"},
    )
    estimator=PyTorch(**kwargs)
    estimator.fit(wait=False,logs=False,job_name=f"nanomaternalpfn-{args.mode}-{__import__('datetime').datetime.now(__import__('datetime').timezone.utc):%Y%m%d-%H%M%S}")
    print("JOB SUBMITTED:",estimator.latest_training_job.name)
    print("Check: aws sagemaker describe-training-job --training-job-name",estimator.latest_training_job.name,
          "--query TrainingJobStatus --output text")


if __name__=="__main__":
    main()
