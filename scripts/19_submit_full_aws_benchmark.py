#!/usr/bin/env python3
"""Stage frozen manifests/checkpoints and launch one complete SageMaker GPU job.

Dry run by default. Use --submit to incur AWS SageMaker + S3 charges.
Only verified frozen manifests and four best v2 + two best v1 weights go to S3.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


DATASETS=(
    "uci_maternal_risk","vat_gdm","vat_delivery","low_birth_weight",
    "cardiotocography","pima_diabetes","tpehg_preterm","ctu_cord_acidemia",
)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()


def frozen_inputs(root,manifest_dir,v1_dir,v2_dir):
    files=[]
    for name in DATASETS:
        meta=manifest_dir/f"{name}.json"
        data=manifest_dir/f"{name}.npz"
        if not meta.is_file() or not data.is_file():
            raise FileNotFoundError(f"Missing frozen data: {meta} / {data}")
        declared=json.loads(meta.read_text())
        if declared["sha256"]!=sha256(data):
            raise RuntimeError(f"SHA256 mismatch: {data}")
        files.extend([
            (meta,f"manifests/{name}.json"),
            (data,f"manifests/{name}.npz"),
        ])
    for name in ("tiny_generic","tiny_mixture","base_generic","base_mixture"):
        src=v1_dir/f"{name}.best.inference.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        files.append((src,f"checkpoints/v1/{src.name}"))
    for name in ("tiny_generic","tiny_mixture","base_generic","base_mixture"):
        src=v2_dir/f"{name}.best.inference.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        files.append((src,f"checkpoints/v2/{src.name}"))
    return files


def create_input_bundle(files,out):
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,"w:gz") as tar:
        for path,relative in files:
            tar.add(path,arcname=relative,recursive=False)
    return {
        "path":str(out),"sha256":sha256(out),
        "files":[{"relative":relative,"sha256":sha256(path),
                  "bytes":Path(path).stat().st_size}
                 for path,relative in files],
    }


def stage_source(root,out):
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/"sagemaker_benchmark.py",out/"sagemaker_benchmark.py")
    shutil.copy2(root/"requirements_benchmark.txt",out/"requirements.txt")
    shutil.copytree(
        root/"src"/"nanomaternalpfn",
        out/"src"/"nanomaternalpfn",
        ignore=shutil.ignore_patterns("__pycache__","*.pyc",".DS_Store"),
    )
    scripts=out/"scripts"
    scripts.mkdir(parents=True,exist_ok=True)
    for name in ("18_run_complete_benchmark.py","20_profile_complete_models.py",
                 "22_analyze_paired_effects.py"):
        shutil.copy2(root/"scripts"/name,scripts/name)
    size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size>10*1024*1024:
        raise RuntimeError(f"SageMaker source bundle unexpectedly large: {size}")
    return size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit",action="store_true")
    p.add_argument("--region",default="us-east-1")
    p.add_argument("--bucket",default="sagemaker-us-east-1-355119164071")
    p.add_argument("--prefix",default="nanomaternalpfn/paper-complete-v1")
    p.add_argument("--role-arn",
                   default="arn:aws:iam::355119164071:role/SageMakerExecutionRole")
    p.add_argument("--instance-type",default="ml.g5.2xlarge")
    p.add_argument("--volume-gb",type=int,default=80)
    p.add_argument("--max-run-seconds",type=int,default=172800)
    p.add_argument("--manifest-dir",type=Path,default=Path("artifacts/manifests"))
    p.add_argument("--v1-dir",type=Path,default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--v2-dir",type=Path,default=Path("artifacts/results/aws_v2_pretrain"))
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--query-batch-size",type=int,default=64)
    args=p.parse_args()
    if not (32<=args.volume_gb<=1000) or args.max_run_seconds < 600:
        p.error("Invalid volume size or maximum run time")
    repo_root=Path(__file__).resolve().parent.parent
    files=frozen_inputs(
        repo_root,args.manifest_dir,args.v1_dir,args.v2_dir
    )
    print(json.dumps({
        "submit":args.submit,"instance":args.instance_type,
        "region":args.region,
        "models":18,"datasets":len(DATASETS),
        "contexts":[16,32,64,128,256,512],
        "draws_per_context":20,
        "nanotab_training_steps":args.steps,
        "nanotab_training_tables":args.num_tables,
        "input_files":len(files),
        "profile":"single-query matched GPU plus all-draw test predictions",
        "S3":f"s3://{args.bucket}/{args.prefix}",
    },indent=2),flush=True)
    if not args.submit:
        print("DRY RUN COMPLETE. Add --submit to upload and launch billable GPU job.")
        return
    try:
        import boto3
        from sagemaker.pytorch import PyTorch
        from sagemaker.session import Session
    except ImportError as exc:
        raise RuntimeError(
            "Install AWS SDK: python -m pip install 'sagemaker>=2.230,<3' boto3"
        ) from exc
    boto=boto3.Session(region_name=args.region)
    account=boto.client("sts").get_caller_identity()["Account"]
    if account!="355119164071":
        raise RuntimeError(f"Refusing unexpected AWS account {account}")
    boto.client("s3").head_bucket(Bucket=args.bucket)
    with tempfile.TemporaryDirectory(prefix="nano-paper-stage-") as temporary:
        temp=Path(temporary)
        bundle_meta=create_input_bundle(files,temp/"inputs.tar.gz")
        stage_source(repo_root,temp/"source")
        s3=boto.client("s3")
        key=f"{args.prefix.strip('/')}/inputs/{bundle_meta['sha256']}.tar.gz"
        s3.upload_file(str(temp/"inputs.tar.gz"),args.bucket,key)
        s3_uri=f"s3://{args.bucket}/{key}"
        s3.put_object(
            Bucket=args.bucket,
            Key=f"{args.prefix.strip('/')}/inputs/input_manifest.json",
            Body=json.dumps(bundle_meta,indent=2).encode(),
        )
        estimator=PyTorch(
            entry_point="sagemaker_benchmark.py",
            source_dir=str(temp/"source"),
            role=args.role_arn,
            framework_version="2.5.1",py_version="py311",
            instance_type=args.instance_type,instance_count=1,
            volume_size=args.volume_gb,max_run=args.max_run_seconds,
            output_path=f"s3://{args.bucket}/{args.prefix.strip('/')}/outputs",
            checkpoint_s3_uri=f"s3://{args.bucket}/{args.prefix.strip('/')}/checkpoint",
            sagemaker_session=Session(boto_session=boto),
            disable_profiler=True,
            environment={"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"},
            hyperparameters={
                "device":"cuda", "steps":args.steps,
                "num-tables":args.num_tables,
                "validation-tables":args.validation_tables,
                "query-batch-size":args.query_batch_size,
            },
        )
        from datetime import datetime,timezone
        job_name="nanomaternalpfn-paper-"+datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S"
        )
        print(f"Submitting GPU benchmark {job_name}",flush=True)
        estimator.fit(
            inputs={"benchmark":s3_uri},
            wait=False,logs=False,job_name=job_name,
        )
        print(json.dumps({
            "submitted_job":estimator.latest_training_job.name,
            "input_s3_uri":s3_uri,
            "checkpoint_prefix":f"s3://{args.bucket}/{args.prefix.strip('/')}/checkpoint",
            "output_prefix":f"s3://{args.bucket}/{args.prefix.strip('/')}/outputs",
        },indent=2),flush=True)


if __name__=="__main__":
    main()
