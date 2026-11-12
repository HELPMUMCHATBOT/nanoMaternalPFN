#!/usr/bin/env python3
"""SageMaker entrypoint: train matched nanoTabPFN, evaluate all 13 arms, profile.

One GPU job, one frozen input bundle, checkpoint-backed resume.
SageMaker checkpoint_s3_uri syncs /opt/ml/checkpoints between jobs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"src"))


def extract_inputs(channel_path, destination):
    """Securely extract our own frozen tar bundle, refusing path traversal."""
    channel_path=Path(channel_path)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    archives=list(channel_path.glob("*.tar.gz"))
    if len(archives)==1:
        tar=archives[0]
        with tarfile.open(tar,"r:gz") as archive:
            members=archive.getmembers()
            target_root=destination.resolve()
            for member in members:
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Unexpected non-file input member {member.name}")
                if (target_root/member.name).resolve().is_relative_to(target_root) is False:
                    raise ValueError("Refusing path escape in benchmark bundle")
                extracted=destination/member.name
                extracted.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source, extracted.open("wb") as output:
                    shutil.copyfileobj(source,output)
    elif (channel_path/"manifests").is_dir():
        for item in channel_path.rglob("*"):
            if item.is_file():
                dest=destination/item.relative_to(channel_path)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(item,dest)
    else:
        raise FileNotFoundError(f"Missing frozen benchmark tar bundle: {channel_path}")
    if not (destination/"manifests").is_dir():
        raise RuntimeError("Input tar omitted frozen manifests")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--microbatch-size",type=int,default=8)
    p.add_argument("--validation-tables",type=int,default=512)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--max-context",type=int,default=512)
    p.add_argument("--query-batch-size",type=int,default=64)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--input-channel",default=os.environ.get(
        "SM_CHANNEL_BENCHMARK","/opt/ml/input/data/benchmark"
    ))
    p.add_argument("--workdir",default="/opt/ml/nanopaper")
    p.add_argument("--checkpoint-dir",default=os.environ.get(
        "SM_CHECKPOINT_DIR","/opt/ml/checkpoints"
    ))
    p.add_argument("--model-dir",default=os.environ.get("SM_MODEL_DIR","/opt/ml/model"))
    args=p.parse_args()
    if args.device=="cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("SageMaker GPU job did not provide CUDA")
        print(f"GPU: {torch.cuda.get_device_name(0)}",flush=True)
    work=Path(args.workdir)
    input_dir=work/"inputs"
    extract_inputs(args.input_channel,input_dir)
    import torch
    from nanomaternalpfn.nanotabpfn_baseline import train_matched_nanotab
    ckpt=Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True,exist_ok=True)
    training=train_matched_nanotab(
        workspace=work/"synthetic_prior",
        checkpoint_dir=ckpt/"nanotab_pretrain",
        steps=args.steps, batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        num_tables=args.num_tables,
        val_tables=args.validation_tables,device=args.device,
        validate_every=args.validate_every,max_context=args.max_context,
    )
    copy_dest=input_dir/"checkpoints"/"nanotabpfn_generic.best.inference.pt"
    copy_dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(training["checkpoint"],copy_dest)
    # Frozen inputs are never modified, but locally staged copy receives
    # the newly trained, held-out-synthetic-selected nanoTabPFN weights.
    metrics_dir=ckpt/"paper_benchmark_results"
    command=[
        sys.executable,str(HERE/"scripts"/"18_run_complete_benchmark.py"),
        "--input-dir",str(input_dir),"--output-dir",str(metrics_dir),
        "--device",args.device,
        "--query-batch-size",str(args.query_batch_size),
    ]
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(
        filter(None,[str(HERE/"src"),env.get("PYTHONPATH","")])
    )
    subprocess.run(command,env=env,check=True)
    # Timed single-query comparisons with same synthetic shape. This is
    # separate from full test-pool inference latency in all_draws.csv.
    profile_command=[
        sys.executable,str(HERE/"scripts"/"20_profile_complete_models.py"),
        "--input-dir",str(input_dir),"--output",str(metrics_dir/"hardware_profile.json"),
        "--device",args.device,"--features","6",
        "--contexts","16","64","128",
        "--repeats","10",
    ]
    subprocess.run(profile_command,env=env,check=True)
    model_dir=Path(args.model_dir)
    model_dir.mkdir(parents=True,exist_ok=True)
    shutil.copytree(metrics_dir,model_dir/"paper_benchmark",dirs_exist_ok=True)
    shutil.copy2(
        ckpt/"nanotab_pretrain"/"training_results.json",
        model_dir/"nanotab_training_results.json"
    )
    shutil.copy2(
        ckpt/"nanotab_pretrain"/"training_protocol.json",
        model_dir/"nanotab_training_protocol.json"
    )
    (model_dir/"COMPLETE.json").write_text(
        json.dumps({
            "training":"completed", "benchmark":"completed",
            "profiling":"completed", "models":13, "datasets":8,
            "note":"exploratory; existing holdouts informed v2",
        },indent=2)+"\n"
    )
    print("COMPLETE: model archive contains every result and raw draw",flush=True)


if __name__=="__main__":
    main()
