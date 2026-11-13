#!/usr/bin/env python3
"""Run the FULL C1-C6 study with automatic 5-day SageMaker job continuation.

This is a real, billable execution controller, NOT a dry run. Uses the
account's 432000-second per-job limit, waits for completion, and relaunches
only when AWS explicitly reports SecondaryStatus=MaxRuntimeExceeded.
Every iteration uses the SAME S3 checkpoint prefix. This controller must
remain running on the user's authenticated Mac (or another always-on host).
No invented results. Unexpected errors stop instead of burning more GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
TERMINAL={"Completed","Stopped","Failed"}
NAME_PREFIX="nanomaternalpfn-allc1c6-"


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    temp.replace(path)


def current_commit():
    p=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True)
    return p.stdout.strip()


def build_spec(args):
    if not 3600<=args.max_run_seconds<=432000:
        raise ValueError("account runtime quota: max-run-seconds <= 432000")
    if args.max_jobs<1 or args.poll_seconds<30:
        raise ValueError("max-jobs >=1 and poll-seconds >=30 required")
    return {
        "schema_version":1,
        "git_commit":current_commit(),
        "region":args.region,
        "instance_type":args.instance_type,
        "bucket":args.bucket,
        "prefix":args.prefix,
        "max_run_seconds":args.max_run_seconds,
        "launcher_extra_args":args.launcher_arg,
    }


def new_job_name():
    return NAME_PREFIX+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def status_details(detail):
    primary=detail["TrainingJobStatus"]
    secondary=detail.get("SecondaryStatus","")
    reason=detail.get("FailureReason","")
    return primary,secondary,reason


def must_continue(detail):
    primary,secondary,reason=status_details(detail)
    # Do not retry user cancellation, a model crash, bad input, OOM or capacity failures.
    return primary=="Stopped" and secondary=="MaxRuntimeExceeded"


def submit(args,state,client):
    if len(state["jobs"])>=args.max_jobs:
        raise RuntimeError("max-jobs reached; refusing unlimited billable relaunches")
    name=new_job_name()
    while any(j["name"]==name for j in state["jobs"]):
        time.sleep(1)
        name=new_job_name()
    job={"name":name,"submitted":False,"status":"SubmissionPending",
         "submitted_at_utc":datetime.now(timezone.utc).isoformat()}
    state["jobs"].append(job)
    atomic_json(args.state,state)
    command=[
        sys.executable,str(ROOT/"scripts"/"26_launch_complete_c1_c6.py"),
        "--job-name",name,
        "--region",args.region,
        "--instance-type",args.instance_type,
        "--bucket",args.bucket,
        "--prefix",args.prefix,
        "--max-run-seconds",str(args.max_run_seconds),
        *args.launcher_arg,
    ]
    print("Submitting billable SageMaker job:",name,flush=True)
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    if process.stdout:
        print(process.stdout,flush=True)
    if process.stderr:
        print(process.stderr,file=sys.stderr,flush=True)
    if process.returncode:
        # The create call might have succeeded before the CLI itself failed.
        # Retain the pending job name to allow recovery, never blindly duplicate.
        job["launch_returncode"]=process.returncode
        atomic_json(args.state,state)
        raise RuntimeError("launch command failed; state saved; rerun controller to inspect job before retry")
    job["submitted"]=True
    job["status"]="Submitted"
    atomic_json(args.state,state)
    return job


def describe_or_fail(client,name):
    from botocore.exceptions import ClientError
    try:
        return client.describe_training_job(TrainingJobName=name)
    except ClientError as exc:
        if exc.response.get("Error",{}).get("Code") in (
            "ValidationException","ResourceNotFound",
        ):
            raise RuntimeError(
                "Pending job not found in SageMaker; inspect failed submission before restarting "
                + name
            ) from exc
        raise


def run(args,session=None,sleep=time.sleep):
    import boto3
    session=session or boto3.Session(region_name=args.region)
    client=session.client("sagemaker")
    spec=build_spec(args)
    target=Path(args.state)
    if target.exists():
        state=json.loads(target.read_text())
        if state.get("spec")!=spec:
            raise ValueError("Cannot continue with changed branch, AWS target or experiment settings")
    else:
        state={"spec":spec,"jobs":[],"completed":False}
        atomic_json(target,state)
    if state.get("completed"):
        print("Full C1-C6 research already marked complete; no new job submitted.")
        return state

    if not state["jobs"]:
        submit(args,state,client)
    else:
        print("Resuming watch for:",state["jobs"][-1]["name"],flush=True)

    while True:
        job=state["jobs"][-1]
        detail=describe_or_fail(client,job["name"])
        primary,secondary,reason=status_details(detail)
        latest=(primary,secondary,reason)
        if job.get("last_observed")!=list(latest):
            print(f"{job['name']}: {primary} / {secondary} {reason}",flush=True)
            job["last_observed"]=list(latest)
            job["status"]=primary
            atomic_json(target,state)
        if primary=="Completed":
            state["completed"]=True
            state["completed_job"]=job["name"]
            atomic_json(target,state)
            print("FULL RESEARCH JOB COMPLETED:",job["name"],flush=True)
            return state
        if must_continue(detail):
            job["status"]="MaxRuntimeExceeded"
            atomic_json(target,state)
            print("Five-day job limit reached. Continuing from S3 checkpoints.",flush=True)
            submit(args,state,client)
            continue
        if primary in TERMINAL:
            raise RuntimeError(
                f"SageMaker job ended without full completion: {job['name']} "
                f"primary={primary}, secondary={secondary}, reason={reason}. "
                "No additional billable job was launched."
            )
        sleep(args.poll_seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region",default="us-east-1")
    p.add_argument("--instance-type",default="ml.g6.2xlarge")
    p.add_argument("--bucket",default="sagemaker-us-east-1-355119164071")
    p.add_argument("--prefix",default="nanomaternalpfn/c1-c6-complete")
    p.add_argument("--max-run-seconds",type=int,default=432000)
    p.add_argument("--max-jobs",type=int,default=8)
    p.add_argument("--poll-seconds",type=int,default=600)
    p.add_argument("--state",default="artifacts/results/c1_c6_auto_resume_state.json")
    p.add_argument("--launcher-arg",action="append",default=[],
                   help="Pass an additional supported argument to the full launcher, one token per occurrence")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
