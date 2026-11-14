#!/usr/bin/env python3
"""Inspect account-specific SageMaker GPU training quotas and active usage.

READ-ONLY: Never creates or starts a training job. A positive training quota
does NOT establish live EC2/SageMaker capacity; only submission can do that.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json

GPU_CHOICES=(
    ("ml.g6.2xlarge","L4",24,"Closest 24GB GPU alternative"),
    ("ml.g6.4xlarge","L4",24,"Same GPU, additional host memory"),
    ("ml.g6e.2xlarge","L40S",48,"Larger 48GB GPU"),
    ("ml.g6e.4xlarge","L40S",48,"Larger GPU and host"),
    ("ml.g5.4xlarge","A10G",24,"Same GPU family, larger host"),
    ("ml.g5.8xlarge","A10G",24,"Same GPU family, still larger host"),
    ("ml.g5.2xlarge","A10G",24,"Original requested instance"),
    ("ml.g4dn.2xlarge","T4",16,"Lower VRAM; OOM risk"),
)


def list_quotas(quota_client):
    values={}
    iterator=quota_client.get_paginator("list_service_quotas").paginate(
        ServiceCode="sagemaker"
    )
    for page in iterator:
        for quota in page.get("Quotas",[]):
            name=str(quota.get("QuotaName",""))
            if " for training job usage" in name and "spot" not in name.lower():
                values[name.split(" for training job usage")[0]]=float(quota["Value"])
    return values


def list_running(sagemaker):
    used=Counter()
    active=[]
    # ListTrainingJobs accepts these statuses (not 'Pending' or 'Starting').
    for status in ("InProgress","Stopping"):
        for page in sagemaker.get_paginator("list_training_jobs").paginate(
            StatusEquals=status
        ):
            for item in page.get("TrainingJobSummaries",[]):
                name=item["TrainingJobName"]
                details=sagemaker.describe_training_job(TrainingJobName=name)
                resource=details.get("ResourceConfig") or {}
                instance=resource.get("InstanceType")
                count=int(resource.get("InstanceCount",1))
                if instance:
                    used[instance]+=count
                active.append({
                    "name":name,"status":status,
                    "instance_type":instance or "unknown (preference list)",
                    "instance_count":count,
                })
    return used,active


def inspect(quotas,used):
    output=[]
    for name,gpu,vram,notes in GPU_CHOICES:
        quota=quotas.get(name)
        occupied=used.get(name,0)
        free=None if quota is None else max(0,quota-occupied)
        if quota is None:
            label="QUOTA UNKNOWN"
        elif quota==0:
            label="QUOTA ZERO"
        elif free==0:
            label="QUOTA IN USE"
        else:
            label="QUOTA HEADROOM (live capacity unknown)"
        output.append({
            "instance":name,"gpu":gpu,"vram_gb_per_gpu":vram,
            "quota":quota,"account_active_instances":occupied,
            "estimated_remaining_quota":free,"status":label,"notes":notes,
        })
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region",default="us-east-1")
    p.add_argument("--json",action="store_true",help="Machine-readable output")
    a=p.parse_args()
    import boto3
    session=boto3.Session(region_name=a.region)
    # Profile and credentials come from the normal AWS SDK chain, not code.
    quota_client=session.client("service-quotas")
    sm=session.client("sagemaker")
    quotas=list_quotas(quota_client)
    used,active=list_running(sm)
    rows=inspect(quotas,used)
    output={"region":a.region,"candidates":rows,"active_jobs":active,
            "live_capacity_confirmed":False,
            "note":"Quota headroom is not a capacity reservation or launch guarantee."}
    if a.json:
        print(json.dumps(output,indent=2))
        return
    print("SageMaker training instance choices in",a.region)
    print("INSTANCE             GPU    VRAM  QUOTA  RUNNING  LEFT  STATUS")
    for row in rows:
        def fmt(value):
            return "?" if value is None else ("%g" % value)
        print(f"{row['instance']:<20} {row['gpu']:<6} {row['vram_gb_per_gpu']:>4}GB "
              f"{fmt(row['quota']):>6} {row['account_active_instances']:>8} "
              f"{fmt(row['estimated_remaining_quota']):>5}  {row['status']}")
    if active:
        print("\nActive account SageMaker training jobs:")
        for job in active:
            print(" -",job["name"],job["instance_type"],
                  job["instance_count"],job["status"])
    print("\nIMPORTANT: Quota available != compute capacity available.")
    print("For a candidate with quota headroom, full study:")
    print(" python scripts/26_launch_complete_c1_c6.py --instance-type ml.g6.2xlarge")
    print("Use the actual eligible instance returned above, not this example,")
    print("and monitor launch outcome for AWS capacity errors.")


if __name__=="__main__":
    main()
