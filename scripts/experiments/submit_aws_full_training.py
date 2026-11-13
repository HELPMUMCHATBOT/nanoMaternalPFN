#!/usr/bin/env python3
"""Launch ALL 11 full E2/E7/E11 experiments sequentially on SageMaker A10G.

DRY RUN unless --submit --confirm-billing are both supplied. Uses the user's
AWS_PROFILE and IAM role; never embeds credentials or secret keys. Each job
has distinct model.tar.gz evidence and a full optimizer budget.

Example:
 AWS_PROFILE=helpmum-training python -m scripts.experiments.submit_aws_full_training \\
  --role-arn "$SAGEMAKER_ROLE_ARN" --bucket "$SAGEMAKER_BUCKET" \\
  --submit --confirm-billing
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from .training_matrix import make_matrix
from .common import write_json


def stage_code(repo,output):
    repo=Path(repo);output=Path(output)
    if not (repo/'src'/'nanomaternalpfn'/'model.py').is_file():
        raise FileNotFoundError('Run from repository root; cannot find src/nanomaternalpfn/model.py')
    output.mkdir(parents=True,exist_ok=True)
    shutil.copytree(repo/'src'/'nanomaternalpfn',output/'src'/'nanomaternalpfn',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
    for rel in ['scripts/__init__.py','scripts/experiments/__init__.py',
                'scripts/experiments/train_case_sagemaker.py',
                'scripts/experiments/train_controlled.py',
                'scripts/experiments/training_matrix.py',
                'scripts/experiments/common.py']:
        dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/rel,dest)
    # Official PyTorch training image already contains torch. Extra packages
    # required to create synthetic tasks are installed by training toolkit.
    (output/'requirements.txt').write_text('numpy>=1.26,<3\nscipy>=1.11\nscikit-learn>=1.4\nh5py>=3.9\npandas>=2.0\n')
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>12_000_000:raise RuntimeError('Unexpected large SageMaker source bundle: inspect repository')
    return size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role-arn',default=os.environ.get('SAGEMAKER_ROLE_ARN'))
    p.add_argument('--bucket',default=os.environ.get('SAGEMAKER_BUCKET'))
    p.add_argument('--region',default=os.environ.get('AWS_DEFAULT_REGION','us-east-1'))
    p.add_argument('--prefix',default='nanomaternalpfn/e1_e12_full')
    p.add_argument('--instance-type',default='ml.g5.2xlarge')
    p.add_argument('--max-run',type=int,default=172800)
    p.add_argument('--jobs-file',type=Path,default=Path('artifacts/results/e1_e12/aws_full_training_jobs.json'))
    p.add_argument('--case-id',default='all',help='A case ID or "all" (11 full runs)')
    p.add_argument('--submit',action='store_true')
    p.add_argument('--confirm-billing',action='store_true')
    a=p.parse_args()
    cases=make_matrix() if a.case_id=='all' else [c for c in make_matrix() if c['id']==a.case_id]
    if not cases:raise SystemExit('Unknown training case '+a.case_id)
    if not a.role_arn or not a.bucket:
        raise SystemExit('Provide --role-arn and --bucket (or SAGEMAKER_ROLE_ARN and SAGEMAKER_BUCKET)')
    print(json.dumps({'case_count':len(cases),'full_budgets':[{'case':c['id'],'steps':c['factors']['steps']}
                    for c in cases], 'instance_type':a.instance_type,
                    'region':a.region,'billable_jobs_will_launch':a.submit},indent=2))
    if not a.submit:
        print('DRY RUN: add BOTH --submit and --confirm-billing to start real billed GPU jobs.')
        return
    if not a.confirm_billing:
        raise SystemExit('AWS charges require --confirm-billing; no jobs submitted')
    import boto3
    from sagemaker.pytorch import PyTorch
    from sagemaker.session import Session
    client_session=boto3.Session(region_name=a.region)
    sm=client_session.client('sagemaker')
    s3=client_session.client('s3')
    s3.head_bucket(Bucket=a.bucket)
    session=Session(boto_session=client_session)
    repo=Path.cwd()
    jobs=[]
    if a.jobs_file.is_file():
        jobs=json.loads(a.jobs_file.read_text()).get('jobs',[])
    known={j['case_id']:j for j in jobs}
    for case in cases:
        if case['id'] in known:
            prior=sm.describe_training_job(TrainingJobName=known[case['id']]['training_job_name'])
            if prior['TrainingJobStatus']=='Completed':
                print('PRESERVING completed full training job',case['id']);continue
            if prior['TrainingJobStatus'] in ('InProgress','Stopping'):
                raise RuntimeError('Previously submitted job still running; no duplicate launch: '+case['id'])
            raise RuntimeError('Previously submitted training failed; inspect AWS job before retrying: '+case['id'])
        with tempfile.TemporaryDirectory(prefix='nanopfn-full-aws-') as tmp:
            root=Path(tmp);stage_code(repo,root)
            estimator=PyTorch(
                entry_point='scripts/experiments/train_case_sagemaker.py',
                source_dir=str(root),
                role=a.role_arn,framework_version='2.5.1',py_version='py311',
                instance_count=1,instance_type=a.instance_type,
                volume_size=120,max_run=a.max_run,
                output_path=f's3://{a.bucket}/{a.prefix.strip("/")}/outputs/',
                sagemaker_session=session,disable_profiler=True,
                hyperparameters={'case-id':case['id']},environment={'OMP_NUM_THREADS':'4'})
            stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
            label=re.sub('[^a-z0-9-]','-',case['id'].lower())
            name=f'nmpfn-{label}-{stamp}'[:63].strip('-')
            estimator.fit(wait=False,logs=False,job_name=name)
            row={'case_id':case['id'],'training_job_name':estimator.latest_training_job.name,
                 'region':a.region,'instance_type':a.instance_type,
                 'full_target_steps':case['factors']['steps'],'state':'submitted'}
            jobs.append(row);known[case['id']]=row
            write_json(a.jobs_file,{'jobs':jobs,'user_billing_confirmed':True})
            print('SUBMITTED FULL GPU TRAINING',name,flush=True)
        # Preserve AWS quota and complete one real full-budget run before
        # requesting the next GPU instance.
        while True:
            response=sm.describe_training_job(TrainingJobName=name)
            status=response['TrainingJobStatus']
            if status in ('Completed','Failed','Stopped'):
                row['state']=status
                row['model_artifacts_s3']=response.get('ModelArtifacts',{}).get('S3ModelArtifacts')
                row['failure_reason']=response.get('FailureReason')
                write_json(a.jobs_file,{'jobs':jobs,'user_billing_confirmed':True})
                if status!='Completed':raise RuntimeError('AWS full training did not complete: '+name+' '+str(row['failure_reason']))
                print('AWS FULL TRAINING FINISHED',name,flush=True)
                break
            time.sleep(60)
    print('Full training jobs finished. Download exact model.tar.gz artifacts before clinical evaluation.')

if __name__=='__main__':main()
