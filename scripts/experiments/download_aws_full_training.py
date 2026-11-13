#!/usr/bin/env python3
"""Download completed real SageMaker full-training artifacts to the expected local directory.

Rejects unfinished AWS jobs, unsafe tar paths and missing COMPLETE.json.
"""
from __future__ import annotations
import argparse
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
from .common import write_json,sha256
from .training_matrix import make_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs-file',type=Path,default=Path('artifacts/results/e1_e12/aws_full_training_jobs.json'))
    p.add_argument('--output',type=Path,default=Path('artifacts/results/e1_e12/training'))
    a=p.parse_args()
    import boto3
    if not a.jobs_file.is_file():raise FileNotFoundError('Actual SageMaker jobs manifest not found')
    jobs=json.loads(a.jobs_file.read_text())['jobs']
    a.output.mkdir(parents=True,exist_ok=True)
    verified=[]
    cases={x['id']:x for x in make_matrix()}
    for j in jobs:
        case=cases[j['case_id']]
        dest=a.output/case['id']
        if (dest/'COMPLETE.json').is_file():
            report=json.loads((dest/'COMPLETE.json').read_text())
            if report.get('steps_completed')!=case['factors']['steps']:
                raise RuntimeError('Existing training directory has different budget: '+case['id'])
            verified.append(case['id']);continue
        sm=boto3.client('sagemaker',region_name=j['region'])
        r=sm.describe_training_job(TrainingJobName=j['training_job_name'])
        if r['TrainingJobStatus']!='Completed':
            raise RuntimeError('Training job is not completed: '+j['training_job_name'])
        uri=r['ModelArtifacts']['S3ModelArtifacts']
        parsed=urlparse(uri)
        if parsed.scheme!='s3':raise RuntimeError('Unexpected model artifact URI')
        client=boto3.client('s3',region_name=j['region'])
        with tempfile.TemporaryDirectory(prefix='nanopfn-model-') as temp:
            tarball=Path(temp)/'model.tar.gz'
            client.download_file(parsed.netloc,parsed.path.lstrip('/'),str(tarball))
            with tarfile.open(tarball,'r:gz') as archive:
                for m in archive.getmembers():
                    if m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts:
                        raise RuntimeError('Unsafe model tar entry '+m.name)
                archive.extractall(temp)  # all members manually checked above for unsafe paths and links
            origin=Path(temp)/case['id']
            if not (origin/'COMPLETE.json').is_file():
                raise RuntimeError('SageMaker model archive missing actual full training marker')
            report=json.loads((origin/'COMPLETE.json').read_text())
            if report.get('steps_completed')!=case['factors']['steps']:
                raise RuntimeError('SageMaker model did not reach required steps')
            if dest.exists():raise RuntimeError('Nonempty existing target must be inspected: '+str(dest))
            shutil.copytree(origin,dest)
            verified.append(case['id'])
            print('RETRIEVED REAL FULL TRAINING',case['id'],'checksum',sha256(dest/'best.inference.pt'),flush=True)
    write_json(a.output/'AWS_ARTIFACTS_RECEIVED.json',{'verified_full_training_cases':verified,
                    'required_full_training_cases':[x['id'] for x in make_matrix()],
                    'all_full_training_cases_received':len(verified)==len(make_matrix())})
    print('Verified',len(verified),'real training artifact sets')

if __name__=='__main__':main()
