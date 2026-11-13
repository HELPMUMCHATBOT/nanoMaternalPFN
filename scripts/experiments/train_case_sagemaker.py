#!/usr/bin/env python3
"""SageMaker GPU entrypoint: exactly ONE full prescribed training-matrix case."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.experiments.training_matrix import make_matrix,case_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',required=True)
    a=p.parse_args()
    case=next((row for row in make_matrix() if row['id']==a.case_id),None)
    if case is None:raise ValueError('Training case not in preregistered matrix: '+a.case_id)
    output=Path(os.environ.get('SM_MODEL_DIR','/opt/ml/model'))
    output.mkdir(parents=True,exist_ok=True)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('SageMaker training job must have actual CUDA GPU')
    cmd=case_command(case,output,'cuda')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')])
    print('Running FULL configured pretraining:',a.case_id,flush=True)
    subprocess.run(cmd,check=True,cwd=str(ROOT),env=env)
    artifact=output/a.case_id/'COMPLETE.json'
    if not artifact.is_file():raise RuntimeError('No full-budget training completion manifest')
    status=json.loads(artifact.read_text())
    if status.get('steps_completed')!=case['factors']['steps']:
        raise RuntimeError('Training did not complete its full prescribed budget')
    print('FULL TRAINING COMPLETE:',a.case_id,flush=True)

if __name__=='__main__':main()
