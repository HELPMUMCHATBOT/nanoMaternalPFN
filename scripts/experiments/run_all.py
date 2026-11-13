#!/usr/bin/env python3
"""One-shot E1--E12 execution coordinator; fails *before* partial clinical runs.

New GPU training and prospective studies are gated by real infrastructure or
external governance. An E12 signed protocol does not equate to patient outcomes.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .common import HISTORICAL_TASKS,load_frozen,write_json,sha256
from .predictors import checkpoint,EXPERTS

DATASET_ORDER=('uci_maternal_risk','vat_gdm','vat_delivery','low_birth_weight',
               'cardiotocography','pima_diabetes','tpehg_preterm','ctu_cord_acidemia')


def prepare(p):
    manifests=[p.manifest_dir/f'{name}.json' for name in DATASET_ORDER]
    missing=[]
    for path in manifests:
        try:load_frozen(path)
        except (FileNotFoundError,ValueError,KeyError) as exc:missing.append(str(exc))
    ckpts=[checkpoint(n,p.v1_dir,p.v2_dir) for n in EXPERTS]
    for path in ckpts:
        if not path.is_file():missing.append('Final research checkpoint missing: '+str(path))
    if missing:raise RuntimeError('Cannot execute full benchmark: required frozen inputs absent:\n'+'\n'.join(missing))
    return manifests,ckpts


def call(command):
    print('EXECUTE',' '.join(command),flush=True)
    subprocess.run(command,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir',type=Path,default=Path('artifacts/manifests'))
    p.add_argument('--v1-dir',type=Path,default=Path('artifacts/results/aws_full_pretrain'))
    p.add_argument('--v2-dir',type=Path,default=Path('artifacts/results/aws_v2_pretrain'))
    p.add_argument('--output',type=Path,default=Path('artifacts/results/e1_e12'))
    p.add_argument('--external-manifest',type=Path,default=None)
    p.add_argument('--onnx',type=Path,default=None,help='Context-64, feature-matched standalone PFN graph')
    p.add_argument('--run-full',action='store_true',help='Run E3/E4/E5/E6/E8; no pilot draw cap')
    p.add_argument('--train-models',action='store_true',help='E2/E7/E11 separate full GPU suite')
    p.add_argument('--confirm-compute',action='store_true')
    a=p.parse_args()
    print('E1-E12 integrated evidence workflow; no simulated patient results.')
    if not a.run_full:
        raise SystemExit('Dry verification only: pass --run-full on the machine with real frozen datasets/checkpoints.')
    if a.train_models:
        if not a.confirm_compute:raise SystemExit('GPU training requested but compute not explicitly confirmed')
        try:
            import torch
            if not torch.cuda.is_available():raise SystemExit('CUDA GPU unavailable for full E2/E7/E11 training')
        except ImportError:raise SystemExit('PyTorch unavailable; no GPU training is possible')
    manifests,checkpoints=prepare(a) # fail before starting any benchmark
    a.output.mkdir(parents=True,exist_ok=True)
    payload=['--manifests']+[str(x) for x in manifests]+['--v1-dir',str(a.v1_dir),
         '--v2-dir',str(a.v2_dir),'--save-predictions']
    planned=[]
    if a.external_manifest is not None:
        load_frozen(a.external_manifest,external=True)
        planned.append(('E1',['--experiment','E1','--manifests',str(a.external_manifest),
            '--v1-dir',str(a.v1_dir),'--v2-dir',str(a.v2_dir),'--save-predictions',
            '--output',str(a.output/'E1')]))
    planned.extend((code,['--experiment',code]+payload+['--output',str(a.output/code)])
                  for code in ('E3','E5','E6','E8'))
    for code,argv in planned:
        call([sys.executable,'-m','scripts.experiments.run_predictions']+argv)
    call([sys.executable,'-m','scripts.experiments.profile_matched','--all',
          '--manifest',str(a.manifest_dir/'low_birth_weight.json'),'--context','64',
          '--repeats','100','--v1-dir',str(a.v1_dir),'--v2-dir',str(a.v2_dir),
          '--out',str(a.output/'E4'/'matched_cpu.json')])
    if a.onnx is not None:
        call([sys.executable,'-m','scripts.experiments.quantize_and_check',
             '--onnx',str(a.onnx),'--output-onnx',str(a.output/'E10'/'tiny.int8.onnx'),
             '--manifest',str(a.manifest_dir/'low_birth_weight.json'),
             '--report',str(a.output/'E10'/'metrics.json')])
    if a.train_models:
        if not a.confirm_compute:raise ValueError('Training compute not confirmed')
        from .training_matrix import make_matrix
        for case in make_matrix():
            call([sys.executable,'-m','scripts.experiments.training_matrix','--output',str(a.output/'training'),
                  '--device','cuda','--confirm-compute','--execute-case',case['id']])
    write_json(a.output/'EXECUTED_COORDINATOR.json',{'historical_manifest_sha256':{str(p):sha256(p) for p in manifests},
          'executed_stage_names':[x[0] for x in planned]+['E4']+(['E10'] if a.onnx else []),
          'E2_E7_E11_training_invoked':a.train_models,
          'E9_status':'requires actual ARM/Android physical machine',
          'E12_status':'requires independent clinical and ethics approvals plus prospective data',
          'E1_external_source_supplied':a.external_manifest is not None,
          'claim':'execution ledger only, not a completed evidence certification'})

if __name__=='__main__':main()
