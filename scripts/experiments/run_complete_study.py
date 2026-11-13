#!/usr/bin/env python3
"""One research-grade E1--E12 campaign coordinator, never a smoke benchmark.

- 'train' runs all 11 *full-budget* controlled synthetic PFN pretraining arms.
- 'evaluate' runs all frozen real-data clinical/model experiments, including
  newly trained E2/E7/E11 checkpoints. It never creates synthetic clinical rows.
- 'all' combines train then evaluate and optional full quantization.
- E1 requires a genuine independently sourced, locked clinical cohort.
- E9 requires execution *on* physical ARM/Android hardware; E12 requires real
  ethics-approved prospective data and clinical study procedures. They cannot
  be completed by the coordinator in their absence.

Results are stored in a local restricted directory. No source datasets, patient
records or research checkpoints are committed automatically to GitHub.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from .common import load_frozen,sha256,write_json
from .predictors import checkpoint,EXPERTS
from .training_matrix import make_matrix,case_command

TASKS=('uci_maternal_risk','vat_gdm','vat_delivery','low_birth_weight',
       'cardiotocography','pima_diabetes','tpehg_preterm','ctu_cord_acidemia')


def check_existing(manifest_dir,v1_dir,v2_dir,external_manifest=None):
    missing=[];manifests=[];sources={}
    for task in TASKS:
        path=Path(manifest_dir)/f'{task}.json'
        try:
            data=load_frozen(path)
            sources[task]={'npz_sha256':data.sha256,'n_rows':len(data.y),
                 'n_context_budgets':len(data.contexts),
                 'n_test':len(data.test),'n_draws':sum(len(v) for v in data.contexts.values())}
            manifests.append(path)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            missing.append(f'{task}: {exc}')
    existing=[]
    for name in EXPERTS:
        path=checkpoint(name,v1_dir,v2_dir)
        if not path.is_file():missing.append(f'Final pretrained checkpoint missing: {path}')
        elif 'smoke' in path.name.lower():missing.append(f'Not a final research checkpoint: {path}')
        else:existing.append({'expert':name,'path':str(path),'sha256':sha256(path)})
    if external_manifest:
        try:load_frozen(external_manifest,external=True)
        except (ValueError,KeyError,FileNotFoundError) as exc:missing.append(f'E1 independent external cohort: {exc}')
    return {'ready':not missing,'missing':missing,'frozen_sources':sources,
            'required_expert_checkpoints':existing,'external_validation_provided':bool(external_manifest),
            'missing_capabilities':{
                'E1':'actual independent cohort required unless external manifest supplied',
                'E9':'native ARM/Android device session and physical measurements required',
                'E12':'ethics approval, clinician oversight, prospectively enrolled subjects and actual outcomes required'}}


def full_training(case_root,device,confirmed,force_resume=True):
    if not confirmed:raise RuntimeError('Full training incurs substantial compute: --confirm-compute is mandatory')
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA accelerator not present: train using SageMaker or a CUDA host')
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple Silicon MPS unavailable: use CUDA instance or supported Mac')
    if device=='cpu':
        raise RuntimeError('Refusing CPU fallback for the full GPU-scale pretraining matrix')
    case_root=Path(case_root)
    for case in make_matrix():
        cfg=case_root/case['id']/'CONFIG.json'
        done=case_root/case['id']/'COMPLETE.json'
        best=case_root/case['id']/'best.inference.pt'
        if done.is_file() and best.is_file():
            old=json.loads(cfg.read_text());f=case['factors']
            factors=('architecture','class_augmentation','sign_augmentation','missingness_encoding',
                     'schedule','maternal_probability','steps')
            if any(old[k]!=f[k] for k in factors):
                raise RuntimeError('Previously completed case has different factors: '+case['id'])
            if int(json.loads(done.read_text()).get('steps_completed',-1))!=f['steps']:
                raise RuntimeError('Training completion steps differ from experiment matrix: '+case['id'])
            print('PRESERVED COMPLETE FULL TRAINING',case['id'],flush=True)
            continue
        cmd=case_command(case,case_root,device)
        if force_resume and '--resume' not in cmd:cmd.append('--resume')
        print('FULL PRETRAIN',case['id'],flush=True)
        subprocess.run(cmd,check=True)
        if not done.is_file() or not best.is_file():
            raise RuntimeError('Full training did not finish: '+case['id'])


def trained_groups(training_dir,require_all):
    training_dir=Path(training_dir)
    matrix=make_matrix();groups={'E2':{},'E7':{},'E11':{}}
    missing=[]
    for case in matrix:
        base=training_dir/case['id']
        done=base/'COMPLETE.json'; best=base/'best.inference.pt'
        if not (done.is_file() and best.is_file()):
            missing.append(str(base));continue
        obj=json.loads(done.read_text())
        if obj.get('steps_completed')!=case['factors']['steps']:
            raise RuntimeError('Incomplete step budget for '+case['id'])
        key=case['experiment']
        if key=='E2':groups['E2'][case['id']]=str(best.resolve())
        if key=='E7':groups['E7'][case['id']]=str(best.resolve())
        if case['id']=='E2_reference':
            groups['E7']['E7_maternal_070_reference']=str(best.resolve())
        if key=='E11':
            for step in (5000,10000,20000):
                snap=base/f'step{step}.inference.pt'
                if not snap.is_file():
                    raise RuntimeError('Missing actual long-training checkpoint: '+str(snap))
                groups['E11'][f'E11_step{step}']=str(snap.resolve())
    if missing and require_all:
        raise RuntimeError('Controlled training cases must be fully executed before evaluation:\n'+'\n'.join(missing))
    return groups,missing


def run_cmd(module,args):
    cmd=[sys.executable,'-m','scripts.experiments.'+module]+list(map(str,args))
    print('EXECUTING:',module,' '.join(map(str,args[:10])),flush=True)
    subprocess.run(cmd,check=True)


def evaluate(a):
    verify=check_existing(a.manifest_dir,a.v1_dir,a.v2_dir,a.external_manifest)
    write_json(a.output/'PREFLIGHT.json',verify)
    if not verify['ready']:
        raise RuntimeError('Cannot run full clinical experiments without genuine frozen inputs:\n'+'\n'.join(verify['missing']))
    manifests=[a.manifest_dir/f'{task}.json' for task in TASKS]
    groups,missing=trained_groups(a.output/'training',a.require_controlled)
    args=['--manifests']+[str(p) for p in manifests]+['--v1-dir',str(a.v1_dir),
         '--v2-dir',str(a.v2_dir),'--save-predictions']
    logged=[]
    if a.external_manifest:
        run_cmd('run_predictions',['--experiment','E1','--manifests',a.external_manifest,
            '--v1-dir',a.v1_dir,'--v2-dir',a.v2_dir,'--save-predictions',
            '--output',a.output/'E1','--resume']);logged.append('E1')
    for phase in ('E3','E5','E6','E8'):
        extra=['--n-search',str(a.n_search)] if phase=='E3' else []
        run_cmd('run_predictions',['--experiment',phase]+args+extra+
               ['--output',a.output/phase,'--resume']);logged.append(phase)
    run_cmd('analyze_full_predictions',['--predictions',a.output/'E5'/'restricted_predictions',
             '--output',a.output/'E5'/'patient_analysis','--bootstraps',a.patient_bootstraps])
    for phase,mapping in groups.items():
        if not mapping:continue
        mp=a.output/'training'/f'{phase}_checkpoint_map.json'
        write_json(mp,mapping)
        run_cmd('run_predictions',['--experiment',phase]+args+['--methods']+list(mapping)+
            ['--checkpoint-map',mp,'--output',a.output/phase,'--resume']);logged.append(phase)
    if a.profile_all:
        for path in manifests:
            d=load_frozen(path)
            for size,draws in d.contexts.items():
                if not any(len(set(d.y[indices].tolist()))>1 for indices in draws):continue
                dest=a.output/'E4'/f'{d.name}_c{size}'/'matched_cpu.json'
                if dest.is_file():continue
                run_cmd('profile_matched',['--all','--manifest',path,'--context',size,
                       '--repeats',a.profile_repeats,'--warmup','10',
                       '--v1-dir',a.v1_dir,'--v2-dir',a.v2_dir,'--out',dest])
        logged.append('E4')
    if a.quantize:
        run_cmd('quantize_full',['--manifests']+[str(p) for p in manifests]+
                ['--checkpoint',checkpoint('tiny_mixture',a.v1_dir,a.v2_dir),
                 '--out',a.output/'E10','--repeats',str(a.profile_repeats)])
        logged.append('E10')
    if a.prospective_records or a.approved_protocol:
        if not a.prospective_records or not a.approved_protocol:
            raise RuntimeError('E12 requires BOTH genuine prospective records and approved protocol')
        run_cmd('analyze_prospective',['--actual-records',a.prospective_records,
                '--approved-protocol',a.approved_protocol,
                '--out',a.output/'E12'/'REAL_PROSPECTIVE_ANALYSIS.json',
                '--bootstrap',str(a.patient_bootstraps)])
        logged.append('E12_PROSPECTIVE_PREDICTIVE_ANALYSIS_NOT_CLINICAL_EFFICACY')
    if a.require_native_arm:
        if platform.machine().lower() not in ('arm64','aarch64','armv8l','armv7l'):
            raise RuntimeError('E9 physical ARM benchmark must run on a native ARM device')
        logged.append('E9_ARM_native_cpu_profile_observed')
    run_cmd('verify_campaign',['--output',a.output,'--manifest-dir',a.manifest_dir])
    write_json(a.output/'ACTUAL_EXECUTIONS.json',{
        'experiment_phases_executed':logged,'training_groups_available':{k:list(v) for k,v in groups.items()},
        'full_context_draws':True,'source_provenance':verify['frozen_sources'],
        'unavailable_training_cases':missing,'E12_prospective':
        'not performed without ethics approval and real prospective patient data',
        'E1_new_cohort_provided':bool(a.external_manifest),
        'does_not_claim_E1_to_E12_are_all_complete':True})
    print('The measured outputs have been written to:',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','train','evaluate','all','status'),default='check')
    p.add_argument('--manifest-dir',type=Path,default=Path('artifacts/manifests'))
    p.add_argument('--v1-dir',type=Path,default=Path('artifacts/results/aws_full_pretrain'))
    p.add_argument('--v2-dir',type=Path,default=Path('artifacts/results/aws_v2_pretrain'))
    p.add_argument('--output',type=Path,default=Path('artifacts/results/e1_e12'))
    p.add_argument('--external-manifest',type=Path,default=None)
    p.add_argument('--device',choices=('cuda','mps','cpu'),default='cuda')
    p.add_argument('--confirm-compute',action='store_true')
    p.add_argument('--require-controlled',action='store_true',
                   help='Fail unless all E2/E7/E11 pretrained models are available')
    p.add_argument('--n-search',type=int,default=8)
    p.add_argument('--patient-bootstraps',type=int,default=2000)
    p.add_argument('--profile-repeats',type=int,default=100)
    p.add_argument('--no-profile-all',dest='profile_all',action='store_false')
    p.add_argument('--quantize',action='store_true',help='Run full E10 ONNX INT8 across the frozen clinical tasks')
    p.add_argument('--require-native-arm',action='store_true')
    p.add_argument('--prospective-records',type=Path,default=None)
    p.add_argument('--approved-protocol',type=Path,default=None)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.phase=='check':
        report=check_existing(a.manifest_dir,a.v1_dir,a.v2_dir,a.external_manifest)
        groups,missing=trained_groups(a.output/'training',False)
        report['controlled_groups_ready']={x:len(v) for x,v in groups.items()}
        report['untrained_full_budget_cases']=missing
        write_json(a.output/'PREFLIGHT.json',report)
        print(json.dumps(report,indent=2))
        return 0 if report['ready'] else 2
    if a.phase=='all':
        before=check_existing(a.manifest_dir,a.v1_dir,a.v2_dir,a.external_manifest)
        write_json(a.output/'PREFLIGHT.json',before)
        if not before['ready']:
            raise RuntimeError('Refusing costly full pretraining before real evaluation inputs are verified:\n'+'\n'.join(before['missing']))
    if a.phase in ('train','all'):
        full_training(a.output/'training',a.device,a.confirm_compute)
    if a.phase in ('evaluate','all'):
        if a.phase=='all':a.require_controlled=True
        evaluate(a)
    if a.phase=='status':
        run_cmd('status',['--manifest-dir',a.manifest_dir,'--evidence-dir',a.output,
                          '--output',a.output/'status.json'])
    return 0

if __name__=='__main__':sys.exit(main())
