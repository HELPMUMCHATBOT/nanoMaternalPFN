#!/usr/bin/env python3
"""E2/E7/E11 controlled training matrix; plan first, execute only on real compute."""
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def make_matrix():
 base={'architecture':'pre_ln_v2','class_augmentation':True,'sign_augmentation':True,
       'missingness_encoding':'native','schedule':'constant','maternal_probability':.7,'steps':5000}
 cases=[]
 def add(case,experiment,**changes):
  cases.append({'id':case,'experiment':experiment,'factors':{**base,**changes},
    'design':'one_factor_at_a_time','test_data_used_for_model_selection':False})
 add('E2_reference','E2')
 add('E2_postln','E2',architecture='post_ln_v1')
 add('E2_no_class_aug','E2',class_augmentation=False)
 add('E2_no_sign_aug','E2',sign_augmentation=False)
 add('E2_no_missing_indicator','E2',missingness_encoding='imputed_no_indicator')
 add('E2_cosine','E2',schedule='cosine')
 for weight in [0,.25,.50,.70,1.0]:
  if weight==.70:continue # reference case already provides 70:30
  add('E7_maternal_{:03d}'.format(int(weight*100)),'E7',maternal_probability=weight)
 cases.append({'id':'E11_long_pretraining','experiment':'E11','factors':{**base,'steps':20000},
         'design':'same_trajectory_checkpoints_at_5000_10000_20000',
         'test_data_used_for_model_selection':False})
 return cases


def case_command(case,out,device,python=sys.executable):
 f=case['factors'];cmd=[python,'-m','scripts.experiments.train_controlled',
 '--output',str(Path(out)/case['id']),'--architecture',f['architecture'],
 '--schedule',f['schedule'],'--missingness-encoding',f['missingness_encoding'],
 '--maternal-probability',str(f['maternal_probability']),
 '--steps',str(f['steps']),'--device',device]
 for k in ('class_augmentation','sign_augmentation'):
  cmd.append('--'+k.replace('_','-') if f[k] else '--no-'+k.replace('_','-'))
 return cmd


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=Path('artifacts/results/e2_e7_e11'))
 p.add_argument('--execute-case',default=None)
 p.add_argument('--device',choices=('cuda','cpu','mps'),default='cuda')
 p.add_argument('--confirm-compute',action='store_true',help='Acknowledge compute cost and actual model training')
 a=p.parse_args();cases=make_matrix();a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'MATRIX.json').write_text(json.dumps(cases,indent=2)+'\n')
 print(f'{len(cases)} full-size training configurations in matrix (E2/E7/E11).')
 if a.execute_case:
  if not a.confirm_compute:raise SystemExit('Refusing to run unapproved compute: add --confirm-compute')
  case=next((c for c in cases if c['id']==a.execute_case),None)
  if case is None:raise SystemExit('Unknown case '+a.execute_case)
  cmd=case_command(case,a.output,a.device)
  print('Executing:',shlex.join(cmd),flush=True)
  subprocess.run(cmd,check=True)
 else:
  for case in cases:print('PLANNED',case['id'],' '.join(shlex.quote(c) for c in case_command(case,a.output,a.device)))

if __name__=='__main__':main()
