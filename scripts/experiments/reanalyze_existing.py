#!/usr/bin/env python3
"""Execute additional *descriptive* E6/E7 diagnostics on previously measured draws.

NO mechanism ablation is claimed: existing checkpoint comparisons differ in more
than one factor. Input must be the exact existing review bundle with verified
internal SHA256 manifest. These outputs contain no newly collected patients.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ZIP_SHA='f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26'
TASKS=['uci_maternal_risk','vat_gdm','vat_delivery','low_birth_weight','cardiotocography','pima_diabetes','tpehg_preterm','ctu_cord_acidemia']
COHORT={t:('vat' if t.startswith('vat_') else t) for t in TASKS}
CONTRASTS={
  'E6_observed_3_vs_2_expert':(('paper_three_expert_full','adaptive_tiny_3expert'),('paper_completion_full','adaptive_tiny')),
  'E7_observed_v2_tiny_generic_vs_70pct_mixture':(('downstream_v2_full','tiny_generic'),('downstream_v2_full','tiny_mixture')),
  'E7_observed_v2_base_generic_vs_70pct_mixture':(('downstream_v2_full','base_generic'),('downstream_v2_full','base_mixture')),
}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--bundle',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();actual=hashlib.sha256(a.bundle.read_bytes()).hexdigest()
 if actual!=ZIP_SHA:raise ValueError('Unexpected source ZIP SHA256; refusing to analyze altered archive')
 rows=[];cell_rows=[]
 with zipfile.ZipFile(a.bundle) as z:
  manifest=json.loads(z.read('EVIDENCE_SHA256_MANIFEST.json'))
  for name in z.namelist():
   if name=='EVIDENCE_SHA256_MANIFEST.json':continue
   z.getinfo(name).CRC # CRC is checked during read below
  for contrast,((dir_a,method_a),(dir_b,method_b)) in CONTRASTS.items():
   tasks=[]
   for t in TASKS:
    name_a=f'results/{dir_a}/{t}.{method_a}.csv';name_b=f'results/{dir_b}/{t}.{method_b}.csv'
    if name_a not in z.namelist() or name_b not in z.namelist():
     raise ValueError('Missing measured draw table '+name_a+' or '+name_b)
    def read_member(name):
     raw=z.read(name)
     return pd.read_csv(io.BytesIO(raw)).drop(columns=['method'],errors='ignore')
    left=read_member(name_a);right=read_member(name_b)
    paired=left.merge(right,on=['context','seed'],how='inner',validate='one_to_one',suffixes=('_a','_b'))
    if len(paired)==0:raise ValueError('No matched measured draws')
    if len(paired)!=len(left) or len(paired)!=len(right):raise ValueError('Missing matched draw; no imputation')
    for metric in ('auroc','brier','ece15','accuracy'):
     mult=-1 if metric in ('brier','ece15') else 1
     diffs=mult*(paired[f'{metric}_a']-paired[f'{metric}_b'])
     if not np.isfinite(diffs).all():continue
     ctx=paired.assign(diff=diffs).groupby('context').agg(mean=('diff','mean'),draws=('diff','count')).reset_index()
     for _,v in ctx.iterrows():cell_rows.append({'contrast':contrast,'task':t,'cohort':COHORT[t],
         'metric':metric,'context':int(v.context),'n_matched_draws':int(v.draws),
         'positive_favors':'model_a','mean_diff':float(v['mean'])})
     tasks.append({'contrast':contrast,'task':t,'cohort':COHORT[t],
                   'metric':metric,'n_matched_draws':len(diffs),
                   'task_effect':float(ctx['mean'].mean())})
   rows.extend(tasks)
 frame=pd.DataFrame(rows);cells=pd.DataFrame(cell_rows)
 a.output.mkdir(parents=True,exist_ok=True)
 frame.to_csv(a.output/'existing_task_effects.csv',index=False)
 cells.to_csv(a.output/'existing_context_effects.csv',index=False)
 summary=[]
 for (contrast,metric),group in frame.groupby(['contrast','metric']):
  effects=group.groupby('cohort').task_effect.mean()
  summary.append({'contrast':contrast,'metric':metric,'source':'frozen_measured_draws',
                  'scope':'descriptive_posthoc_not_mechanism_isolation',
                  'n_tasks':len(group),'n_distinct_cohorts':len(effects),
                  'n_matched_draws_each_model':int(group.n_matched_draws.sum()),
                  'equal_task_delta':float(group.task_effect.mean()),
                  'n_task_wins':int((group.task_effect>0).sum())})
 pd.DataFrame(summary).to_csv(a.output/'existing_descriptive_summary.csv',index=False)
 (a.output/'STATUS.json').write_text(json.dumps({'E6':'retrospective_descriptive_completed_only',
      'E7':'observed_generic_vs_mixture_comparison_completed_only',
      'source_zip_sha256':ZIP_SHA,'causal_ablation_complete':False,
      'patient_level_data_available':False},indent=2)+'\n')
 print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
