#!/usr/bin/env python3
"""E5: analyze all *real* captured binary patient-level comparisons.

All patients and probabilities must originate from the actual frozen source
and final pretrained checkpoint. Repeated contexts share patients and are NOT
independent cohort replicates. Multiclass cases are retained as exclusions,
not silently supplied with invented binary labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .analyze_predictions import paired_bootstrap
from .common import jsonl_append,write_json,sha256

COMPARISONS = [('tiny_generic','tabpfn_v2_default'),
               ('tiny_mixture','tabpfn_v2_default')]


def collect(directory, output, bootstraps=2000):
    if bootstraps < 100:raise ValueError('At least 100 bootstrap draws')
    directory,output=Path(directory),Path(output)
    if not directory.is_dir():raise FileNotFoundError('Real patient prediction directory not found: '+str(directory))
    output.mkdir(parents=True,exist_ok=True)
    journal=output/'real_patient_analysis.jsonl'
    saved=set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line)
            saved.add((row['task'],row['context'],row['seed'],row['model_a'],row['model_b']))
    measured=0;excluded=[]; missing=[]
    for a,b in COMPARISONS:
        for first in sorted(directory.glob(f'*.{a}.c*.s*.clean.npz')):
            prefix,_,suffix=first.name.partition(f'.{a}.')
            expected=directory/(prefix+f'.{b}.'+suffix)
            if not expected.is_file():
                missing.append(str(expected));continue
            c=int(suffix.split('.')[0].lstrip('c'));seed=int(suffix.split('.')[1].lstrip('s'))
            key=(prefix,c,seed,a,b)
            if key in saved:continue
            try:
                r=paired_bootstrap(first,expected,n=bootstraps)
            except ValueError as exc:
                if 'binary' in str(exc).lower() or 'both outcome classes' in str(exc).lower():
                    excluded.append({'task':prefix,'context':c,'seed':seed,'reason':str(exc)})
                    continue
                raise
            record={'task':prefix,'context':c,'seed':seed,'model_a':a,'model_b':b,
                    'patient_analysis':r,'source_sha256_model_a':sha256(first),
                    'source_sha256_model_b':sha256(expected)}
            jsonl_append(journal,record);saved.add(key);measured+=1
    write_json(output/'ANALYSIS_STATUS.json',{'new_measured_binary_comparisons':measured,
        'total_measured_binary_comparisons':len(saved),
        'excluded_multiclass_cases':excluded,'missing_pairs':missing,
        'not_independent_cohorts':True,'based_on_original_real_patient_predictions':True})
    if missing:raise RuntimeError(f'Incomplete paired prediction files: {len(missing)} missing')
    print('Completed E5 binary patient comparisons:',len(saved))
    return len(saved)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstraps',type=int,default=2000)
    a=p.parse_args();collect(a.predictions,a.output,a.bootstraps)

if __name__=='__main__':main()
