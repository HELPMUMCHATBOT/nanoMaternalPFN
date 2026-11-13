#!/usr/bin/env python3
"""E1: freeze a *new actual* external clinical cohort with source attestation.

Never downloads, manufactures, samples, or substitutes synthetic clinical data.
User supplies reviewed real de-identified X/y in one NPZ and an independent
source/protocol attestation JSON. The original model must already be frozen.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import HISTORICAL_TASKS,EXTERNAL_FIELDS,load_frozen,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-npz',type=Path,required=True,help='Actual independently sourced X/y dataset, not synthetic')
    p.add_argument('--attestation',type=Path,required=True,help='Reviewed source and model-freeze metadata')
    p.add_argument('--output-manifest',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261113)
    a=p.parse_args()
    if a.output_manifest.stem in HISTORICAL_TASKS:
        raise RuntimeError('The original eight tasks cannot be registered as fresh external validation')
    metadata=json.loads(a.attestation.read_text())
    if any(k not in metadata for k in EXTERNAL_FIELDS):
        raise RuntimeError('External provenance and independent-review metadata incomplete')
    if not metadata.get('frozen_before_model_evaluation') or not metadata.get('independent_of_development_cohorts'):
        raise RuntimeError('An independent, previously unseen source is mandatory')
    if not a.real_npz.is_file():raise FileNotFoundError('Real deidentified dataset not found')
    with np.load(a.real_npz,allow_pickle=False) as file:
        if not all(x in file for x in ('X','y')):raise ValueError('NPZ must contain actual X and y')
        x=np.asarray(file['X'],dtype=np.float32);y=np.asarray(file['y'])
    if len(y)!=len(x) or len(y)<40 or len(np.unique(y)) not in (2,3):
        raise ValueError('Invalid real cohort: insufficient cases/labels or invalid dimensions')
    from nanomaternalpfn.data.freeze_splits import freeze
    # Existing source-verified group-safe freezing (patient/profile leakage protection).
    manifest=freeze(x,y,a.output_manifest,seed=a.seed,n_seeds=20,
         budgets=(16,32,64,128,256,512))
    manifest['external_validation']=metadata
    manifest['real_source_archive_sha256']=sha256(a.real_npz)
    write_json(a.output_manifest,manifest)
    verified=load_frozen(a.output_manifest,external=True)
    print(json.dumps({'name':verified.name,'real_records':len(verified.y),
            'source_sha256':verified.sha256,'test_records':len(verified.test),
            'context_draw_counts':{k:len(v) for k,v in verified.contexts.items()},
            'note':'Registered source is evaluated only after model freeze.'},indent=2))

if __name__=='__main__':main()
