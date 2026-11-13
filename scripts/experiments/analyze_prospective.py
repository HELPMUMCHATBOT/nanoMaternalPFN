#!/usr/bin/env python3
"""E12: analysis of REAL ethics-approved prospective evaluation records.

This script NEVER enrolls participants or claims causal clinical effectiveness.
Model predictions must be timestamped before the actual recorded outcomes;
protocol/ethics approval must precede enrollment. No identifiers are exported.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from .common import sha256,write_json

REQUIRED=('participant_id','site_id','prediction_time','outcome_time','y_true','predicted_risk')
PROTOCOL_REQUIRED=('study_id','ethics_approval_reference','clinical_principal_investigator',
                   'model_sha256','approved_protocol_date','first_enrollment_date',
                   'model_frozen_before_first_enrollment','outcome_definition',
                   'risk_action_threshold','data_use_authorization')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual-records',type=Path,required=True)
    p.add_argument('--approved-protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    if a.bootstrap<1000:raise ValueError('Proper patient bootstrap needs at least 1000 replicates')
    protocol=json.loads(a.approved_protocol.read_text())
    missing=[k for k in PROTOCOL_REQUIRED if not protocol.get(k)]
    if missing:raise ValueError('Governance/protocol declarations missing: '+', '.join(missing))
    if protocol['model_frozen_before_first_enrollment'] is not True:
        raise RuntimeError('Model must be finalized before enrollment')
    if len(protocol['model_sha256'])!=64:raise ValueError('Model must have SHA-256 provenance')
    threshold=float(protocol['risk_action_threshold'])
    if not 0<threshold<1:raise ValueError('Decision threshold must be specified prospectively')
    frame=pd.read_csv(a.actual_records)
    if any(c not in frame.columns for c in REQUIRED):raise ValueError('Missing real prospective record fields')
    if len(frame)<2 or frame.participant_id.isna().any() or frame.participant_id.duplicated().any():
        raise ValueError('Patients must have unique identifiers; cannot treat repeated encounters as independent')
    predtime=pd.to_datetime(frame.prediction_time,utc=True,errors='raise')
    outtime=pd.to_datetime(frame.outcome_time,utc=True,errors='raise')
    approval=pd.to_datetime(protocol['approved_protocol_date'],utc=True)
    first=pd.to_datetime(protocol['first_enrollment_date'],utc=True)
    if approval>first or (predtime<first).any() or (outtime<=predtime).any():
        raise ValueError('Prospective temporal or ethics order invalid; verify study governance')
    if frame.site_id.isna().any():raise ValueError('Missing clinical site provenance')
    y=frame.y_true.to_numpy();risk=frame.predicted_risk.to_numpy(dtype=float)
    if not np.isin(y,[0,1]).all() or len(np.unique(y))<2:
        raise ValueError('Binary outcome classes required and must contain events and non-events')
    if not np.isfinite(risk).all() or (risk<0).any() or (risk>1).any():
        raise ValueError('Actual predicted risks must be finite probabilities in [0,1]')
    n=len(y);events=int(np.sum(y));n_sites=frame.site_id.nunique()
    point_auc=float(roc_auc_score(y,risk));point_brier=float(brier_score_loss(y,risk))
    rng=np.random.default_rng(20261113);aucs=[];briers=[]
    for i in range(a.bootstrap):
        sample=rng.integers(0,n,n)
        if np.unique(y[sample]).size<2:continue
        aucs.append(float(roc_auc_score(y[sample],risk[sample])))
        briers.append(float(brier_score_loss(y[sample],risk[sample])))
    if len(aucs)<1000:raise RuntimeError('Outcome too rare for requested patient bootstrap')
    bins=np.linspace(0,1,11)
    calibration=[]
    for j in range(10):
        ok=(risk>=bins[j])&(risk<bins[j+1] if j<9 else risk<=1)
        if ok.any():calibration.append({'risk_bin':j,'n':int(ok.sum()),
                        'mean_predicted_risk':float(risk[ok].mean()),
                        'observed_outcome_rate':float(y[ok].mean())})
    flagged=risk>=threshold
    result={'study_id':protocol['study_id'],'source':
              {'records_sha256':sha256(a.actual_records),
               'approved_protocol_sha256':sha256(a.approved_protocol),
               'locked_model_sha256':protocol['model_sha256']},
            'patients':n,'independent_sites':int(n_sites),'events':events,
            'auroc':point_auc,'patient_bootstrap_95ci':np.quantile(aucs,[.025,.975]).tolist(),
            'brier':point_brier,'brier_patient_bootstrap_95ci':np.quantile(briers,[.025,.975]).tolist(),
            'calibration_bins':calibration,'prespecified_risk_threshold':threshold,
            'flagged_count':int(flagged.sum()),
            'sensitivity_at_threshold':float(flagged[y==1].mean()),
            'specificity_at_threshold':float((~flagged)[y==0].mean()),
            'observational_predictive_analysis_only':True,
            'randomized_clinical_effectiveness_established':False,
            'ethics_provenance_is_author_attestation_not_external_confirmation':True}
    write_json(a.out,result)
    print('Completed real prospective predictive analysis for study',protocol['study_id'],
          'with',n,'patients. No causal treatment efficacy conclusion.')

if __name__=='__main__':main()
