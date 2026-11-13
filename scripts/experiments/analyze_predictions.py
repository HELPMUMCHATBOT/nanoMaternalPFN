#!/usr/bin/env python3
"""E5 patient-level ROC uncertainty and calibration on REAL stored probabilities.

Analyzes one frozen test set/draw per comparison. Never treats repeated context
seeds sharing the same patients as independent held-out participants.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.linear_model import LogisticRegression


def read_predictions(p):
    with np.load(p,allow_pickle=False) as z:
        y=z['y_true'];probs=z['probs'];classes=z['classes'];ids=z['test_row_index']
    if len(y)!=len(ids) or len(y)!=len(probs):raise ValueError('Prediction dimension mismatch')
    if len(np.unique(ids))!=len(ids):raise ValueError('Test row IDs must be unique')
    if len(classes)!=2:raise ValueError('Binary calibration slope and paired AUROC implemented for binary outcomes only')
    target=(y==classes[1]).astype(int)
    if target.min()==target.max():raise ValueError('Both outcome classes required')
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1,atol=1e-4):
        raise ValueError('Invalid probability matrix')
    return ids,target,probs[:,1]


def paired_bootstrap(first,second,n=5000,seed=20261113):
    ids1,y1,p1=read_predictions(first)
    ids2,y2,p2=read_predictions(second)
    if not np.array_equal(ids1,ids2) or not np.array_equal(y1,y2):
        raise ValueError('Paired predictions must refer to exactly the same test rows and labels')
    observed=float(roc_auc_score(y1,p1)-roc_auc_score(y1,p2))
    rng=np.random.default_rng(seed);d=[]
    for _ in range(n):
        idx=rng.integers(0,len(y1),len(y1))
        if np.unique(y1[idx]).size!=2:continue
        d.append(float(roc_auc_score(y1[idx],p1[idx])-roc_auc_score(y1[idx],p2[idx])))
    if len(d)<max(100,n//2):raise ValueError('Too few valid patient bootstrap draws')
    x=np.clip(p1,1e-6,1-1e-6);logit=np.log(x/(1-x))
    slope=LogisticRegression(C=1e6,solver='lbfgs').fit(logit.reshape(-1,1),y1)
    bins=np.minimum((x*10).astype(int),9)
    calibration=[{'bin':i,'n':int((bins==i).sum()),
                  'predicted':float(x[bins==i].mean()),'observed':float(y1[bins==i].mean())}
                 for i in range(10) if (bins==i).any()]
    return {'n_patients':len(y1),'n_events':int(y1.sum()),
            'observed_paired_auroc_difference':observed,
            'patient_bootstrap_ci':np.quantile(d,[.025,.975]).tolist(),
            'bootstrap_valid_replicates':len(d),'calibration_slope_model_a':float(slope.coef_[0][0]),
            'calibration_intercept_model_a':float(slope.intercept_[0]),
            'brier_model_a':float(brier_score_loss(y1,p1)),
            'brier_model_b':float(brier_score_loss(y2,p2)),
            'calibration_bins_model_a':calibration,
            'scope':'single frozen test set, single context/draw; not independent-source confirmation'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-a',type=Path,required=True)
 p.add_argument('--model-b',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--bootstraps',type=int,default=5000);a=p.parse_args()
 if a.bootstraps<100:raise ValueError('At least 100 bootstrap replicates required')
 r=paired_bootstrap(a.model_a,a.model_b,a.bootstraps)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
