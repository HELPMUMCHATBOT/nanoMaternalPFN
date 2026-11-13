"""Shared evidence contracts for E1--E12: never synthesize clinical observations."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

HISTORICAL_TASKS = frozenset({
    'uci_maternal_risk','vat_gdm','vat_delivery','low_birth_weight',
    'cardiotocography','pima_diabetes','tpehg_preterm','ctu_cord_acidemia'
})
COHORT = {'vat_gdm':'physionet_vat','vat_delivery':'physionet_vat'}
EXTERNAL_FIELDS = ('source_accession','population','outcome_definition','data_license',
                   'frozen_before_model_evaluation','model_selection_freeze_sha',
                   'independent_of_development_cohorts','independent_reviewer')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')


def jsonl_append(path,entry):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n')


@dataclass
class FrozenData:
    name:str
    source:str
    x:np.ndarray
    y:np.ndarray
    test:np.ndarray
    contexts:dict
    sha256:str
    manifest:dict


def load_frozen(manifest_path,*,external=False):
    manifest_path=Path(manifest_path)
    if not manifest_path.is_file(): raise FileNotFoundError(f'Frozen manifest missing: {manifest_path}')
    m=json.loads(manifest_path.read_text())
    path=manifest_path.with_suffix('.npz')
    if not path.is_file():raise FileNotFoundError(f'Frozen patient dataset missing: {path}')
    digest=sha256(path)
    if m.get('sha256')!=digest:raise ValueError('Frozen source NPZ SHA256 mismatch')
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z['X'],dtype=np.float32)
        y=np.asarray(z['y'])
    if x.ndim!=2 or y.ndim!=1 or len(y)!=len(x):raise ValueError('X/y shape mismatch')
    if not np.issubdtype(y.dtype,np.integer):raise ValueError('Outcome must use integer class labels')
    if not np.isfinite(np.nan_to_num(x,nan=0)).all():raise ValueError('Infinite feature values')
    test=np.asarray(m.get('test_idx',[]),dtype=np.int64)
    if not len(test) or len(set(test.tolist()))!=len(test) or test.min()<0 or test.max()>=len(y):
        raise ValueError('Invalid frozen test indices')
    if len(np.unique(y)) not in (2,3):raise ValueError('PFN benchmark requires 2 or 3 classes')
    contexts=m.get('context_draws',{})
    if not contexts:raise ValueError('No frozen context draws')
    used_test=set(test.tolist())
    for budget,draws in contexts.items():
        if int(budget)<1:raise ValueError('Invalid context size')
        for draw in draws:
            indices=np.asarray(draw,dtype=int)
            if len(indices)!=int(budget) or len(set(draw))!=len(draw):
                raise ValueError('Wrong context size or duplicate row')
            if np.any(indices<0) or np.any(indices>=len(y)) or used_test.intersection(draw):
                raise ValueError('Frozen context overlaps held-out test or out of range')
    name=manifest_path.stem
    if external:
        if name in HISTORICAL_TASKS:
            raise ValueError('Previously examined task cannot be called independent external validation')
        info=m.get('external_validation',{})
        missing=[f for f in EXTERNAL_FIELDS if not info.get(f)]
        if missing: raise ValueError('External source provenance incomplete: '+', '.join(missing))
        if not isinstance(info['model_selection_freeze_sha'],str) or len(info['model_selection_freeze_sha'])!=40:
            raise ValueError('Model freeze must be a 40-character commit SHA')
        if not isinstance(info['frozen_before_model_evaluation'],bool) or not info['frozen_before_model_evaluation']:
            raise ValueError('External test must be frozen before model evaluation')
        if not isinstance(info['independent_of_development_cohorts'],bool) or not info['independent_of_development_cohorts']:
            raise ValueError('External test must have an independent source cohort')
        source=info['source_accession']
    else:
        source=COHORT.get(name,name)
    return FrozenData(name,source,x,y,test,contexts,digest,m)


def aligned_probs(estimator, probs, classes):
    probs=np.asarray(probs,dtype=np.float64)
    from_classes=np.asarray(estimator.classes_)
    classes=np.asarray(classes)
    if probs.ndim!=2 or probs.shape[1]!=len(from_classes) or len(set(from_classes))!=len(from_classes):
        raise ValueError('Invalid class-probability shape')
    out=np.zeros((len(probs),len(classes)),dtype=np.float64)
    for col,c in enumerate(from_classes):
        loc=np.where(classes==c)[0]
        if len(loc)!=1:raise ValueError('Predicted unseen class')
        out[:,loc[0]]=probs[:,col]
    if not np.isfinite(out).all() or np.any(out<0) or not np.allclose(out.sum(1),1,atol=1e-4):
        raise ValueError('Invalid predicted probabilities')
    return out


def metrics(y,p,classes=None):
    from sklearn.metrics import accuracy_score,roc_auc_score
    from nanomaternalpfn.evaluation.calibration import multiclass_brier,ece
    classes=np.unique(y) if classes is None else np.asarray(classes)
    if len(classes)!=p.shape[1]:
        raise ValueError("Predictor classes do not align with metric classes")
    if not np.isin(y,classes).all():
        raise ValueError("Unexpected held-out target label")
    truth=np.searchsorted(classes,y)
    score={'accuracy':float(accuracy_score(truth,p.argmax(1))),
           'brier':float(multiclass_brier(truth,p)),
           'ece15':float(ece(truth,p))}
    try:score['auroc']=float(roc_auc_score(truth,p[:,1] if len(classes)==2 else p,multi_class='ovr'))
    except ValueError:score['auroc']=None
    return score


def provenance(input_manifest,method,checkpoint_paths=()):
    return {'dataset_npz_sha256':sha256(Path(input_manifest).with_suffix('.npz')),
            'dataset_manifest_sha256':sha256(input_manifest), 'method':method,
            'checkpoint_sha256':{str(p):sha256(p) for p in checkpoint_paths}}


def discover_missing_inputs(manifest_dir,checkpoints):
    manifest_dir=Path(manifest_dir)
    return {'frozen_manifest_pairs':[x for name in sorted(HISTORICAL_TASKS)
                                      for x in (str(manifest_dir/f'{name}.json'),str(manifest_dir/f'{name}.npz'))
                                      if not Path(x).is_file()],
            'checkpoint_files':[str(p) for p in checkpoints if not Path(p).is_file()]}
