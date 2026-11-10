"""Execute low-data context sweep against frozen holdout indices."""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from .baselines import get_baseline
from .calibration import multiclass_brier,ece
from ..classifier import NanoMaternalPFNClassifier

def evaluate(manifest_path, checkpoint=None, methods=('logistic','random_forest'), output='artifacts/results/transfer.csv'):
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    npz=manifest_path.with_suffix('.npz')
    if hashlib.sha256(npz.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('frozen dataset SHA256 mismatch')
    data=np.load(npz,allow_pickle=False)
    X,y=data['X'],data['y']
    test=np.asarray(manifest['test_idx']);classes=np.unique(y)
    if len(classes) not in (2,3):
        raise ValueError('supports 2 or 3 classes')
    results=[]
    for method in methods:
        if method=='pfn' and not checkpoint:
            raise ValueError('pretrained checkpoint required for PFN comparison')
        for budget,draws in manifest['context_draws'].items():
            for seed,draw in enumerate(draws):
                draw=np.asarray(draw)
                if len(np.unique(y[draw])) < 2:
                    continue
                estimator=NanoMaternalPFNClassifier(checkpoint) if method=='pfn' else get_baseline(method,seed)
                estimator.fit(X[draw],y[draw])
                pred=estimator.predict_proba(X[test])
                aligned=np.zeros((len(test),len(classes)))
                for j,c in enumerate(estimator.classes_):
                    aligned[:,np.where(classes==c)[0][0]]=pred[:,j]
                scores={'method':method,'context':int(budget),'seed':seed,
                        'accuracy':float(accuracy_score(y[test],classes[aligned.argmax(axis=1)])),
                        'brier':multiclass_brier(np.searchsorted(classes,y[test]),aligned),
                        'ece15':ece(np.searchsorted(classes,y[test]),aligned)}
                try:
                    scores['auroc']=float(roc_auc_score(np.searchsorted(classes,y[test]),aligned[:,1] if len(classes)==2 else aligned,multi_class='ovr'))
                except ValueError:
                    scores['auroc']=float('nan')
                results.append(scores)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if results:
        with output.open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return results
