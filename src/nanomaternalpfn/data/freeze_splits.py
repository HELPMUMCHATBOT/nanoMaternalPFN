"""Group-safe reproducible holdouts with SHA256 provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .deduplicate import audit_profiles, profile_groups

def freeze(x, y, output, seed=20261107, test_fraction=.3, budgets=(16,32,64,128,256,512), n_seeds=20):
    x,y = np.asarray(x),np.asarray(y)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError('array shape mismatch')
    groups = profile_groups(x)
    if np.unique(groups).size < 4:
        raise ValueError('too few unique profiles')
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(x,y,groups))
    assert not set(groups[train_idx]).intersection(groups[test_idx])
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    data_path=output.with_suffix('.npz')
    np.savez_compressed(data_path, X=x.astype(np.float32),y=y,train_idx=train_idx,test_idx=test_idx)
    context_draws = {}
    for b in budgets:
        if b > len(train_idx):
            continue
        draws=[]
        for i in range(n_seeds):
            rng=np.random.default_rng(seed + b*1009 + i)
            draws.append(rng.choice(train_idx, size=b, replace=False).tolist())
        context_draws[str(b)] = draws
    manifest={'source':'real_external_dataset_requires_provenance','seed':seed,
              'npz':data_path.name,'sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'audit':audit_profiles(x,y),'n_train':len(train_idx),'n_test':len(test_idx),
              'context_draws':context_draws,'test_idx':test_idx.tolist()}
    output.write_text(json.dumps(manifest,indent=2))
    return manifest
