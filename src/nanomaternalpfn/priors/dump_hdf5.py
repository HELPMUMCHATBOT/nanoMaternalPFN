"""Generate reproducible labeled synthetic HDF5 tables."""
from pathlib import Path
import h5py
import numpy as np
from .maternal_scm import sample_task

def dump(path, n_tables=1000, n_rows=128, n_features=10, n_classes=3, prior='mixture', seed=20261107):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as f:
        f.attrs.update({'source':'synthetic', 'prior':prior, 'seed':seed, 'clinical_validated':False})
        xx = f.create_dataset('X', (n_tables,n_rows,n_features), dtype='f4', chunks=(1,n_rows,n_features), compression='gzip')
        yy = f.create_dataset('y', (n_tables,n_rows), dtype='i8', chunks=(1,n_rows), compression='gzip')
        for i in range(n_tables):
            x,y,_ = sample_task(n_rows,n_features,n_classes,prior,rng=rng)
            xx[i],yy[i] = x,y
    return path
