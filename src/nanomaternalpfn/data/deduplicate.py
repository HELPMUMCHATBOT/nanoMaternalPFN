"""Group exact feature profiles; never overwrite contradictory patient labels."""
import numpy as np
import pandas as pd

def profile_groups(x, decimals=6):
    frame = pd.DataFrame(np.asarray(x)).round(decimals)
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    _, inverse = np.unique(hashes, return_inverse=True)
    return inverse

def audit_profiles(x, y):
    groups = profile_groups(x)
    labels = np.asarray(y)
    conflicts = sum(len(set(labels[groups == k])) > 1 for k in np.unique(groups))
    return {'n_rows': len(labels), 'n_profiles': len(np.unique(groups)), 'conflicting_profiles': conflicts,
            'n_duplicate_rows': len(labels)-len(np.unique(groups))}
