import numpy as np

def mcar(x, probability=0.05, seed=None):
    rng = np.random.default_rng(seed)
    out = np.array(x, dtype=np.float32, copy=True)
    out[rng.random(out.shape) < probability] = np.nan
    return out
