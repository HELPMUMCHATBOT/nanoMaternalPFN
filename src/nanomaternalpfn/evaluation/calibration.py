"""Calibration metrics for classification."""
import numpy as np

def multiclass_brier(y_true, probs):
    p=np.asarray(probs,float)
    y=np.asarray(y_true,int)
    t=np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p-t)**2,axis=1)))

def ece(y_true, probs, n_bins=15):
    p=np.asarray(probs,float)
    y=np.asarray(y_true,int)
    confidence=p.max(axis=1)
    correct=(p.argmax(axis=1)==y)
    edges=np.linspace(0,1,n_bins+1)
    result=0.
    for i in range(n_bins):
        subset=(confidence>=edges[i]) & (confidence<=edges[i+1] if i==n_bins-1 else confidence<edges[i+1])
        if subset.any():
            result+=subset.mean()*abs(correct[subset].mean()-confidence[subset].mean())
    return float(result)
