import numpy as np

def net_benefit(y, risk, threshold):
    if not 0 < threshold < 1:raise ValueError('threshold must be between 0 and 1')
    y=np.asarray(y);pred=np.asarray(risk)>=threshold
    return float(((pred & (y==1)).sum()-(pred & (y==0)).sum()*threshold/(1-threshold))/len(y))

def group_summary(y, risk, groups):
    result={}
    for group in np.unique(groups):
        mask=np.asarray(groups)==group
        result[str(group)]={'n':int(mask.sum()),'observed_rate':float(np.asarray(y)[mask].mean()),'mean_predicted_risk':float(np.asarray(risk)[mask].mean())}
    return result
