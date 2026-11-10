from scipy.stats import wilcoxon

def paired_wilcoxon(a,b):
    if len(a)!=len(b):raise ValueError('paired scores require matching tasks/seeds')
    stat,p=wilcoxon(a,b)
    return {'statistic':float(stat),'p_value':float(p)}
