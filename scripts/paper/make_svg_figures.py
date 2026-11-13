#!/usr/bin/env python3
"""Generate publication-quality vector figures solely from audited metric CSVs.

Numbers are loaded from frozen benchmark summaries, never generated or inferred.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt


def render(source: Path, target: Path) -> tuple[Path, Path, Path]:
    target.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(source / 'leaderboard_all.csv').sort_values('auroc')
    if not {'adaptive_tiny_3expert', 'tabpfn_v2_default'}.issubset(set(leaderboard.method)):
        raise ValueError('Missing measured core comparison')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars=ax.barh(leaderboard.method, leaderboard.auroc)
    ax.set_xlabel('Equal-task macro AUROC (observed; higher is better)')
    ax.set_xlim(0, max(0.76, float(leaderboard.auroc.max())+0.02))
    ax.set_title('Frozen eight-task benchmark: discrimination')
    ax.grid(axis='x',alpha=0.2)
    for bar, score in zip(bars,leaderboard.auroc):
        ax.text(float(score)+0.003,bar.get_y()+bar.get_height()/2,f'{score:.3f}',va='center',fontsize=8)
    fig.tight_layout()
    a=target/'full_auroc_leaderboard.svg';fig.savefig(a,format='svg',metadata={'Date':None});plt.close(fig)

    tasks=pd.read_csv(source/'primary_task_effects.csv')
    tasks=tasks[(tasks.comparator=='tabpfn_v2_default')&(tasks.metric=='auroc')].sort_values('effect')
    if len(tasks)!=8:
        raise ValueError('Expected eight observed tasks for primary comparison')
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.barh(tasks.dataset,tasks.effect)
    ax.axvline(0,linewidth=1)
    ax.set_xlabel('Paired task mean AUROC: three-expert Tiny minus TabPFN v2 Default')
    ax.set_title('Heterogeneous observed effects; task level, no patient CIs')
    ax.grid(axis='x',alpha=0.2)
    fig.tight_layout()
    b=target/'task_auroc_effects.svg';fig.savefig(b,format='svg',metadata={'Date':None});plt.close(fig)

    effects=pd.read_csv(source/'three_expert_contrasts_all_metrics.csv')
    effects=effects[(effects.metric=='auroc')].sort_values('equal_task_effect')
    if len(effects)!=12:
        raise ValueError('Expected twelve measured AUROC comparators')
    fig,ax=plt.subplots(figsize=(10,6))
    y=np.arange(len(effects))
    x=effects.equal_task_effect.to_numpy(float)
    lo=effects.cohort_boot_ci_lower.to_numpy(float)
    hi=effects.cohort_boot_ci_upper.to_numpy(float)
    ax.errorbar(x,y,xerr=np.vstack([x-lo,hi-x]),fmt='o',capsize=2)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(y,labels=effects.comparator)
    ax.set_xlabel('Mean paired AUROC difference (+ favors three-expert Tiny)')
    ax.set_title('Exploratory seven-source bootstrap sensitivity intervals')
    ax.grid(axis='x',alpha=0.2)
    fig.tight_layout()
    c=target/'cohort_auroc_effects.svg';fig.savefig(c,format='svg',metadata={'Date':None});plt.close(fig)
    return a,b,c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in render(a.source,a.output): print(path)

if __name__=='__main__':main()
