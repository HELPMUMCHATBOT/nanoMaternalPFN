#!/usr/bin/env python3
"""Prespecified paired benchmark contrasts on identical frozen context draws.

Bootstrap intervals resample *context draws*, not patients or independent
hospital cohorts; they are descriptive and not external-validation CIs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS=(
    ("tiny_generic","nanotabpfn_generic"),
    ("tiny_mixture","nanotabpfn_mixture"),
    ("tiny_mixture","tabpfn_v2_single"),
    ("tiny_mixture","tabpfn_v2_default"),
    ("base_mixture","tabpfn_v2_default"),
    ("tiny_mixture","xgboost"),
    ("tiny_mixture","lightgbm"),
    ("adaptive_tiny","tiny_mixture"),
    ("adaptive_base","base_mixture"),
    ("tiny_mixture","tiny_generic"),
    ("base_mixture","base_generic"),
)
METRICS=("auroc","accuracy","brier","ece15")
LOWER_BETTER={"brier","ece15"}


def bootstrap_interval(differences, *, seed=20261112, resamples=2000):
    values=np.asarray(differences,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):
        return float("nan"),float("nan")
    if len(values)==1:
        return float("nan"),float("nan")
    rng=np.random.default_rng(seed)
    offsets=rng.integers(0,len(values),(resamples,len(values)))
    means=values[offsets].mean(axis=1)
    return tuple(float(q) for q in np.quantile(means,[0.025,0.975]))


def evaluate_paired(raw, output, resamples=2000):
    frame=pd.read_csv(raw)
    keys=["dataset","context","seed"]
    if frame.duplicated(keys+["method"]).any():
        raise ValueError("Duplicate draw/method rows")
    rows=[]
    for a,b in CONTRASTS:
        left=frame.loc[frame.method==a]
        right=frame.loc[frame.method==b]
        if left.empty or right.empty:
            raise RuntimeError(f"Missing prespecified contrast {a} versus {b}")
        merged=left.merge(right,on=keys,suffixes=("_a","_b"),validate="one_to_one")
        for (dataset,context),part in merged.groupby(["dataset","context"]):
            for metric in METRICS:
                x=part[f"{metric}_a"].to_numpy(dtype=float)
                y=part[f"{metric}_b"].to_numpy(dtype=float)
                diff=(y-x) if metric in LOWER_BETTER else (x-y)
                diff=diff[np.isfinite(diff)]
                if not len(diff):
                    continue
                low,high=bootstrap_interval(
                    diff,seed=20261112+int(context),resamples=resamples
                )
                rows.append({
                    "dataset":dataset,"context":int(context),
                    "model_a":a,"model_b":b,
                    "metric":metric,"positive_favors":"model_a",
                    "n_shared_draws":len(diff),
                    "mean_directional_improvement":float(np.mean(diff)),
                    "ci2_5":low,"ci97_5":high,
                    "fraction_draws_better":float(np.mean(diff>0)),
                })
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    report={
        "rows":len(rows),"contrasts":[list(x) for x in CONTRASTS],
        "metrics":list(METRICS),"resamples":resamples,
        "inference":"descriptive bootstrap of frozen context draws; "
                    "not independent patient-level or external cohort uncertainty",
        "nonindependent_cohorts":[["vat_gdm","vat_delivery"]],
        "exploratory":True,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resamples",type=int,default=2000)
    args=p.parse_args()
    if args.resamples<100:
        p.error("resamples must be >=100")
    print(json.dumps(evaluate_paired(args.raw,args.output,args.resamples),
                     indent=2))


if __name__=="__main__":
    main()
