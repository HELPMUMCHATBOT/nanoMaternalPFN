#!/usr/bin/env python3
"""Merge original v2 and newly collected paper comparator metrics safely."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,
                   default=Path("artifacts/results/downstream_v2_full/summary.csv"))
    p.add_argument("--additional",type=Path,
                   default=Path("artifacts/results/paper_completion_full/summary.csv"))
    p.add_argument("--output",type=Path,
                   default=Path("artifacts/results/paper_all_methods_summary.csv"))
    args=p.parse_args()
    original=pd.read_csv(args.original)
    added=pd.read_csv(args.additional)
    result=pd.concat([original,added],ignore_index=True)
    keys=["dataset","method","context","metric"]
    duplicates=result.duplicated(keys,keep=False)
    if duplicates.any():
        raise ValueError("Duplicate comparator rows:\n"+str(result.loc[duplicates,keys]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f"SAVED: {len(result)} summary rows to {args.output}")


if __name__=="__main__":
    main()
