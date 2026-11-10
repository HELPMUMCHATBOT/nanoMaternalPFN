#!/usr/bin/env python3
"""Fetch public uncredentialed datasets, audit schemas, freeze grouped splits.

No government DUA datasets, synthetic fallback data, or clinical pilot stage.
Writes a JSON source-and-split status report even if some sources fail.
"""
import argparse
import json
from nanomaternalpfn.data.open_sources import SOURCES,acquire_and_freeze

def main():
    p=argparse.ArgumentParser(description="Acquire and audit open maternal datasets")
    p.add_argument("--names",nargs="+",default=[
        "uci_maternal_risk","vat_gdm","vat_delivery",
        "low_birth_weight","cardiotocography","pima_diabetes",
        "tpehg_preterm","ctu_cord_acidemia",
    ],choices=sorted(SOURCES))
    p.add_argument("--output-dir",default="artifacts/manifests")
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--require-all",action="store_true",help="Nonzero exit if any specified dataset unavailable")
    args=p.parse_args()
    report=acquire_and_freeze(args.names,args.output_dir,seed=args.seed)
    for name,record in report["dataset_results"].items():
        print(f"{name}: {record['status']}" +
              (f" ({record['reason']})" if record["status"]!="frozen" else
               f" ({record['source']['rows']} records, {len(record['source']['features'])} encoded features)"))
    print(json.dumps({name:entry["status"] for name,entry in report["dataset_results"].items()},indent=2))
    if args.require_all and any(x["status"]!="frozen" for x in report["dataset_results"].values()):
        raise SystemExit(2)

if __name__=="__main__":
    main()
