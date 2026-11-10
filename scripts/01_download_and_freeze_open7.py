#!/usr/bin/env python3
"""Freeze a user-supplied, source-verified CSV. No synthetic stand-ins."""
import argparse
from nanomaternalpfn.data.registry import load_local, REGISTRY
from nanomaternalpfn.data.freeze_splits import freeze

def main():
    parser=argparse.ArgumentParser(description='Audited local open dataset split freezer')
    parser.add_argument('--dataset',required=True,choices=sorted(REGISTRY))
    parser.add_argument('--directory',default='data/raw')
    parser.add_argument('--target',default=None)
    args=parser.parse_args()
    x,y,metadata=load_local(args.dataset,args.directory,args.target)
    manifest=freeze(x,y,f'artifacts/manifests/{args.dataset}.json')
    print(metadata,manifest['audit'])
if __name__=='__main__':main()
