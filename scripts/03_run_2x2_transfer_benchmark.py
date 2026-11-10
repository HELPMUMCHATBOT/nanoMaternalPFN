#!/usr/bin/env python3
import argparse
from nanomaternalpfn.evaluation.transfer_curves import evaluate
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',default=None);p.add_argument('--methods',nargs='+',default=['logistic','random_forest']);a=p.parse_args()
print('Runs:',len(evaluate(a.manifest,a.checkpoint,methods=a.methods)))
