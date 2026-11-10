#!/usr/bin/env python3
import argparse
from nanomaternalpfn.priors.dump_hdf5 import dump
from nanomaternalpfn.train import pretrain
p=argparse.ArgumentParser();p.add_argument('--tables',type=int,default=100);p.add_argument('--steps',type=int,default=10);p.add_argument('--prior',default='mixture',choices=['maternal','generic','mixture']);a=p.parse_args()
print('Synthetic dump:',dump(f'artifacts/priors_h5/{a.prior}.h5',n_tables=a.tables,prior=a.prior))
print('Training:',pretrain('artifacts/checkpoints/tiny_maternal.pt',steps=a.steps,prior=a.prior))
