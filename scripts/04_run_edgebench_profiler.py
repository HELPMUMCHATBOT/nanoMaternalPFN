#!/usr/bin/env python3
import json
import torch
from nanomaternalpfn.model import NanoMaternalPFN
from nanomaternalpfn.edgebench.profiler import profile
m=NanoMaternalPFN();x=torch.randn(1,33,6);y=torch.zeros((1,32),dtype=torch.long)
print(json.dumps(profile(m,x,y),indent=2))
print('IMPORTANT: untrained weights; latency measurement only, not predictive efficacy.')
