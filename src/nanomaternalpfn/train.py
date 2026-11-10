"""Reproducible synthetic prior training; no claim of clinical validity."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import NanoMaternalPFN, ModelConfig
from .priors.maternal_scm import sample_task

def pretrain(out, model_cfg=None, prior='mixture', steps=100, batch_size=4, n_rows=64, n_features=8, n_ctx=32, lr=.0005, seed=20261107, device='cpu'):
    if not 1 <= n_ctx < n_rows:
        raise ValueError('context size must be smaller than total rows')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = model_cfg or ModelConfig()
    model=NanoMaternalPFN(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    model.train()
    last_loss=None
    for step in range(steps):
        batch=[sample_task(n_rows,n_features,cfg.max_classes,prior,rng=rng) for _ in range(batch_size)]
        x=torch.as_tensor(np.stack([t[0] for t in batch]),device=device)
        y=torch.as_tensor(np.stack([t[1] for t in batch]),device=device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x,y[:,:n_ctx])
        loss=F.cross_entropy(logits.reshape(-1,cfg.max_classes),y[:,n_ctx:].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        last_loss=float(loss.detach().cpu())
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':asdict(cfg),'prior':prior,'steps':steps,'seed':seed,'synthetic_only':True},out)
    return {'checkpoint':str(out),'last_loss':last_loss,'parameter_count':model.parameter_count(),'synthetic_only':True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='artifacts/checkpoints/tiny_maternal.pt')
    p.add_argument('--prior',choices=['maternal','generic','mixture'],default='mixture')
    p.add_argument('--tier',choices=['tiny','small','base'],default='tiny')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    cfg={'tiny':ModelConfig(),'small':ModelConfig(128,4,4,256),'base':ModelConfig(192,6,6,384)}[a.tier]
    print(json.dumps(pretrain(a.out,cfg,a.prior,a.steps,a.batch_size,device=a.device),indent=2))
if __name__=='__main__':
    main()
