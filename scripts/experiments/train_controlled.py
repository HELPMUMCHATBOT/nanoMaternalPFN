#!/usr/bin/env python3
"""E2/E7/E11: controlled synthetic PFN training with factor-isolating switches.

Real clinical data are NEVER used by this module. A result is valid as a full
pretraining experiment only when the prespecified step budget is completed and
checkpoint selection uses held-out synthetic validation loss only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .common import write_json


def configuration(args):
    return {'family':'nanomaternalpfn_controlled','tier':args.tier,
       'architecture':args.architecture,'class_augmentation':args.class_augmentation,
       'sign_augmentation':args.sign_augmentation,'missingness_encoding':args.missingness_encoding,
       'schedule':args.schedule,'maternal_probability':args.maternal_probability,
       'steps':args.steps,'batch_size':args.batch_size,'microbatch_size':args.microbatch_size,
       'seed':args.seed,'validation_seed':args.validation_seed,'max_context':args.max_context,
       'query_rows':args.query_rows,'validate_every':args.validate_every,
       'val_tables_per_budget':args.val_tables_per_budget,
       'save_steps':args.save_steps,'device':args.device,'peak_lr':args.peak_lr}


def prepare_task(rows,features,n_classes,prob,seed,aug_class,aug_sign,context,missingness):
    from nanomaternalpfn.priors.maternal_scm import sample_task
    rng=np.random.default_rng(seed)
    x,y,_=sample_task(rows,features,n_classes,prior='mixture',rng=rng,
                      maternal_probability=prob,
                      augment_class_permutation=aug_class,
                      augment_feature_sign_flip=aug_sign)
    if missingness=='imputed_no_indicator':
        # Control: remove explicit missingness information, using ONLY context means.
        center=np.nanmean(x[:context],axis=0)
        center=np.nan_to_num(center,nan=0.)
        x=np.where(np.isnan(x),center[None,:],x).astype('float32')
    return x,y


def loss_on_batch(model,x,context_y,query_y,k):
    import torch
    from torch.nn import functional as F
    logits=model(x,context_y,n_classes=k)
    return F.cross_entropy(logits.reshape(-1,k),query_y.reshape(-1))


def make_batch(step,sub,seed,max_context,query_rows,prob,augment_class,augment_sign,missingness,batch,device):
    import torch
    rng=np.random.default_rng(np.random.SeedSequence([seed,step,sub,991]))
    budgets=[c for c in (16,32,64,128,256,512) if c<=max_context]
    context=int(rng.choice(budgets));features=int(rng.integers(5,17));k=int(rng.choice([2,3]))
    xx=[];yy=[]
    for b in range(batch):
        task_seed=np.random.SeedSequence([seed,step,sub,b,747])
        x,y=prepare_task(context+query_rows,features,k,prob,task_seed,
                   augment_class,augment_sign,context,missingness)
        xx.append(x);yy.append(y)
    x=torch.as_tensor(np.stack(xx),device=device,dtype=torch.float32)
    y=torch.as_tensor(np.stack(yy),device=device,dtype=torch.long)
    return x,y[:,:context],y[:,context:],k


def val_loss(model,step,cfg):
    import torch
    rng=np.random.default_rng(cfg['validation_seed'])
    budgets=[b for b in (16,32,64,128,256,512) if b<=cfg['max_context']]
    totals=[]
    was_training=model.training
    model.eval()
    with torch.no_grad():
        for context in budgets:
            scores=[]
            for j in range(cfg['val_tables_per_budget']):
                seed=np.random.SeedSequence([cfg['validation_seed'],context,j,202])
                random_state=np.random.default_rng(seed)
                features=int(random_state.integers(5,17));k=int(random_state.choice([2,3]))
                x,y=prepare_task(context+cfg['query_rows'],features,k,cfg['maternal_probability'],seed,
                      cfg['class_augmentation'],cfg['sign_augmentation'],context,cfg['missingness_encoding'])
                tx=torch.as_tensor(x[None],device=cfg['device']);ty=torch.as_tensor(y[None],device=cfg['device'])
                scores.append(float(loss_on_batch(model,tx,ty[:,:context],ty[:,context:],k).item()))
            totals.append(float(np.mean(scores)))
    if was_training:model.train()
    return {'step':step,'contexts':budgets,'validation_loss_by_context':totals,
            'validation_mean_loss':float(np.mean(totals)),
            'validation_set':'disjoint_seed_synthetic_only'}


def train(args):
    import torch
    from nanomaternalpfn.model import NanoMaternalPFN
    from nanomaternalpfn.train_replay import TIERS
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no NVIDIA GPU is accessible; no silent CPU fallback')
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but not available')
    if args.microbatch_size<1 or args.batch_size%args.microbatch_size:
        raise ValueError('Microbatch must divide effective batch size')
    if args.validation_seed==args.seed:raise ValueError('Validation stream must be independent')
    if args.steps<1 or args.query_rows<1:raise ValueError('Positive training dimensions needed')
    if not 0<=args.maternal_probability<=1:raise ValueError('Invalid synthetic prior ratio')
    cfg=configuration(args)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():
        raise ValueError('Training already complete; refusing to overwrite research checkpoints')
    spec=args.output/'CONFIG.json'
    if spec.exists() and json.loads(spec.read_text())!=cfg:
        raise ValueError('Existing run has different training configuration')
    write_json(spec,cfg)
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    mcfg=replace(TIERS[args.tier],architecture=args.architecture)
    model=NanoMaternalPFN(mcfg).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.peak_lr)
    best=float('inf');best_step=None;history=[];start_step=0
    resume_path=args.output/'resume_state.pt'
    if args.resume and resume_path.is_file():
        snapshot=torch.load(resume_path,map_location=args.device,weights_only=False)
        if snapshot.get('config')!=cfg:
            raise RuntimeError('Unsafe resume: controlled training configuration changed')
        model.load_state_dict(snapshot['model'])
        optimizer.load_state_dict(snapshot['optimizer'])
        best=float(snapshot['best']);best_step=snapshot['best_step']
        history=list(snapshot['validation_history']);start_step=int(snapshot['step'])
        torch.set_rng_state(snapshot['torch_rng_state'])
        if args.device=='cuda' and snapshot.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(snapshot['cuda_rng_states'])
        if start_step>=args.steps:
            raise RuntimeError('Training state reached target steps but COMPLETE.json is absent; audit the interrupted run')
        print(f'RESUMING complete-budget training from {start_step}/{args.steps}',flush=True)
    elif args.resume and spec.exists() and not (args.output/'best.inference.pt').exists():
        # A fresh run with only its CONFIG.json is valid. Never infer that its
        # pretraining has completed from the configuration alone.
        pass
    elif not args.resume and spec.exists() and (args.output/'best.inference.pt').exists():
        raise RuntimeError('Interrupted experiment already exists; use --resume')
    def save_inference(path,step,loss,complete):
        from nanomaternalpfn.model import ModelConfig
        blob={'state_dict':model.state_dict(),'config':asdict(mcfg),
              'variant':'controlled_experiment_v2', 'step':step,
              'synthetic_only':True,'validation_loss':loss,
              'complete':complete,'factor_spec':cfg,
              'feature_mode':cfg['missingness_encoding']}
        torch.save(blob,path)
    for step in range(start_step+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        if args.schedule=='constant':lr=args.peak_lr
        elif args.schedule=='cosine':
            warmup=min(250,max(1,args.steps//10))
            progress=max(0,(step-warmup)/max(1,args.steps-warmup))
            lr=args.peak_lr*(step/warmup if step<=warmup else (.05+.95*.5*(1+math.cos(math.pi*progress))))
        else:raise ValueError('Unknown schedule')
        for group in optimizer.param_groups:group['lr']=lr
        for sub in range(args.batch_size//args.microbatch_size):
            x,y,query,k=make_batch(step,sub,args.seed,args.max_context,args.query_rows,
             args.maternal_probability,args.class_augmentation,args.sign_augmentation,
             args.missingness_encoding,args.microbatch_size,args.device)
            (loss_on_batch(model,x,y,query,k)/(args.batch_size//args.microbatch_size)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%args.validate_every==0 or step==args.steps or step in args.save_steps:
            val=val_loss(model,step,cfg);history.append(val)
            if val['validation_mean_loss']<best:
                best=val['validation_mean_loss'];best_step=step
                save_inference(args.output/'best.inference.pt',step,best,complete=(step==args.steps))
            if step in args.save_steps:
                save_inference(args.output/f'step{step}.inference.pt',step,val['validation_mean_loss'],complete=(step==args.steps))
            write_json(args.output/'synthetic_validation.json',history)
            # Persist both weights and optimizer for an exact continuation.
            # The task RNG is keyed by (seed, step, microbatch, item), so no
            # synthetic draws are repeated after restoration.
            snapshot={'config':cfg,'step':step,'model':model.state_dict(),
                      'optimizer':optimizer.state_dict(),'best':best,
                      'best_step':best_step,'validation_history':history,
                      'torch_rng_state':torch.get_rng_state(),
                      'cuda_rng_states':torch.cuda.get_rng_state_all() if args.device=='cuda' else None}
            temporary=resume_path.with_suffix('.tmp')
            torch.save(snapshot,temporary);temporary.replace(resume_path)
            print(f'SYNTHETIC TRAIN step={step}/{args.steps} val_loss={val["validation_mean_loss"]:.5f}',flush=True)
    save_inference(args.output/'final.inference.pt',args.steps,history[-1]['validation_mean_loss'],complete=True)
    # The best-validation checkpoint may have been selected before the final
    # optimization step. Its selection step is retained, but the metadata now
    # also proves that the complete prescribed budget was run.
    bestpath=args.output/'best.inference.pt'
    finalbest=torch.load(bestpath,map_location='cpu',weights_only=True)
    finalbest['full_budget_completed']=True
    finalbest['total_training_steps']=args.steps
    torch.save(finalbest,bestpath)
    write_json(args.output/'COMPLETE.json',{'steps_completed':args.steps,'best_synthetic_validation_step':best_step,
             'best_synthetic_validation_loss':best,'method':'controlled_synthetic_pretraining',
             'paper_results_produced':False,'requires_frozen_clinical_transfer':True,
             'weights_selection':'independent_synthetic_validation_only'})


def cli(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--tier',choices=('tiny','base'),default='tiny')
 p.add_argument('--architecture',choices=('pre_ln_v2','post_ln_v1'),default='pre_ln_v2')
 p.add_argument('--class-augmentation',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--sign-augmentation',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--missingness-encoding',choices=('native','imputed_no_indicator'),default='native')
 p.add_argument('--schedule',choices=('constant','cosine'),default='constant')
 p.add_argument('--maternal-probability',type=float,default=.7)
 p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=32)
 p.add_argument('--microbatch-size',type=int,default=8);p.add_argument('--query-rows',type=int,default=8)
 p.add_argument('--max-context',type=int,default=512);p.add_argument('--seed',type=int,default=20261107)
 p.add_argument('--validation-seed',type=int,default=20271008)
 p.add_argument('--validate-every',type=int,default=250);p.add_argument('--val-tables-per-budget',type=int,default=16)
 p.add_argument('--save-steps',nargs='*',type=int,default=[5000,10000,20000])
 p.add_argument('--peak-lr',type=float,default=5e-4)
 p.add_argument('--device',choices=('cuda','cpu','mps'),default='cuda')
 p.add_argument('--resume',action='store_true',help='Resume exact optimizer state from prior checkpoint')
 a=p.parse_args(argv)
 train(a)

if __name__=='__main__':cli()
