#!/usr/bin/env python3
"""E4/E9 matched single-query CPU benchmark on actual frozen tasks.

Each method runs in its own fresh process (with --all) to avoid inherited RSS.
No synthetic accuracy results, and no Android/ARM claim unless genuinely
executed natively on the declared device and architecture.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .common import load_frozen,write_json,sha256
from .predictors import build_classical,make_expert,checkpoint,ContextOnlyExpertEnsemble,EXPERTS


def one_method(args):
 import psutil
 import nanomaternalpfn
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1',None):
    raise ValueError('This is a CPU comparison; disable CUDA_VISIBLE_DEVICES')
 process=psutil.Process()
 baseline_rss=process.memory_info().rss/1e6
 data=load_frozen(args.manifest)
 draws=data.contexts.get(str(args.context))
 if not draws:raise ValueError('Context not available for frozen task')
 indices=np.asarray(draws[0],dtype=int)
 if len(np.unique(data.y[indices]))<2:raise ValueError('First context has fewer than 2 classes')
 x,y=data.x[indices],data.y[indices]
 query=data.x[data.test[:1]]
 start=time.perf_counter()
 ckpts=[]
 if args.method in ('logistic','random_forest','xgboost','lightgbm'):
   model=build_classical(args.method,args.seed)[0]
 elif args.method.startswith('tabpfn_v2_'):
   from nanomaternalpfn.evaluation.baselines import get_baseline
   model=get_baseline(args.method,seed=args.seed,tabpfn_device='cpu')
 elif args.method=='adaptive_tiny_3expert':
   model=ContextOnlyExpertEnsemble(EXPERTS,args.v1_dir,args.v2_dir,seed=args.seed)
   ckpts=[checkpoint(n,args.v1_dir,args.v2_dir) for n in EXPERTS]
 elif args.method=='adaptive_tiny_2expert':
   from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
   ckpts=[checkpoint('v1_tiny_mixture',args.v1_dir,args.v2_dir),
          checkpoint('tiny_mixture',args.v1_dir,args.v2_dir)]
   model=AdaptiveNanoMaternalPFNClassifier(ckpts,device='cpu',query_batch_size=16)
 else:
   model=make_expert(args.method,args.v1_dir,args.v2_dir)
   ckpts=[checkpoint(args.method,args.v1_dir,args.v2_dir)]
 model_load_ms=(time.perf_counter()-start)*1000
 start=time.perf_counter();model.fit(x,y)
 fit_ms=(time.perf_counter()-start)*1000
 for _ in range(args.warmup):model.predict_proba(query)
 rss_after_warmup=process.memory_info().rss/1e6
 timings=[];rss_samples=[]
 for i in range(args.repeats):
   start=time.perf_counter();probs=model.predict_proba(query)
   timings.append((time.perf_counter()-start)*1000)
   rss_samples.append(process.memory_info().rss/1e6)
 probs=np.asarray(probs)
 if probs.ndim!=2 or len(probs)!=1:raise ValueError('Predictor shape incorrect')
 return {'method':args.method,'dataset':data.name,'dataset_sha256':data.sha256,
    'context':args.context,'query_rows':1,'features':data.x.shape[1],
    'warmup':args.warmup,'repeats':args.repeats,'device':'cpu',
    'processor':platform.processor(),'architecture':platform.machine(),
    'platform':platform.platform(),'python':platform.python_version(),
    'cpu_count':os.cpu_count(),'torch_num_threads':os.getenv('OMP_NUM_THREADS'),
    'model_load_ms':model_load_ms,'context_fit_ms':fit_ms,
    'query_p50_ms':float(np.percentile(timings,50)),
    'query_p95_ms':float(np.percentile(timings,95)),
    'query_p99_ms':float(np.percentile(timings,99)),
    'query_mean_ms':float(np.mean(timings)),
    'initial_process_rss_mb':baseline_rss,
    'rss_after_context_fit_and_warmup_mb':rss_after_warmup,
    'observed_rss_peak_mb':float(max(rss_samples+[rss_after_warmup])),
    'observed_rss_increment_mb':float(max(rss_samples+[rss_after_warmup])-baseline_rss),
    'checkpoint_total_mb':sum(Path(p).stat().st_size for p in ckpts)/1e6,
    'checkpoint_sha256':{str(p):sha256(p) for p in ckpts},
    'scope':'native-process CPU benchmark; observed RSS snapshots, not peak allocator high-water',
    'mobile_device_validated':platform.system()=='Android'}


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--v1-dir',type=Path,default=Path('artifacts/results/aws_full_pretrain'))
 p.add_argument('--v2-dir',type=Path,default=Path('artifacts/results/aws_v2_pretrain'))
 p.add_argument('--context',type=int,default=64);p.add_argument('--seed',type=int,default=20261113)
 p.add_argument('--method',default='tiny_mixture')
 p.add_argument('--methods',nargs='+',default=['tiny_mixture','tiny_generic','adaptive_tiny_2expert',
                         'adaptive_tiny_3expert','tabpfn_v2_default','logistic','random_forest','xgboost','lightgbm'])
 p.add_argument('--all',action='store_true');p.add_argument('--warmup',type=int,default=10)
 p.add_argument('--repeats',type=int,default=100);p.add_argument('--out',type=Path,required=True)
 args=p.parse_args(argv)
 if args.context<1 or args.warmup<1 or args.repeats<10:raise ValueError('Invalid benchmark settings')
 if args.all:
   outputs=[]
   for method in args.methods:
     cmd=[sys.executable,'-m','scripts.experiments.profile_matched','--manifest',str(args.manifest),
          '--v1-dir',str(args.v1_dir),'--v2-dir',str(args.v2_dir),
          '--context',str(args.context),'--seed',str(args.seed),
          '--warmup',str(args.warmup),'--repeats',str(args.repeats),
          '--method',method,'--out',str(args.out.parent/(method+'.json'))]
     env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
     proc=subprocess.run(cmd,capture_output=True,text=True,env=env)
     if proc.returncode:raise RuntimeError(f'{method} isolated benchmark failed:\n{proc.stderr[-4000:]}')
     outputs.append(json.loads((args.out.parent/(method+'.json')).read_text()))
   write_json(args.out,{'matched_protocol':True,'method_results':outputs})
 else:write_json(args.out,one_method(args))
 print(f'Hardware results: {args.out}')

if __name__=='__main__':main()
