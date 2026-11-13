#!/usr/bin/env python3
"""E1/E3/E5/E6/E8 benchmark runner with frozen held-out patients.

No clinical source fetch, hidden synthetic fallback, test-label model selection,
or automatic external-validation declaration. Only real NPZ manifests accepted.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .common import load_frozen,metrics,aligned_probs,write_json,jsonl_append,provenance
from .predictors import build_classical,tuned_baseline,make_expert,ContextOnlyExpertEnsemble,CLASSICAL,SINGLE,EXPERTS,EXPERT_RULES,checkpoint


def perturb(x,context_x,condition,seed):
    x=np.array(x,copy=True)
    if condition=='clean':return x
    rng=np.random.default_rng(seed)
    if condition.startswith('missing_'):
        fraction=float(condition.split('_',1)[1]);
        if not 0<=fraction<=1:raise ValueError('Missingness must be in [0,1]')
        x[rng.random(x.shape)<fraction]=np.nan;return x
    if condition.startswith('noise_'):
        fraction=float(condition.split('_',1)[1]);
        if fraction<0:raise ValueError('Noise cannot be negative')
        scale=np.nanstd(context_x,axis=0);scale=np.nan_to_num(scale,nan=0)
        x += rng.normal(size=x.shape).astype(np.float32)*scale[None,:]*fraction
        return x
    raise ValueError('Unknown perturbation: '+condition)


def thin_minority_context(indices,labels,fraction,seed):
    """Simulated class-prevalence stress, never a new real cohort."""
    if not 0<fraction<=1:raise ValueError('Must retain a nonzero minority fraction')
    values,counts=np.unique(labels,return_counts=True)
    if len(values)<2:raise ValueError('At least two context classes required')
    minority=values[int(np.argmin(counts))]
    low=np.where(labels==minority)[0];high=np.where(labels!=minority)[0]
    keep=max(2,int(round(len(low)*fraction)))
    if len(low)<2 or len(high)<2:
        raise ValueError('Insufficient minority/majority patients for imbalance stress')
    rng=np.random.default_rng(seed)
    selected=np.sort(np.r_[high,rng.choice(low,size=min(keep,len(low)),replace=False)])
    return np.asarray(indices)[selected]


def _parse(a):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=('E1','E3','E5','E6','E8'),required=True)
    p.add_argument('--manifests',type=Path,nargs='+',required=True)
    p.add_argument('--v1-dir',type=Path,default=Path('artifacts/results/aws_full_pretrain'))
    p.add_argument('--v2-dir',type=Path,default=Path('artifacts/results/aws_v2_pretrain'))
    p.add_argument('--methods',nargs='+',default=None)
    p.add_argument('--checkpoint-map',type=Path,default=None,help='JSON name-to-checkpoint map for E2/E7/E11 trained arms')
    p.add_argument('--contexts',type=int,nargs='+',default=None)
    p.add_argument('--max-draws',type=int,default=None,help='Diagnostic-only cap; omit for paper runs')
    p.add_argument('--n-search',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261113)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p.parse_args(a)


def run(args):
    args=_parse(args)
    if args.max_draws is not None and args.max_draws<1:raise ValueError('Positive draws only')
    if args.n_search<1:raise ValueError('Positive tuning iterations only')
    methods=args.methods or ({'E1':['tiny_generic','tiny_mixture','tabpfn_v2_default'],
        'E3':list(CLASSICAL),'E5':['tiny_generic','tiny_mixture','tabpfn_v2_default'],
        'E6':list(EXPERT_RULES),'E8':['tiny_generic','tiny_mixture','tabpfn_v2_default']}[args.experiment])
    custom_map=json.loads(args.checkpoint_map.read_text()) if args.checkpoint_map else {}
    if any(not isinstance(name,str) or not isinstance(path,str) for name,path in custom_map.items()):
        raise ValueError('Checkpoint map must contain string names and paths')
    conditions=['clean','missing_0.10','missing_0.25','noise_0.10','noise_0.25','imbalance_minority_0.50'] if args.experiment=='E8' else ['clean']
    cfg={'experiment':args.experiment,'manifests':{str(p):provenance(p,'input')['dataset_npz_sha256'] for p in args.manifests},
         'methods':methods,'contexts':args.contexts,'max_draws':args.max_draws,'n_search':args.n_search,
         'seed':args.seed,'save_predictions':args.save_predictions,'conditions':conditions,'checkpoint_map_sha256':__import__('hashlib').sha256(args.checkpoint_map.read_bytes()).hexdigest() if args.checkpoint_map else None}
    if args.output.exists() and (args.output/'RUN_CONFIG.json').exists():
        if json.loads((args.output/'RUN_CONFIG.json').read_text())!=cfg:raise ValueError('Existing output config differs')
        if not args.resume:raise ValueError('Existing run: --resume or use a different directory')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'RUN_CONFIG.json',cfg)
    done=set()
    journal=args.output/'measured_draws.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            r=json.loads(line);done.add((r['dataset'],r['method'],r['context'],r['draw_seed'],r['condition']))
    count=0
    for manifest in args.manifests:
        data=load_frozen(manifest,external=args.experiment=='E1')
        all_classes=np.unique(data.y)
        for ctx_str,all_draws in sorted(data.contexts.items(),key=lambda p:int(p[0])):
            budget=int(ctx_str)
            if args.contexts and budget not in args.contexts:continue
            draws=all_draws[:args.max_draws] if args.max_draws else all_draws
            for draw_seed,indices in enumerate(draws):
                indices=np.asarray(indices,dtype=int)
                labels=data.y[indices]
                if len(np.unique(labels))<2:continue
                for method in methods:
                    if method in CLASSICAL:
                        model,info=tuned_baseline(method,data.x[indices],labels,args.seed+draw_seed,args.n_search) if args.experiment=='E3' else (build_classical(method,args.seed+draw_seed)[0],{'model':'baseline_fixed'})
                        if args.experiment!='E3':model.fit(data.x[indices],labels)
                        paths=[]
                    elif method in custom_map:
                        path=Path(custom_map[method]);model=make_expert(method,args.v1_dir,args.v2_dir,override_path=path)
                        model.fit(data.x[indices],labels);paths=[path];info={'controlled_checkpoint':str(path)}
                    elif method in SINGLE:
                        model=make_expert(method,args.v1_dir,args.v2_dir)
                        paths=[checkpoint(method,args.v1_dir,args.v2_dir)]
                        model.fit(data.x[indices],labels);info={}
                    elif method.startswith('tabpfn_v2_'):
                        from nanomaternalpfn.evaluation.baselines import get_baseline
                        model=get_baseline(method,seed=args.seed+draw_seed,tabpfn_device='cpu')
                        model.fit(data.x[indices],labels);paths=[];info={'comparator':'official_TabPFN_v2'}
                    elif args.experiment=='E6' and method in EXPERT_RULES:
                        model=ContextOnlyExpertEnsemble(EXPERTS,args.v1_dir,args.v2_dir,rule=method,seed=args.seed+draw_seed)
                        model.fit(data.x[indices],labels);info=model.diagnostics_
                        paths=[checkpoint(n,args.v1_dir,args.v2_dir) for n in EXPERTS]
                    else:raise ValueError('Unknown method '+method)
                    for condition in conditions:
                        key=(data.name,method,budget,draw_seed,condition)
                        if key in done:continue
                        if condition.startswith('imbalance_minority_'):
                            try:
                                selected=thin_minority_context(indices,labels,.50,args.seed+1009*budget+draw_seed)
                            except ValueError:
                                jsonl_append(args.output/'skipped.jsonl',{'dataset':data.name,'method':method,
                                     'context':budget,'draw_seed':draw_seed,'condition':condition,
                                     'reason':'too_few_minority_context_rows'})
                                continue
                            model.fit(data.x[selected],data.y[selected])
                            query=data.x[data.test]
                        else:
                            query=perturb(data.x[data.test],data.x[indices],condition,args.seed+1009*budget+draw_seed)
                        probs=aligned_probs(model,model.predict_proba(query),all_classes)
                        result={'experiment':args.experiment,'dataset':data.name,'source_cohort':data.source,
                           'method':method,'context':budget,'draw_seed':draw_seed,
                           'condition':condition,'n_test':len(data.test),
                           'n_context_used':len(selected) if condition.startswith('imbalance_minority_') else len(indices),
                           'metrics':metrics(data.y[data.test],probs),
                           'selection':info,'provenance':provenance(manifest,method,paths),
                           'diagnostic_draw_cap':args.max_draws is not None}
                        if args.save_predictions:
                            # No raw features, patient names or identifiers are exported.
                            predpath=args.output/'restricted_predictions'/f'{data.name}.{method}.c{budget}.s{draw_seed}.{condition}.npz'
                            predpath.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(predpath,y_true=data.y[data.test],probs=probs,classes=all_classes,
                                                test_row_index=data.test)
                            result['predictions_local_path']=str(predpath)
                            result['predictions_sha256']=provenance_predict_file(predpath)
                        jsonl_append(journal,result);done.add(key);count+=1
                        print(f'MEASURED {data.name} {method} ctx={budget} draw={draw_seed} {condition}: {result["metrics"]}',flush=True)
    write_json(args.output/'COMPLETION.json',{'newly_measured_records':count,
                 'total_completed_records':len(done),'scope':args.experiment,
                 'all_draws_requested':args.max_draws is None,
                 'external_confirmation':args.experiment=='E1',
                 'caveat':'Do not claim E1 independent validation without independent source attestation and locked model selection.'})
    return count


def provenance_predict_file(path):
    from .common import sha256
    return sha256(path)

if __name__=='__main__':run(sys.argv[1:])
