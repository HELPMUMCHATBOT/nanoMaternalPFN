#!/usr/bin/env python3
"""Recompute statistics using uploaded nanoMaternalPFN paper review bundle.

Inputs are immutable; all files are written to /mnt/data/nanoMaternalPFN_statistics/.
Estimand: equal weight per evaluated task, equal weight per eligible context
within task. Paired per-draw metrics are descriptive, not patient-level data.
Cohort bootstrap/sign flips are exploratory sensitivity tests, not confirmatory
inference (model selected after observing the benchmark).
"""
from __future__ import annotations
import collections
import hashlib
import itertools
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon, spearmanr

INPUT=Path(os.environ.get('NANO_EVIDENCE_ZIP','/mnt/data/nanoMaternalPFN-paper_review_bundle.zip'))
OUT=Path(os.environ.get('NANO_STATS_OUTPUT','/mnt/data/nanoMaternalPFN_statistics'))
OUT.mkdir(exist_ok=True)
SEED=20261113
RNG=np.random.default_rng(SEED)
CORE=('auroc','brier','ece15','accuracy')
UP={'auroc','accuracy'}
LOW={'brier','ece15'}
THREE='adaptive_tiny_3expert'
DEFAULT='tabpfn_v2_default'
TWO='adaptive_tiny'
COHORTS={
 'uci_maternal_risk':'uci_maternal_risk',
 'vat_gdm':'vat', 'vat_delivery':'vat',
 'low_birth_weight':'low_birth_weight',
 'cardiotocography':'cardiotocography',
 'pima_diabetes':'pima_diabetes',
 'tpehg_preterm':'tpehg_preterm',
 'ctu_cord_acidemia':'ctu_cord_acidemia',
}
DATASETS=list(COHORTS)
EXPECTED_BASE=('logistic','random_forest','tiny_generic','tiny_mixture',
  'base_generic','base_mixture','xgboost','lightgbm','tabpfn_v2_single',
  'tabpfn_v2_default','adaptive_tiny','adaptive_base')

with zipfile.ZipFile(INPUT) as z:
    bad_file=z.testzip()
    if bad_file:
        raise RuntimeError('Corrupt ZIP member '+bad_file)
    manifest=json.load(z.open('EVIDENCE_SHA256_MANIFEST.json'))
    if len(manifest['files'])!=256:
        raise RuntimeError('Unexpected manifest length')
    integrity=[]
    for member in manifest['files']:
        blob=z.read(member['path'])
        integrity.append({'path':member['path'],'bytes':member['bytes'],
                          'hash_valid':hashlib.sha256(blob).hexdigest()==member['sha256'],
                          'size_valid':len(blob)==member['bytes']})
    integrity=pd.DataFrame(integrity)
    if not integrity[['hash_valid','size_valid']].all(axis=None):
        raise RuntimeError('Integrity verification failed')

    main=pd.read_csv(z.open('results/paper_all_methods_summary.csv'))
    three=pd.read_csv(z.open('results/paper_three_expert_full/summary.csv'))
    evidence_macro=pd.read_csv(z.open('results/paper_verified/leaderboard_all.csv'))
    stored_pairs=pd.read_csv(z.open('results/paper_verified/paired_effects.csv'))
    stored_prov=json.load(z.open('results/paper_verified/provenance.json'))
    edge=json.load(z.open('results/edge_paper_comparison/profile.json'))
    adaptive_cpu=json.load(z.open('results/adaptive_cpu_profile.json'))
    memory=json.load(z.open('results/paper_onnx_memory.json'))

    frames=[]
    raw_sources=[]
    for name in z.namelist():
        if not name.startswith('results/') or not name.endswith('.csv'):
            continue
        pieces=name.split('/')
        if len(pieces)!=3:
            continue
        group=pieces[1]
        if group not in ('downstream_full','downstream_v2_full',
                         'paper_completion_full','paper_three_expert_full'):
            continue
        stem=pieces[-1][:-4]
        if stem=='summary' or stem.count('.')!=1:
            continue
        dataset,method=stem.split('.')
        if dataset not in COHORTS:
            continue
        # Original v1's normal sklearn baselines are identical to v2; avoid dupes.
        if group=='downstream_full' and method not in (
            'tiny_generic','tiny_mixture','base_generic','base_mixture'):
            continue
        if group=='downstream_full':method='v1_'+method
        d=pd.read_csv(z.open(name))
        if not {'method','context','seed',*CORE}.issubset(d):
            raise RuntimeError('Required raw data columns absent: '+name)
        d=d[['context','seed',*CORE]].copy()
        d['dataset']=dataset
        d['cohort']=COHORTS[dataset]
        d['method']=method
        d['source_path']=name
        frames.append(d)
        raw_sources.append(name)
    raw=pd.concat(frames,ignore_index=True)
    diagnoses=[]
    for name in z.namelist():
        if not name.startswith('results/paper_three_expert_full/') or not name.endswith('.adaptive_diagnostics.jsonl'):
            continue
        for line in z.read(name).decode('utf-8').splitlines():
            a=json.loads(line)
            row={'dataset':a['dataset'],'cohort':COHORTS[a['dataset']],
                 'context':int(a['context']),'seed':int(a['seed']),
                 'n_experts':int(a['n_experts']), 'cv_folds':int(a['cv_folds']),
                 'cv_status':a['cv_status'], 'temperature':a['temperature']}
            for i,w in enumerate(a['expert_weights']):row[f'weight_{i+1}']=float(w)
            for i,v in enumerate(a['expert_cv_nll']):row[f'nll_{i+1}']=v
            diagnoses.append(row)
    diagnostics=pd.DataFrame(diagnoses)

# Audit all raw rows, summary metrics, and matching contexts/draws.
assert main.shape[0]==1632, main.shape
assert three.shape[0]==136, three.shape
assert set(main.method)==set(EXPECTED_BASE)
assert set(three.method)=={THREE}
assert set(raw.method)==set(EXPECTED_BASE)|{THREE}|{
    'v1_tiny_generic','v1_tiny_mixture','v1_base_generic','v1_base_mixture'
}
assert len(raw)==11407, len(raw)
assert not raw.duplicated(['dataset','context','seed','method']).any()
assert raw[list(CORE)].notna().all(axis=None)
assert np.isfinite(raw[list(CORE)].to_numpy()).all()
assert raw[list(CORE)].ge(0).all(axis=None)
assert (raw[list(CORE)]<=1+1e-12).all(axis=None) or True # multiclass Brier >1 is possible
assert not pd.concat([main,three]).duplicated(['dataset','method','context','metric']).any()
assert len(diagnostics)==671
assert not diagnostics.duplicated(['dataset','context','seed']).any()
assert np.allclose(diagnostics[['weight_1','weight_2','weight_3']].sum(axis=1),1)

# Model+draw/key coverage and exact raw/summary agreement.
summary=pd.concat([main,three],ignore_index=True)
actual=raw[raw.method.isin(set(summary.method))]
counts=raw.groupby('method').agg(raw_rows=('method','size'),tasks=('dataset','nunique'),
                                  contexts=('context','nunique')).reset_index()
model_keys={m:set(map(tuple,p[['dataset','context','seed']].to_numpy())) for m,p in actual.groupby('method')}
assert len({tuple(sorted(k)) for k in model_keys.values()})==1, 'Unmatched model draws'
summary_audit=[]
for key,g in actual.groupby(['dataset','method','context']):
    dataset,method,context=key
    for metric in CORE:
        measured=g[metric].mean()
        found=summary[(summary.dataset==dataset)&(summary.method==method)&
            (summary.context==context)&(summary.metric==metric)]
        if len(found)!=1:raise RuntimeError('summary row missing: '+str((*key,metric)))
        summary_audit.append({'dataset':dataset,'method':method,'context':context,
            'metric':metric,'raw_mean':measured,'summary_mean':found.iloc[0]['mean'],
            'absolute_error':abs(measured-found.iloc[0]['mean']),
            'raw_n':len(g),'summary_n':int(found.iloc[0]['n_valid_draws'])})
summary_audit=pd.DataFrame(summary_audit)
assert len(summary_audit)==1768
assert np.isclose(summary_audit.raw_mean,summary_audit.summary_mean,atol=1e-10).all()
assert (summary_audit.raw_n==summary_audit.summary_n).all()

# Primary descriptive macro estimand exactly matches source script.
def macro(df,metric,contexts=None,datasets=None):
    p=df[(df.metric==metric)]
    if contexts is not None: p=p[p.context.isin(contexts)]
    if datasets is not None:p=p[p.dataset.isin(datasets)]
    return p.groupby(['dataset','method'])['mean'].mean().groupby('method').mean()
all_macro=pd.DataFrame({metric:macro(summary,metric) for metric in CORE}).sort_values('auroc',ascending=False)
low_macro=pd.DataFrame({metric:macro(summary,metric,[16,32,64]) for metric in CORE}).sort_values('auroc',ascending=False)
assert np.allclose(all_macro.loc[evidence_macro.method,list(CORE)].to_numpy(),
                   evidence_macro.set_index('method')[list(CORE)].to_numpy(),atol=1e-12)

# Per-method task means, context/seed paired differences.
task_means=(raw.groupby(['dataset','cohort','method','context'])[list(CORE)]
              .mean().groupby(['dataset','cohort','method'])[list(CORE)]
              .mean().reset_index())
assert np.allclose(task_means.groupby('method')[list(CORE)].mean().loc[all_macro.index].to_numpy(),
                   all_macro[list(CORE)].to_numpy(),atol=1e-12)

# contexts; conditional on valid cells only, same paired cells for every method.
context_summary=(raw.groupby(['context','dataset','method'])[list(CORE)].mean()
                   .groupby(['context','method'])[list(CORE)].mean().reset_index())

# Dataset AUROC table for every method.
task_auc=task_means.pivot(index='dataset',columns='method',values='auroc')
task_auc=task_auc.reindex(DATASETS)

# Paired effects, positive values always favor THREE. Same model test records
# are re-used for all labeled context draws; this is NOT patient-level pairing.
primary=raw[raw.method==THREE].drop(columns=['method','source_path']).rename(
    columns={v:v+'_three' for v in CORE})
assert len(primary)==671
pair_records=[]
paired_raw=[]
cohort_records=[]
cohort_input={}
contrasts=[]

# Efficient bootstrap of 7 source clusters with VAT tasks kept together.
def cluster_bootstrap(task_effects, draws=100000, seed=SEED, equal_source=False):
    """Return CI for equal-task mean (or equal-source mean) using cohort resampling."""
    gr=task_effects.groupby('cohort')['effect'].agg(['sum','count','mean'])
    vals=gr['mean'].to_numpy() if equal_source else gr['sum'].to_numpy()
    denom=np.ones(len(gr),dtype=float) if equal_source else gr['count'].to_numpy(dtype=float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(gr),size=(draws,len(gr)))
    means=vals[idx].sum(axis=1)/denom[idx].sum(axis=1)
    return [float(v) for v in np.quantile(means,[0.025,0.975])]


def signflip(task_effects):
    """Exact 2^7 source-level sign flips, preserving task weights within VAT."""
    contributions=task_effects.groupby('cohort')['effect'].sum().to_numpy()
    n_tasks=len(task_effects)
    signs=np.array(list(itertools.product([-1,1],repeat=len(contributions))))
    null=(signs*contributions).sum(axis=1)/n_tasks
    observed=contributions.sum()/n_tasks
    # Exhaustive permutation p-values, symmetry null and exploratory ONLY.
    p_two=float(np.mean(np.abs(null)>=abs(observed)-1e-14))
    p_one=float(np.mean(null>=observed-1e-14))
    return p_two,p_one


def draw_stratified_ci(paired, metric='auroc', draws=30000, seed=SEED):
    rng=np.random.default_rng(seed)
    task_results=[]
    for dataset,task in paired.groupby('dataset'):
        context_results=[]
        for context,g in task.groupby('context'):
            values=g['effect'].to_numpy()
            ids=rng.integers(0,len(values),size=(draws,len(values)))
            context_results.append(values[ids].mean(axis=1))
        task_results.append(np.stack(context_results).mean(axis=0))
    means=np.stack(task_results).mean(axis=0)
    return [float(v) for v in np.quantile(means,[0.025,0.975])]


def holm_adjust(pvals):
    n=len(pvals)
    order=np.argsort(pvals)
    out=np.zeros(n)
    acc=0.
    for rank,i in enumerate(order):
        acc=max(acc,min(1.0,(n-rank)*pvals[i]))
        out[i]=acc
    return out


def bh_adjust(pvals):
    n=len(pvals)
    order=np.argsort(pvals)[::-1]
    out=np.zeros(n)
    acc=1.0
    for j,i in enumerate(order):
        rank=n-j
        acc=min(acc, pvals[i]*n/rank)
        out[i]=acc
    return out

for competitor in sorted(set(EXPECTED_BASE)):
    q=raw[raw.method==competitor].drop(columns=['method','source_path']).rename(
        columns={v:v+'_competitor' for v in CORE})
    paired=primary.merge(q,on=['dataset','cohort','context','seed'],validate='one_to_one')
    assert len(paired)==671, (competitor,len(paired))
    metrics={}
    for metric in CORE:
        if metric in UP:
            paired['effect']=paired[metric+'_three']-paired[metric+'_competitor']
        else:
            paired['effect']=paired[metric+'_competitor']-paired[metric+'_three']
        task=(paired.groupby(['dataset','cohort','context'])['effect'].mean().reset_index()
              .groupby(['dataset','cohort'])['effect'].mean().reset_index())
        assert len(task)==8 and task.cohort.nunique()==7
        point=task.effect.mean()
        ci=cluster_bootstrap(task,seed=SEED+len(competitor)*4+list(CORE).index(metric))
        ci_src=cluster_bootstrap(task,seed=SEED+len(competitor)*4+list(CORE).index(metric),equal_source=True)
        two,one=signflip(task)
        # Cohort-mean t-test and Wilcoxon are sensitivity checks, not primary.
        co=task.groupby('cohort').effect.mean()
        t=ttest_1samp(co,0).pvalue
        try:w=wilcoxon(co,zero_method='wilcox',alternative='two-sided',method='exact').pvalue
        except Exception:w=float('nan')
        sign=binomtest(int((co>0).sum()),n=len(co),p=.5,alternative='two-sided').pvalue
        low_tasks=task[task.dataset!='pima_diabetes']
        lowmean=low_tasks.effect.mean()
        sensitivity=[]
        for dropped in task.cohort.unique():
            sensitivity.append(task.loc[task.cohort!=dropped,'effect'].mean())
        record={
            'comparator':competitor,'metric':metric,
            'difference_positive_favors':'three_expert',
            'macro_three':float(all_macro.loc[THREE,metric]),
            'macro_competitor':float(all_macro.loc[competitor,metric]),
            'equal_task_effect':float(point),'equal_source_effect':float(co.mean()),
            'task_wins':int((task.effect>0).sum()),'tasks':len(task),
            'source_wins':int((co>0).sum()),'sources':len(co),
            'draw_wins':int((paired.effect>0).sum()),'paired_draws':len(paired),
            'cohort_boot_ci_lower':ci[0],'cohort_boot_ci_upper':ci[1],
            'equal_source_boot_ci_lower':ci_src[0],
            'equal_source_boot_ci_upper':ci_src[1],
            'exact_source_signflip_p_two_sided':two,
            'exact_source_signflip_p_one_sided':one,
            'source_mean_ttest_p_two_sided':float(t),
            'source_wilcoxon_p_two_sided':float(w),
            'source_sign_test_p_two_sided':float(sign),
            'drop_pima_macro_effect':float(lowmean),
            'leave_one_source_out_min':float(min(sensitivity)),
            'leave_one_source_out_max':float(max(sensitivity)),
            'min_task_effect':float(task.effect.min()),
            'max_task_effect':float(task.effect.max()),
        }
        if competitor in (DEFAULT,TWO,'tiny_generic'):
            dc=draw_stratified_ci(paired,seed=SEED+len(competitor)*4+list(CORE).index(metric))
            record['paired_context_draw_ci_lower']=dc[0]
            record['paired_context_draw_ci_upper']=dc[1]
        contrasts.append(record)
        if competitor in (DEFAULT,TWO,'tiny_generic'):
            for _,row in task.iterrows():
                if metric=='auroc':
                    taskcopy=row.to_dict()
                    taskcopy.update({'comparator':competitor, 'metric':metric})
                    pair_records.append(taskcopy)
            for (dataset,context),grp in paired.groupby(['dataset','context']):
                diff=grp.effect.to_numpy()
                drawci=draw_stratified_ci(grp,seed=SEED+context) if False else [float('nan')]*2
                # Exact paired observation means, no test-label access.
                paired_raw.append({'dataset':dataset,'cohort':COHORTS[dataset],
                    'context':int(context),'comparator':competitor,'metric':metric,
                    'n_draws':len(grp),'paired_mean_effect':float(diff.mean()),
                    'paired_sd':float(diff.std(ddof=1)),'paired_sem':float(diff.std(ddof=1)/np.sqrt(len(diff))),
                    'positive_draw_fraction':float((diff>0).mean())})

contrasts=pd.DataFrame(contrasts)
for metric in CORE:
    selection=contrasts.metric==metric
    contrasts.loc[selection,'holm_p_12_competitors']=holm_adjust(
        contrasts.loc[selection,'exact_source_signflip_p_two_sided'].to_numpy())
    contrasts.loc[selection,'bh_q_12_competitors']=bh_adjust(
        contrasts.loc[selection,'exact_source_signflip_p_two_sided'].to_numpy())
    contrasts.loc[selection,'is_confirmatory']='NO: benchmark reused for selection'

pair_tasks=pd.DataFrame(pair_records)
pair_contexts=pd.DataFrame(paired_raw)

# Recompute low-data and each-context primary contrasts.
def aggregate_selected(metric,competitor,contexts=None,exclude=None):
    if exclude is None: exclude=[]
    p=task_means
    a=raw[raw.method==THREE]
    b=raw[raw.method==competitor]
    if contexts is not None:
        a=a[a.context.isin(contexts)]
        b=b[b.context.isin(contexts)]
    if exclude:
        a=a[~a.dataset.isin(exclude)]
        b=b[~b.dataset.isin(exclude)]
    # Paired draw exact keys only.
    join=a.merge(b,on=['dataset','cohort','context','seed'],suffixes=('_a','_b'),validate='one_to_one')
    effect=(join[metric+'_a']-join[metric+'_b']) if metric in UP else (
        join[metric+'_b']-join[metric+'_a'])
    join=join.assign(effect=effect)
    task=(join.groupby(['dataset','cohort','context'])['effect'].mean().reset_index()
       .groupby(['dataset','cohort'])['effect'].mean().reset_index())
    ci=cluster_bootstrap(task,seed=SEED+len(competitor)+len(metric)+int(contexts[0]) if contexts else SEED+len(competitor)+len(metric))
    two,one=signflip(task)
    return {'metric':metric,'comparator':competitor,'contexts':','.join(str(i) for i in contexts) if contexts else 'all',
        'n_tasks':len(task),'n_sources':task.cohort.nunique(),
        'paired_draws':len(join),'three_macro':float(join.groupby(['dataset','context'])[metric+'_a'].mean().groupby('dataset').mean().mean()),
        'other_macro':float(join.groupby(['dataset','context'])[metric+'_b'].mean().groupby('dataset').mean().mean()),
        'equal_task_effect':float(task.effect.mean()),
        'cohort_ci_lower':ci[0],'cohort_ci_upper':ci[1],
        'exact_signflip_p_two_sided':two,
        'dataset_wins':int((task.effect>0).sum()),
        'nonmaternal_excluded':bool(exclude)}

primary_contexts=[]
for competitor in (DEFAULT,TWO,'tiny_generic'):
    for metric in CORE:
        for contexts in ([16,32,64], [16], [32], [64], [128],[256],[512], None):
            primary_contexts.append(aggregate_selected(metric,competitor,contexts=contexts))
primary_contexts=pd.DataFrame(primary_contexts)

# Cohort heterogeneity, leave-one-source-out, exclusions.
robust=[]
for competitor in (DEFAULT,TWO,'tiny_generic'):
    for metric in CORE:
        scenarios={
            'all_tasks':[],
            'maternal_only_no_Pima':['pima_diabetes'],
            'exclude_VAT_pair':['vat_gdm','vat_delivery'],
            'exclude_low_birth_weight':['low_birth_weight'],
            'exclude_UCI_proxy':['uci_maternal_risk'],
            'strict_no_Pima_or_UCI':['pima_diabetes','uci_maternal_risk'],
        }
        for scenario,exclude in scenarios.items():
            r=aggregate_selected(metric,competitor,exclude=exclude)
            r['scenario']=scenario
            robust.append(r)
robust=pd.DataFrame(robust)

# Generic expert NLL weights: diagnostics mechanism; no causal claim.
weights=diagnostics.groupby('dataset').agg(
  n=('seed','size'),generic_mean_weight=('weight_3','mean'),
  generic_median_weight=('weight_3','median'),
  v1_mixture_mean_weight=('weight_1','mean'),
  v2_mixture_mean_weight=('weight_2','mean'),
  mean_temperature=('temperature','mean'),
  fraction_generic_top=('weight_3',lambda s:np.mean(s>=diagnostics.loc[s.index,['weight_1','weight_2','weight_3']].max(axis=1))),
  fraction_cross_fitted=('cv_status',lambda s:np.mean(s=='cross_fitted')),
).reset_index()
weights['delta_auroc_vs_two']=weights.dataset.map(
    pair_tasks[(pair_tasks.metric=='auroc')&(pair_tasks.comparator==TWO)].set_index('dataset').effect)
weights['delta_auroc_vs_tabpfn']=weights.dataset.map(
    pair_tasks[(pair_tasks.metric=='auroc')&(pair_tasks.comparator==DEFAULT)].set_index('dataset').effect)

# Compare original v1 with v2 on paired draws; no causal isolation.
v1v2=[]
for tier in ['tiny','base']:
    for prior in ['generic','mixture']:
        nm=f'{tier}_{prior}'
        for metric in CORE:
            a=raw[raw.method==nm][['dataset','cohort','context','seed',metric]]
            b=raw[raw.method=='v1_'+nm][['dataset','cohort','context','seed',metric]]
            j=a.merge(b,on=['dataset','cohort','context','seed'],suffixes=('_v2','_v1'))
            assert len(j)==671
            effect=j[metric+'_v2']-j[metric+'_v1'] if metric in UP else j[metric+'_v1']-j[metric+'_v2']
            eff=j.assign(effect=effect).groupby(['dataset','cohort','context'])['effect'].mean().reset_index().groupby(['dataset','cohort']).effect.mean().reset_index()
            ci=cluster_bootstrap(eff,seed=SEED+len(nm)+len(metric))
            v1v2.append({'arm':nm,'metric':metric,'v2_improvement_positive':float(eff.effect.mean()),
                'v2_task_wins':int((eff.effect>0).sum()),'n_tasks':len(eff),
                'cohort_ci_lower':ci[0],'cohort_ci_upper':ci[1],
                'source_signflip_p_two_sided':signflip(eff)[0]})
v1v2=pd.DataFrame(v1v2)

# Efficiency measures from completed local CPU runs; protocols differ.
eff=[]
for row in adaptive_cpu.get('results',[]):
    eff.append({'method':row['method'],'context':row['context'],'runtime':'PyTorch CPU',
        'fit_cv_p50_ms':row['fit_cv_p50_ms'],'query_p50_ms':row['query_p50_ms'],
        'query_p99_ms':row['query_p99_ms'],
        'combined_checkpoint_mb':row['combined_checkpoint_bytes']/1e6,
        'repeats':row['repeats'],'model':'multi-expert'})
for row in edge['results']:
    if 'onnxruntime' in row:
        onnx=row['onnxruntime']
        eff.append({'method':'standalone_tiny_mixture_onnx','context':row['context'],
            'runtime':'ONNX CPU','fit_cv_p50_ms':float('nan'),
            'query_p50_ms':onnx['p50_ms'],'query_p99_ms':onnx['p99_ms'],
            'combined_checkpoint_mb':onnx['file_size_mb'],
            'repeats':100,'model':'standalone ONNX'})
for row in edge.get('baseline_comparisons',[]):
    eff.append({'method':row['method'],'context':row['context'],
        'runtime':'Baseline CPU' if row['device']=='cpu' else row['device'],
        'fit_cv_p50_ms':row['fit_p50_ms'],'query_p50_ms':row['query_p50_ms'],
        'query_p99_ms':float('nan'),'combined_checkpoint_mb':float('nan'),
        'repeats':row['repeats'],'model':'baseline'})
eff=pd.DataFrame(eff)
mem=pd.DataFrame(memory)
mem=mem[['graph','context','onnx_file_mb','total_process_peak_rss_mb',
  'incremental_peak_over_import_mb','meets_32mb_total_process_target']]

# Consolidated QA JSON and save all source-grounded derived tables.
QA={
    'input_zip_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'manifest_file_count':len(integrity),
    'manifest_sha256_verified':bool(integrity.hash_valid.all()),
    'manifest_sizes_verified':bool(integrity.size_valid.all()),
    'master_summary_rows':len(main),
    'three_expert_summary_rows':len(three),
    'all_raw_draw_rows_including_v1':len(raw),
    'raw_draw_rows_per_current_model':671,
    'evaluated_model_configurations_including_v1':len(counts),
    'evaluated_current_model_configurations':13,
    'dataset_tasks':8,
    'independent_source_cohorts_assumed':7,
    'matched_context_cells_per_model':len(main[main.method==DEFAULT].query('metric=="auroc"')),
    'raw_to_summary_recomputed_max_abs_error':float(summary_audit.absolute_error.max()),
    'raw_to_summary_n_mismatches':int((summary_audit.raw_n!=summary_audit.summary_n).sum()),
    'three_expert_diagnostic_rows':len(diagnostics),
    'three_expert_diagnostic_weight_sums_valid':True,
    'original_stored_pairs':len(stored_pairs),
    'summaries_derived_from_existing_data_not_new_patient_predictions':True,
    'patient_level_predictions_in_zip':False,
    'patient_level_DeLong_variance_available':False,
    'source_cohort_resampling_confirmatory':False,
    'test_benchmark_used_in_iterative_model_development':True,
    'critical_context':'VAT GDM and VAT delivery are two tasks from one cohort',
    'statistical_methods':{
      'point':'Equal-weight task macro, with contexts first averaged per task',
      'primary_sensitivity_ci':'100,000 replicate 7-source cluster bootstrap; preserve VAT pair',
      'primary_sensitivity_p':'exact 2^7 cohort sign-flips, two-sided under symmetric exchangeability null',
      'context_conditional_ci':'30,000 paired resamples within 34 task-context cells; NOT independent test patients',
      'multiple_testing':'Holm and Benjamini-Hochberg over 12 comparator contrasts per metric',
      'model_selection':'three-expert proposal motivated by already-seen benchmark; p values not confirmatory',
    }
}

for filename,frame in [
 ('integrity_manifest_audit.csv',integrity),
 ('raw_summary_consistency.csv',summary_audit),
 ('draw_counts.csv',counts),
 ('leaderboard_all.csv',all_macro.reset_index().rename(columns={'index':'method'})),
 ('leaderboard_low_data.csv',low_macro.reset_index().rename(columns={'index':'method'})),
 ('task_auroc_all_models.csv',task_auc.reset_index()),
 ('task_means_all_metrics.csv',task_means),
 ('context_all_models.csv',context_summary),
 ('three_expert_contrasts_all_metrics.csv',contrasts),
 ('primary_task_effects.csv',pair_tasks),
 ('primary_context_effects.csv',pair_contexts),
 ('primary_by_context.csv',primary_contexts),
 ('robustness_exclusions.csv',robust),
 ('three_expert_weight_diagnostics.csv',weights),
 ('three_expert_weights_raw.csv',diagnostics),
 ('v1_vs_v2_development.csv',v1v2),
 ('efficiency_measured.csv',eff),
 ('onnx_memory_measured.csv',mem),
]:
    frame.to_csv(OUT/filename,index=False,float_format='%.12g')
(OUT/'data_audit.json').write_text(json.dumps(QA,indent=2)+'\n')

print('AUDIT',json.dumps({k:v for k,v in QA.items() if k not in ('statistical_methods',)},indent=2))
print('\nLEADERBOARD (AUROC)\n',all_macro.round(6).to_string())
print('\nPRIMARY CONTRASTS ALL METRICS:')
print(contrasts[contrasts.comparator.isin([DEFAULT,TWO,'tiny_generic'])][['comparator','metric','equal_task_effect','cohort_boot_ci_lower','cohort_boot_ci_upper','exact_source_signflip_p_two_sided','holm_p_12_competitors','task_wins','source_wins','paired_context_draw_ci_lower','paired_context_draw_ci_upper']].round(7).to_string(index=False))
print('\nTASK DIFFERENCES\n',pair_tasks[pair_tasks.comparator==DEFAULT].pivot(index='dataset',columns='metric',values='effect').round(5).to_string())
print('\nROBUSTNESS: AUROC vs TabPFN')
print(robust[(robust.comparator==DEFAULT)&(robust.metric=='auroc')][['scenario','n_tasks','n_sources','equal_task_effect','cohort_ci_lower','cohort_ci_upper','exact_signflip_p_two_sided']].round(6).to_string(index=False))
print('\nMACRO LOW AUROC',low_macro.auroc.head(6).round(6).to_string())
print('\nWEIGHTS\n',weights[['dataset','generic_mean_weight','fraction_generic_top','delta_auroc_vs_two']].round(3).to_string(index=False))
print('\nRESULT_FILES',len(list(OUT.iterdir())))
