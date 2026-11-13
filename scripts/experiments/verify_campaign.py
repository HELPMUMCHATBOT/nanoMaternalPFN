#!/usr/bin/env python3
"""Evidence gate for real E1–E12 outputs; never equates code tests with research.

A COMPLETE.json is written only after all prespecified frozen draw/method/
condition keys are present (or an explicitly justified skipped draw) and
source/checkpoint hashes are cross-checked against actual local bytes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import load_frozen,sha256,write_json

PREDICTION_STAGES=('E1','E2','E3','E5','E6','E7','E8','E11')


def jsonlines(path):
    if not path.is_file():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_prediction_stage(path,code):
    cfgfile=path/'RUN_CONFIG.json';journal=path/'measured_draws.jsonl'
    if not cfgfile.is_file():return {'status':'not_executed'}
    cfg=json.loads(cfgfile.read_text())
    if cfg.get('experiment')!=code or cfg.get('max_draws') is not None:
        raise RuntimeError(code+': wrong experiment or abbreviated pilot draw count')
    methods=cfg['methods'];conditions=cfg['conditions']
    if not methods:raise RuntimeError(code+': no evaluated methods')
    expected=set();sources={};n_test=0
    for manifest in cfg['manifests']:
        data=load_frozen(manifest,external=code=='E1')
        sources[data.name]=data.sha256;n_test+=len(data.test)
        if cfg['manifests'][manifest]!=data.sha256:
            raise RuntimeError(code+': source changed since actual execution: '+manifest)
        for ctx,draws in data.contexts.items():
            if cfg.get('contexts') and int(ctx) not in cfg['contexts']:continue
            for i,draw in enumerate(draws):
                if len(set(data.y[draw].tolist()))<2:continue
                for method in methods:
                    for cond in conditions:
                        expected.add((data.name,method,int(ctx),i,cond))
    seen={}
    for row in jsonlines(journal):
        key=(row['dataset'],row['method'],int(row['context']),int(row['draw_seed']),row['condition'])
        if key in seen:raise RuntimeError(code+': duplicate real clinical draw metric')
        if row.get('diagnostic_draw_cap'):raise RuntimeError(code+': diagnostic draw cap in results')
        if key not in expected:raise RuntimeError(code+': unexpected dataset/model/draw/condition: '+str(key))
        if row['provenance']['dataset_npz_sha256']!=sources[row['dataset']]:
            raise RuntimeError(code+': inconsistent clinical source checksum')
        for checkpoint,digest in row['provenance'].get('checkpoint_sha256',{}).items():
            if not Path(checkpoint).is_file() or sha256(checkpoint)!=digest:
                raise RuntimeError(code+': pretrained checkpoint missing or changed: '+checkpoint)
        seen[key]=row
    skipped={}
    for row in jsonlines(path/'skipped.jsonl'):
        key=(row['dataset'],row['method'],int(row['context']),int(row['draw_seed']),row['condition'])
        if key in skipped or key in seen or key not in expected:
            raise RuntimeError(code+': invalid skip record or duplicated draw')
        allowed=(code=='E8' and row['condition'].startswith('imbalance_')
                 and row['reason']=='too_few_minority_context_rows') or (
                 code=='E3' and row['condition']=='clean'
                 and row['reason']=='insufficient_context_labels_for_inner_cv')
        if not allowed:raise RuntimeError(code+': unexpected skipped clinical prediction')
        skipped[key]=row
    unresolved=expected-set(seen)-set(skipped)
    if unresolved:raise RuntimeError(f'{code}: missing {len(unresolved)} required draw/condition measurements')
    if code=='E5':
        status=path/'patient_analysis'/'ANALYSIS_STATUS.json'
        if not status.is_file():raise RuntimeError('E5 lacks full patient-level binary analyses')
        analysis=json.loads(status.read_text())
        if analysis.get('missing_pairs') or analysis.get('total_measured_binary_comparisons',0)<1:
            raise RuntimeError('E5 patient-level analyses incomplete')
    if code in ('E2','E7','E11'):
        mp=path.parent/'training'/f'{code}_checkpoint_map.json'
        if not mp.is_file():raise RuntimeError('Missing full trained checkpoint map for '+code)
        for label,cp in json.loads(mp.read_text()).items():
            if label not in methods or not Path(cp).is_file():raise RuntimeError('Missing trained experimental checkpoint '+label)
    result={'measured':True,'verified':True,'experiment':code,
            'source_sha256':sources,'n_real_clinical_draw_conditions':len(seen),
            'n_scientifically_justified_skips':len(skipped),
            'n_frozen_test_records_across_tasks':n_test,'n_models':len(methods),
            'full_draws_used':True,'source':'actual frozen real clinical cohort NPZ + pretrained checkpoints',
            'restrictions':'Retrospective results are not fresh independent validation' if code!='E1' else 'Independent source attestation provided'}
    write_json(path/'COMPLETE.json',result)
    return {'status':'measured_verified','n_draw_conditions':len(seen)}


def verify_profile(path,manifests):
    if not path.is_dir():return {'status':'not_executed'}
    expected=[]
    for manifest in manifests:
        data=load_frozen(manifest)
        for ctx,draws in data.contexts.items():
            if any(len(set(data.y[draw].tolist()))>=2 for draw in draws):
                expected.append((data.name,int(ctx),data.sha256))
    for task,context,digest in expected:
        path_json=path/f'{task}_c{context}'/'matched_cpu.json'
        if not path_json.is_file():raise RuntimeError('E4 missing matched profile '+str(path_json))
        doc=json.loads(path_json.read_text())
        if not doc.get('matched_protocol') or not doc.get('method_results'):
            raise RuntimeError('E4 needs isolated same-protocol matched methods')
        for r in doc['method_results']:
            if r.get('dataset_sha256')!=digest or int(r.get('context',-1))!=context:
                raise RuntimeError('E4 benchmark provenance/context changed')
            if r.get('repeats',0)<10 or r.get('query_p99_ms') is None:
                raise RuntimeError('E4 CPU benchmark does not include measured p99')
    sources={m.stem:load_frozen(m).sha256 for m in manifests}
    write_json(path/'COMPLETE.json',{'measured':True,'verified':True,'experiment':'E4',
                   'source_sha256':sources,'matched_task_context_profiles':len(expected),
                   'target_hardware':'native CPU reported per measured method',
                   'not_claimed':'Android app RAM/energy or unmeasured process high water'})
    return {'status':'measured_verified','profiles':len(expected)}


def verify_quantization(path):
    ver=path/'VERIFICATION.json'
    if not ver.is_file():return {'status':'not_executed'}
    doc=json.loads(ver.read_text())
    if not doc.get('all_real_draws_measured') or doc.get('actually_measured_real_draws',0)<1:
        raise RuntimeError('E10 full actual frozen draw coverage incomplete')
    journal=jsonlines(path/'real_quantized_draws.jsonl')
    if len(journal)!=doc['actually_measured_real_draws']:
        raise RuntimeError('E10 measured summary and real draw journal disagree')
    sources=doc['source_files_sha256']
    for row in journal:
        if row['source_npz_sha256']!=sources[row['dataset']] or not row['actual_heldout_real_source']:
            raise RuntimeError('E10 source provenance not actual frozen clinical data')
    write_json(path/'COMPLETE.json',{'measured':True,'verified':True,'experiment':'E10',
                   'source_sha256':sources,'real_quantized_draws':len(journal),
                   'not_claimed':'Android or power or patient deployment readiness'})
    return {'status':'measured_verified','draws':len(journal)}


def verify_all(out,manifest_dir):
    out=Path(out);manifests=[Path(manifest_dir)/f'{t}.json' for t in (
      'uci_maternal_risk','vat_gdm','vat_delivery','low_birth_weight',
      'cardiotocography','pima_diabetes','tpehg_preterm','ctu_cord_acidemia')]
    statuses={}
    for code in PREDICTION_STAGES:
        statuses[code]=verify_prediction_stage(out/code,code)
    statuses['E4']=verify_profile(out/'E4',manifests)
    statuses['E10']=verify_quantization(out/'E10')
    statuses['E9']={'status':'requires_real_native_target_hardware'}
    statuses['E12']={'status':'requires_approved_recruited_prospective_clinical_study'}
    summary={'observed_evidence_status':statuses,
             'code_does_not_equal_experiment_completion':True}
    write_json(out/'VERIFIED_EVIDENCE_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/results/e1_e12'))
    p.add_argument('--manifest-dir',type=Path,default=Path('artifacts/manifests'))
    args=p.parse_args();verify_all(args.output,args.manifest_dir)

if __name__=='__main__':main()
