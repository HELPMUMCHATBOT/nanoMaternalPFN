#!/usr/bin/env python3
"""E1--E12 evidence gate: classify actual measured deliverables, not intentions."""
import argparse
import json
from pathlib import Path
from .common import discover_missing_inputs,write_json,HISTORICAL_TASKS

EVIDENCE={
 'E1':('external_dataset_manifest','new independent clinical cohort, frozen model, complete test predictions'),
 'E2':('architecture_ablation','controlled trained checkpoints + paired real-world evaluation'),
 'E3':('tuned_baselines','context-only CV on frozen clinical draw/holdout'),
 'E4':('matched_cpu','same protocol, hardware and models, resource measurements'),
 'E5':('patient_level','patient-level probabilities and outcomes plus calibration/intervals'),
 'E6':('expert_mechanisms','uniform vs context CV vs fixed expert on paired real draws'),
 'E7':('prior_composition','0/25/50/70/100% controlled pretraining, independent test comparisons'),
 'E8':('robustness','frozen-source perturbation tests plus distinct hospital cohorts for real shift'),
 'E9':('arm_android','native hardware records, total RSS, latency, optional calibrated power device'),
 'E10':('onnx_int8','real-dataset parity, accuracy, latency/memory after actual quantization'),
 'E11':('long_training','5000/10000/20000-step validation history and frozen transfer'),
 'E12':('prospective_clinical','ethics approval, locked protocol and prospectively observed patient outcomes'),
}


def gate(manifest_dir,checkpoint_dirs,evidence_dir):
    missing=discover_missing_inputs(manifest_dir,checkpoint_dirs)
    status={}
    for code,(key,meaning) in EVIDENCE.items():
        evidence=Path(evidence_dir)/code/'COMPLETE.json'
        if evidence.exists():
            obj=json.loads(evidence.read_text())
            ok=bool(obj.get('measured',False) and obj.get('source_sha256') and obj.get('verified',False))
            label='measured_verified' if ok else 'evidence_insufficient'
        else:label='not_measured'
        status[code]={'status':label,'required':meaning,
              'cannot_mark_complete_from_runners_alone':True,'evidence_file':str(evidence)}
    return {'experiments':status,'local_missing_inputs':missing,
            'source':'evidence_gate_reproducible_protocol_not_paper_results'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir',type=Path,default=Path('artifacts/manifests'))
    p.add_argument('--checkpoint',type=Path,nargs='*',default=[])
    p.add_argument('--evidence-dir',type=Path,default=Path('artifacts/results/e1_e12'))
    p.add_argument('--output',type=Path,default=Path('artifacts/results/e1_e12/status.json'))
    a=p.parse_args();r=gate(a.manifest_dir,a.checkpoint,a.evidence_dir)
    write_json(a.output,r)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
