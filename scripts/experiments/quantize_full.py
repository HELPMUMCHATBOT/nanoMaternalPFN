#!/usr/bin/env python3
"""E10: full real-data ONNX FP32 versus INT8 accuracy/latency campaign.

All eligible frozen context draws and held-out real records are evaluated.
No synthetic patient observations, no one-draw 'full study' substitution.
Shape-specific ONNX graphs are exported from the *actual* supplied weights.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .common import load_frozen,metrics,jsonl_append,write_json,sha256


def predict_pool(session,data,draw):
    """Run fixed-shape ONNX inference for every real held-out query."""
    classes=np.unique(data.y)
    local=np.unique(data.y[draw])
    context_y=np.searchsorted(local,data.y[draw]).astype('int64')[None]
    context=data.x[draw]
    inputs={p.name for p in session.get_inputs()}
    if inputs!={'x','y_train'}:raise RuntimeError('Unexpected ONNX graph inputs '+str(inputs))
    result=np.zeros((len(data.test),len(classes)),dtype=float)
    for j,idx in enumerate(data.test):
        x=np.concatenate((context,data.x[idx:idx+1]),axis=0).astype('float32')[None]
        logits=np.asarray(session.run(None,{'x':x,'y_train':context_y})[0],dtype=float)
        if logits.ndim!=3 or logits.shape[:2]!=(1,1) or logits.shape[2]<len(local):
            raise RuntimeError('Exported graph is not the expected one-query classifier')
        z=logits[0,0,:len(local)];z-=np.max(z)
        ex=np.exp(z);p=ex/ex.sum()
        for k,label in enumerate(local):
            result[j,np.where(classes==label)[0][0]]=p[k]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1),1,atol=1e-5):
        raise RuntimeError('Nonfinite/unnormalized ONNX patient probabilities')
    return result


def timed_one(session,data,draw,n=100):
    local=np.unique(data.y[draw]);labels=np.searchsorted(local,data.y[draw]).astype('int64')[None]
    query=data.x[data.test[0]:data.test[0]+1]
    x=np.concatenate((data.x[draw],query)).astype('float32')[None]
    inp={'x':x,'y_train':labels}
    for _ in range(10):session.run(None,inp)
    vals=[]
    for _ in range(n):
        t=time.perf_counter();session.run(None,inp);vals.append((time.perf_counter()-t)*1000)
    return {'n_repeats':n,'p50_ms':float(np.percentile(vals,50)),
            'p95_ms':float(np.percentile(vals,95)),
            'p99_ms':float(np.percentile(vals,99))}


def run(manifests,ckpt,out,repeats=100):
    import torch
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType,quantize_dynamic
    from nanomaternalpfn.model import NanoMaternalPFN,ModelConfig
    from nanomaternalpfn.edgebench.export_onnx import export
    ckpt=Path(ckpt);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if 'smoke' in ckpt.name.lower():raise ValueError('No smoke checkpoint may enter paper experiment E10')
    source=torch.load(ckpt,map_location='cpu',weights_only=True)
    model=NanoMaternalPFN(ModelConfig(**source['config']))
    model.load_state_dict(source['state_dict'],strict=True)
    model.eval().cpu()
    checksum=sha256(ckpt)
    journal=out/'real_quantized_draws.jsonl'
    done={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record=json.loads(line);done[(record['dataset'],record['context'],record['seed'])]=record
            if record['checkpoint_sha256']!=checksum:
                raise ValueError('Cannot resume after a different checkpoint was selected')
    datasets={};expected=0
    for path in manifests:
        data=load_frozen(path);datasets[data.name]=data.sha256
        for size_str,draws in sorted(data.contexts.items(),key=lambda x:int(x[0])):
            size=int(size_str)
            valid=[(i,np.asarray(v,dtype=int)) for i,v in enumerate(draws) if len(np.unique(data.y[v]))>=2]
            expected+=len(valid)
            if not valid:continue
            target=out/'graphs'/data.name
            target.mkdir(parents=True,exist_ok=True)
            fp=target/f'ctx{size}.fp32.onnx';q=target/f'ctx{size}.int8.onnx'
            # Fixed shape must be specific to the actual clinical feature count.
            if not fp.is_file():
                export(model,fp,n_context=size,n_query=1,n_features=data.x.shape[1])
            if not q.is_file():
                quantize_dynamic(str(fp),str(q),weight_type=QuantType.QInt8,
                                 op_types_to_quantize=['MatMul','Gemm'])
            fp_session=ort.InferenceSession(str(fp),providers=['CPUExecutionProvider'])
            q_session=ort.InferenceSession(str(q),providers=['CPUExecutionProvider'])
            for seed,draw in valid:
                key=(data.name,size,seed)
                if key in done:continue
                original=predict_pool(fp_session,data,draw)
                compressed=predict_pool(q_session,data,draw)
                observed={'dataset':data.name,'context':size,'seed':seed,
                    'n_real_patients':len(data.test),'n_context':len(draw),
                    'source_npz_sha256':data.sha256,'checkpoint_sha256':checksum,
                    'fp32_graph_sha256':sha256(fp),'int8_graph_sha256':sha256(q),
                    'fp32_graph_size_mb':fp.stat().st_size/1e6,
                    'int8_graph_size_mb':q.stat().st_size/1e6,
                    'fp32':metrics(data.y[data.test],original,classes=np.unique(data.y)),
                    'int8':metrics(data.y[data.test],compressed,classes=np.unique(data.y)),
                    'max_abs_probability_difference':float(np.max(np.abs(original-compressed))),
                    'fp32_timing':timed_one(fp_session,data,draw,repeats),
                    'int8_timing':timed_one(q_session,data,draw,repeats),
                    'actual_heldout_real_source':True,'android_energy_measured':False}
                jsonl_append(journal,observed);done[key]=observed
                print('MEASURED E10',data.name,'context',size,'draw',seed,'FP32',observed['fp32']['auroc'],'INT8',observed['int8']['auroc'],flush=True)
    write_json(out/'VERIFICATION.json',{'expected_real_draws':expected,
            'actually_measured_real_draws':len(done),'all_real_draws_measured':len(done)==expected,
            'source_files_sha256':datasets,'real_checkpoint_sha256':checksum,
            'no_synthetic_clinical_test_rows':True,'on_device_claim':False})
    if len(done)!=expected:raise RuntimeError('Not all eligible frozen context draws measured')
    print('E10 finished with',len(done),'measured real context draws')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args()
    if a.repeats<10:raise ValueError('Not a real timing benchmark: expected >=10 repetitions')
    run(a.manifests,a.checkpoint,a.out,a.repeats)

if __name__=='__main__':main()
