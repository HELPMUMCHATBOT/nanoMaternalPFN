#!/usr/bin/env python3
"""E10 measured dynamic INT8 ONNX comparison on actual held-out patients.

Fixed-context graph must match dataset shape. Measures both FP32 and INT8
on exactly the same query rows with the same CPU execution provider.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .common import load_frozen,write_json,sha256,metrics


def run(model,output,manifest,context=64,repeats=100):
 import onnxruntime as ort
 from onnxruntime.quantization import QuantType,quantize_dynamic
 path=Path(model);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if not path.is_file():raise FileNotFoundError(path)
 data=load_frozen(manifest)
 if str(context) not in data.contexts:raise ValueError('Frozen context unavailable')
 draw=np.asarray(data.contexts[str(context)][0],dtype=int)
 if len(np.unique(data.y[draw]))<2:raise ValueError('First frozen context has too few classes')
 classes=np.unique(data.y)
 y_ctx=np.searchsorted(classes,data.y[draw]).astype('int64')[None]
 quantize_dynamic(str(path),str(output),weight_type=QuantType.QInt8,
                  op_types_to_quantize=['MatMul','Gemm'])
 fp=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
 q=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
 def probs(session,query):
   inputs={'x':np.concatenate([data.x[draw],query]).astype('float32')[None],
           'y_train':y_ctx}
   if set(inputs)!={x.name for x in session.get_inputs()}:
     raise ValueError('ONNX graph input names unexpected')
   logits=np.asarray(session.run(None,inputs)[0],dtype=float)
   logits=logits[..., :len(classes)]
   logits=logits-logits.max(axis=-1,keepdims=True)
   ex=np.exp(logits);p=ex/ex.sum(axis=-1,keepdims=True)
   if p.shape!=(1,1,len(classes)) or not np.isfinite(p).all():
     raise ValueError('ONNX model must provide one-query normalized predictions')
   return p[0,0]
 test_rows=data.x[data.test]
 fp_prob=np.array([probs(fp,row[None]) for row in test_rows])
 q_prob=np.array([probs(q,row[None]) for row in test_rows])
 def timed(session):
   for _ in range(10):probs(session,test_rows[:1])
   times=[]
   for _ in range(repeats):
     start=time.perf_counter();probs(session,test_rows[:1]);times.append((time.perf_counter()-start)*1000)
   return {'p50_ms':float(np.percentile(times,50)),'p99_ms':float(np.percentile(times,99))}
 result={'context':context,'dataset':data.name,'source_data_sha256':data.sha256,
        'fp32_graph_sha256':sha256(path),'int8_graph_sha256':sha256(output),
        'file_size_fp32_mb':path.stat().st_size/1e6,'file_size_int8_mb':output.stat().st_size/1e6,
        'n_real_heldout_patients':len(test_rows),
        'max_abs_prediction_difference':float(np.max(np.abs(fp_prob-q_prob))),
        'fp32_metrics':metrics(data.y[data.test],fp_prob),
        'int8_metrics':metrics(data.y[data.test],q_prob),
        'fp32_cpu':timed(fp),'int8_cpu':timed(q),
        'scope':'One frozen context draw and its real patient test pool; repeat every draw/task for full evidence.',
        'quantized_model_is_a_different_predictor':True,
        'energy_power_or_android_not_measured':True}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output-onnx',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--context',type=int,default=64)
 p.add_argument('--repeats',type=int,default=100);p.add_argument('--report',type=Path,required=True)
 a=p.parse_args()
 if a.repeats<10:raise ValueError('At least 10 timed queries required')
 result=run(a.onnx,a.output_onnx,a.manifest,a.context,a.repeats)
 write_json(a.report,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
