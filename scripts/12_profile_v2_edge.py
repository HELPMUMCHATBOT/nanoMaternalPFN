#!/usr/bin/env python3
"""Profile a trained nanoMaternalPFN v2 checkpoint with one held-out-free query.

This profiles native PyTorch CPU and optionally exports/evaluates ONNX CPU
through onnxruntime. It is NOT an Android, INT8, energy, or deployment benchmark.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig
from nanomaternalpfn.edgebench.export_onnx import export
from nanomaternalpfn.edgebench.profiler import profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path,
                   default=Path('artifacts/results/aws_v2_pretrain/tiny_mixture.best.inference.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/results/edge_v2'))
    p.add_argument('--contexts', nargs='+', type=int, default=[16, 64])
    p.add_argument('--features', type=int, default=6)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--onnx', action='store_true', help='Export and time optional ONNX Runtime CPU')
    args = p.parse_args()
    if args.repeats < 1 or min(args.contexts) < 1:
        p.error('repeats and contexts must be positive')
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = NanoMaternalPFN(ModelConfig(**state['config']))
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {'checkpoint':str(args.checkpoint),
              'model_variant':model.cfg.architecture,
              'runtime':'single-patient CPU',
              'not_measured':['Android','energy','INT8 RAM'],
              'results':[]}
    for n in args.contexts:
        torch.manual_seed(23)
        X = torch.randn(1, n+1, args.features)
        y = torch.arange(n, dtype=torch.long)[None] % 2
        observed = profile(model, X, y, repeats=args.repeats)
        item = {'context':n,'features':args.features,'query_rows':1,
                'pytorch':observed}
        if args.onnx:
            import onnxruntime as ort
            path = export(model, args.output_dir / f'v2_ctx{n}.onnx',
                          n_context=n,n_query=1,n_features=args.features)
            session = ort.InferenceSession(str(path),
                       providers=['CPUExecutionProvider'])
            values = {'x':X.numpy(), 'y_train':y.numpy()}
            for _ in range(5):
                session.run(None, values)
            times=[]
            for _ in range(args.repeats):
                start=time.perf_counter()
                session.run(None, values)
                times.append((time.perf_counter()-start)*1000)
            item['onnxruntime']={
                'p50_ms':float(statistics.median(times)),
                'p99_ms':float(np.percentile(times,99)),
                'file_size_mb':path.stat().st_size/1e6,
                'execution_provider':'CPUExecutionProvider'}
        output['results'].append(item)
        print(json.dumps(item,indent=2),flush=True)
    (args.output_dir/'profile.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
