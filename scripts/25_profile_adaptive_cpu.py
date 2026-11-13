#!/usr/bin/env python3
"""Actually measure local CPU cost of two- and three-expert adaptive predictors.

Reports warmed fit/context-CV and single-query inference separately.
Does NOT pretend standalone Tiny ONNX timings apply to adaptive ensembles.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
from nanomaternalpfn.multi_expert_classifier import ContextOnlyMultiExpertPFN


def timed(model, X, y, query, repeats):
    # Warm-up includes all context-dependent fitting and cross-validation.
    model.fit(X,y)
    model.predict_proba(query)
    fit_ms = []
    query_ms = []
    for _ in range(repeats):
        start=time.perf_counter()
        model.fit(X,y)
        fit_ms.append(1000*(time.perf_counter()-start))
        start=time.perf_counter()
        model.predict_proba(query)
        query_ms.append(1000*(time.perf_counter()-start))
    def stats(seq):
        return {"p50_ms":float(statistics.median(seq)),
                "p99_ms":float(np.percentile(seq,99))}
    return stats(fit_ms),stats(query_ms)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-dir",type=Path,
                   default=Path("artifacts/results/aws_full_pretrain"))
    p.add_argument("--v2-dir",type=Path,
                   default=Path("artifacts/results/aws_v2_pretrain"))
    p.add_argument("--contexts",nargs="+",type=int,default=[16,64,128])
    p.add_argument("--features",type=int,default=6)
    p.add_argument("--repeats",type=int,default=7)
    p.add_argument("--output",type=Path,
                   default=Path("artifacts/results/adaptive_cpu_profile.json"))
    args=p.parse_args()
    if args.repeats<2 or args.features<2 or min(args.contexts)<2:
        p.error("Need >=2 repeats, features and context rows")
    two = [args.v1_dir/"tiny_mixture.best.inference.pt",
           args.v2_dir/"tiny_mixture.best.inference.pt"]
    three = two + [args.v2_dir/"tiny_generic.best.inference.pt"]
    for ckpt in three:
        if not ckpt.is_file():
            raise FileNotFoundError(f"Actual pretrained checkpoint missing: {ckpt}")
    groups=[("adaptive_tiny",two,AdaptiveNanoMaternalPFNClassifier),
            ("adaptive_tiny_3expert",three,ContextOnlyMultiExpertPFN)]
    rows=[]
    for context in args.contexts:
        rng=np.random.default_rng(20261113+context)
        X=rng.standard_normal((context+1,args.features)).astype(np.float32)
        X[0,0]=np.nan
        y=np.arange(context,dtype=np.int64) % 2
        query=X[-1:]
        for name,checkpoints,model_class in groups:
            model=model_class(checkpoints,device="cpu",query_batch_size=8,
                              cv_folds=4,calibrate=True)
            fit, predict=timed(model,X[:context],y,query,args.repeats)
            result={
                "method":name,"context":context,"features":args.features,
                "device":"cpu","single_query":True,
                "input":"synthetic timing only",
                "n_checkpoints":len(checkpoints),
                "combined_checkpoint_bytes":sum(c.stat().st_size for c in checkpoints),
                "fit_cv_p50_ms":fit["p50_ms"],"fit_cv_p99_ms":fit["p99_ms"],
                "query_p50_ms":predict["p50_ms"],
                "query_p99_ms":predict["p99_ms"],"repeats":args.repeats,
                "excludes_model_initial_load":True,
            }
            rows.append(result)
            print(json.dumps(result,indent=2),flush=True)
    out={"status":"MEASURED_NOT_ESTIMATED","system":platform.platform(),
         "runtime":"local CPU PyTorch adaptive context-CV, not ONNX",
         "results":rows,
         "not_measured":["Android","mobile peak RSS","energy"],
         "caveat":"Checkpoint bytes exclude Python, model activations and runtime RAM"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
