#!/usr/bin/env python3
"""Check fixed-shape ONNX outputs versus PyTorch on missing and 2/3-class data."""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,
                        default=Path("artifacts/results/aws_v2_pretrain"))
    parser.add_argument("--edge-dir",type=Path,
                        default=Path("artifacts/results/edge_v2"))
    parser.add_argument("--contexts",nargs="+",type=int,default=[16,64,128])
    parser.add_argument("--tolerance",type=float,default=1e-3)
    args=parser.parse_args()
    torch.set_num_threads(1)
    count=0
    for arm in ("tiny_generic","tiny_mixture","base_generic","base_mixture"):
        state=torch.load(args.checkpoint_dir/f"{arm}.best.inference.pt",
                         map_location="cpu",weights_only=True)
        model=NanoMaternalPFN(ModelConfig(**state["config"]))
        model.load_state_dict(state["state_dict"],strict=True)
        model.eval()
        for n in args.contexts:
            graph=args.edge_dir/arm/f"v2_ctx{n}.onnx"
            session=ort.InferenceSession(str(graph),
                                         providers=["CPUExecutionProvider"])
            rng=np.random.default_rng(20261111)
            x=rng.normal(size=(1,n+1,6)).astype(np.float32)
            x[0,1,0]=np.nan
            for classes in (2,3):
                y=(np.arange(n)%classes).astype(np.int64)[None]
                with torch.inference_mode():
                    expected=model(torch.from_numpy(x),
                                   torch.from_numpy(y)).numpy()
                actual=session.run(None,{"x":x,"y_train":y})[0]
                err=float(np.max(np.abs(expected-actual)))
                passed=np.allclose(expected,actual,
                                   atol=args.tolerance,rtol=args.tolerance)
                print(f"{arm} n={n} classes={classes} "
                      f"max_abs_error={err:.7g} {'PASS' if passed else 'FAIL'}")
                if not passed:
                    raise AssertionError(f"Parity failure: {arm}/{n}/{classes}")
                count+=1
    print(f"PASS: {count} graph parity comparisons")


if __name__=="__main__":
    main()
