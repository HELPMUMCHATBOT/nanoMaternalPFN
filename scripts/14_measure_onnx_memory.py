#!/usr/bin/env python3
"""Isolated ONNX Runtime RSS on CPU, without PyTorch loaded in each worker.

Reports total process peak and incremental peak separately. An incremental
value under 32 MB does NOT prove total runtime under 32 MB. Neither number is
Android peak RAM or model-specific allocator usage.
"""
import argparse
import csv
import glob
import json
from pathlib import Path
import subprocess
import sys


def worker(path, repeats, features):
    import os
    import platform
    import resource
    import numpy as np
    import onnxruntime as ort
    import psutil
    graph = Path(path)
    if not graph.stem.startswith("v2_ctx"):
        raise ValueError("graph filename should be v2_ctx<size>.onnx")
    context = int(graph.stem[len("v2_ctx"):])
    process = psutil.Process(os.getpid())
    def rss():
        return process.memory_info().rss / 1e6
    def highwater():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024
    start_rss, start_peak = rss(), highwater()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    loaded_rss = rss()
    X = np.zeros((1, context + 1, features), dtype=np.float32)
    y = (np.arange(context) % 2).astype(np.int64)[None]
    for _ in range(repeats):
        session.run(None, {"x":X, "y_train":y})
    after_rss, total_peak = rss(), highwater()
    return {
        "graph":str(graph), "context":context, "features":features,
        "repeats":repeats, "onnxruntime_version":ort.__version__,
        "platform":platform.platform(), "without_torch":True,
        "rss_before_session_mb":start_rss,
        "rss_after_loading_mb":loaded_rss,
        "rss_after_inference_mb":after_rss,
        "total_process_peak_rss_mb":total_peak,
        "incremental_peak_over_import_mb":max(0.0,total_peak-start_peak),
        "rss_delta_over_import_mb":after_rss-start_rss,
        "onnx_file_mb":graph.stat().st_size/1e6,
        "meets_32mb_total_process_target":total_peak < 32.0,
        "not_measured":"Android, energy, INT8, isolated native model allocations",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob",default="artifacts/results/edge_v2/*/v2_ctx*.onnx")
    parser.add_argument("--output",type=Path,
                        default=Path("artifacts/results/paper_onnx_memory.json"))
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--features",type=int,default=6)
    parser.add_argument("--worker",type=Path,default=None,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.repeats < 1 or args.features < 1:
        parser.error("positive repeats and features required")
    if args.worker is not None:
        print(json.dumps(worker(args.worker,args.repeats,args.features)))
        return
    paths=sorted(Path(p) for p in glob.glob(args.glob))
    if not paths:
        parser.error("No ONNX graphs matched --glob")
    results=[]
    for graph in paths:
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",str(graph),
             "--repeats",str(args.repeats),"--features",str(args.features)]
        process=subprocess.run(cmd,text=True,capture_output=True,check=True)
        row=json.loads(process.stdout)
        results.append(row)
        print(f"{graph.parent.name} {row['context']} | "
              f"total peak {row['total_process_peak_rss_mb']:.1f} MB | "
              f"incremental {row['incremental_peak_over_import_mb']:.1f} MB",
              flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    with args.output.with_suffix(".csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"SAVED: {args.output}")


if __name__=="__main__":
    main()
