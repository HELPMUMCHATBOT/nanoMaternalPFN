#!/usr/bin/env python3
"""SageMaker entrypoint for the nanoMaternalPFN GPU benchmark and 2x2 study.

This runs only synthetic-prior pretraining, not clinical validation.
The generated HDF5 archives live on instance scratch storage. Final model
weights, resumable training states, and provenance are saved separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _benchmark(args, output_dir: Path) -> None:
    if args.device != "cuda":
        raise ValueError("SageMaker benchmark must use CUDA")
    path = output_dir / "a10g_throughput.json"
    command = [
        sys.executable, str(ROOT / "scripts" / "07_benchmark_sagemaker_a10g.py"),
        "--tiers", "tiny", "base",
        "--contexts", "16", "128", "512",
        "--batch-size", str(args.batch_size),
        "--warmup", "3", "--timed", "8",
        "--steps", str(args.steps),
        "--output", str(path),
    ]
    # SageMaker's source_dir contains the package under src/, but subprocesses
    # do not inherit this process's sys.path. Export an absolute PYTHONPATH so
    # the benchmark script can import nanomaternalpfn from a fresh interpreter.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "src"), env.get("PYTHONPATH", ""),
    )))
    subprocess.run(command, check=True, env=env)


def _train(args, model_dir: Path, output_dir: Path, checkpoint_dir: Path) -> None:
    import h5py
    import numpy as np
    import torch
    from nanomaternalpfn.priors.hdf5_replay import create_replay
    from nanomaternalpfn.train_replay import train_replay

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA GPU is available")
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    provenance = {"synthetic_only": True, "seed": args.seed,
                  "num_tables_per_prior": args.num_tables, "archives": [],
                  "config": {"steps_per_arm": args.steps,
                             "effective_batch_size": args.batch_size,
                             "microbatch_size": args.microbatch_size,
                             "gradient_accumulation_steps": args.batch_size // args.microbatch_size,
                             "context_max": args.max_context,
                             "query_rows": 8}}
    _write_json(output_dir / "job_config.json", provenance["config"])
    for prior in ("generic", "mixture"):
        replay = scratch / f"{prior}_{args.num_tables}.h5"
        generation = create_replay(
            replay, num_tables=args.num_tables, seed=args.seed,
            prior=prior, min_rows=args.min_rows, max_rows=args.max_rows,
            min_features=5, max_features=args.max_features,
            maternal_probability=0.7,
        )
        if not generation["fully_generated"]:
            raise RuntimeError(f"Incomplete synthetic archive: {replay}")
        with h5py.File(replay, "r") as h5:
            n = int(h5.attrs["completed"])
            count_maternal = int(np.count_nonzero(h5["is_maternal"][:]))
            n_binary = int(np.count_nonzero(h5["n_classes"][:] == 2))
            n_three = n - n_binary
        if n != args.num_tables or (prior == "generic" and count_maternal):
            raise RuntimeError("Synthetic archive validation failed")
        provenance["archives"].append({
            "prior": prior, "completed": n, "maternal": count_maternal,
            "generic": n - count_maternal, "binary": n_binary,
            "three_class": n_three, "sha256": generation["sha256"],
            "settings": {k: generation[k] for k in (
                "seed", "min_rows", "max_rows", "min_features", "max_features",
                "maternal_probability", "num_tables", "schema_version")},
        })
        _write_json(output_dir / "generated_prior_provenance.json", provenance)
        print(f"Validated {prior}: {n} synthetic tasks, {count_maternal} maternal; SHA-256 {generation['sha256']}", flush=True)
        for tier in ("tiny", "base"):
            ckpt = checkpoint_dir / f"{tier}_{prior}.pt"
            result = train_replay(
                replay, ckpt, tier=tier, steps=args.steps,
                batch_size=args.batch_size, microbatch_size=args.microbatch_size, seed=args.seed,
                device=args.device, max_context=args.max_context,
                query_rows=8, save_every=max(1, args.steps // 10),
                resume=ckpt.exists(),
            )
            for src in (ckpt, ckpt.with_name(ckpt.stem + ".inference.pt")):
                shutil.copy2(src, model_dir / src.name)
            result["prior"] = prior
            result["status"] = "complete"
            runs.append(result)
            _write_json(output_dir / "phase12_training_runs.json", runs)
            print(f"Finished {tier}/{prior}: step {result['step']}, loss {result['loss_last']}", flush=True)
    _write_json(model_dir / "synthetic_prior_provenance.json", provenance)
    _write_json(model_dir / "training_results.json", runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "train"), required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--num-tables", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Effective tasks per optimizer step")
    parser.add_argument("--microbatch-size", type=int, default=8,
                        help="Tasks per forward/backward; four accumulations at defaults")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--min-rows", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=576)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--scratch-dir", default="/tmp/nanomaternalpfn-priors")
    args = parser.parse_args()
    if args.steps < 1 or args.num_tables < 1 or args.batch_size < 1:
        raise ValueError("steps, num-tables, batch-size must be positive")
    if args.microbatch_size < 1 or args.batch_size % args.microbatch_size:
        raise ValueError("microbatch-size must be a positive divisor of batch-size")
    if args.max_context < 16:
        raise ValueError("max-context must be at least 16")
    model_dir = Path(os.getenv("SM_MODEL_DIR", "artifacts/sagemaker/model"))
    # SageMaker archives non-model artifacts only from /opt/ml/output/data.
    # SM_OUTPUT_DIR can point to its parent /opt/ml/output in some toolkit versions.
    if os.getenv("SM_MODEL_DIR"):
        output_dir = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    else:
        output_dir = Path("artifacts/sagemaker/output")
    checkpoint_dir = Path("/opt/ml/checkpoints" if os.getenv("SM_MODEL_DIR") else "artifacts/sagemaker/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        _benchmark(args, output_dir)
    else:
        _train(args, model_dir, output_dir, checkpoint_dir)


if __name__ == "__main__":
    main()
