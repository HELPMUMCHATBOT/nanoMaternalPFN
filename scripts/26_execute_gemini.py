#!/usr/bin/env python3
"""One-shot local evidence pipeline: frozen 3-expert eval, profile and paper.

Does not invoke AWS or alter original paper_* summary/benchmark data.
May take substantial CPU time because all frozen contexts/draws are evaluated.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-three-expert",action="store_true")
    p.add_argument("--skip-adaptive-profile",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    protected=(
        root/"artifacts/results/paper_all_methods_summary.csv",
        root/"artifacts/results/edge_paper_comparison/profile.json",
        root/"artifacts/results/paper_onnx_memory.json",
    )
    for file in protected:
        if not file.is_file():
            raise FileNotFoundError(f"Original measured result missing: {file}")
    before={str(file):sha(file) for file in protected}
    commands=[]
    if not args.skip_three_expert:
        commands.append([sys.executable,"scripts/09_run_downstream_study.py",
            "--full","--resume","--methods","adaptive_tiny_3expert",
            "--checkpoint-dir","artifacts/results/aws_v2_pretrain",
            "--v1-checkpoint-dir","artifacts/results/aws_full_pretrain",
            "--output-dir","artifacts/results/paper_three_expert_full"])
    if not args.skip_adaptive_profile:
        commands.append([sys.executable,"scripts/25_profile_adaptive_cpu.py"])
    report=[sys.executable,"scripts/23_build_verified_manuscript.py"]
    if not args.skip_three_expert:
        report.append("--include-three-expert")
    commands.append(report)
    commands.append([sys.executable,"scripts/24_package_paper_evidence.py"])
    for cmd in commands:
        print("COMMAND:", " ".join(cmd),flush=True)
        if not args.dry_run:
            subprocess.run(cmd,cwd=root,check=True)
    if not args.dry_run:
        after={str(file):sha(file) for file in protected}
        if before!=after:
            raise RuntimeError("Protected original measured evidence changed")
        print("SUCCESS: measured evidence processed; originals unchanged")


if __name__=="__main__":
    main()
