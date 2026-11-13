#!/usr/bin/env python3
"""Single SageMaker job: C1-C6 training, distillation and full clinical benchmarking.

All results are measured. Synthetic validation selects research candidates;
real-data evaluation remains explicitly exploratory. The process fails if
a phase is incomplete, and never writes a fabricated AUROC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def log_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_command(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "src"), env.get("PYTHONPATH", ""),
    )))
    print("EXECUTING:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


def extract_frozen_inputs(channel_dir, destination):
    channel_dir = Path(channel_dir)
    inputs = sorted(channel_dir.glob("*.tar.gz"))
    if len(inputs) != 1:
        raise RuntimeError("expected exactly one staged frozen dataset/checkpoint archive")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(inputs[0], "r:gz") as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("unsafe input path")
            if not member.isfile():
                raise RuntimeError("unsupported archive member")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("missing archived content")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
    metadata = json.loads((destination / "input_manifest.json").read_text())
    for item in metadata:
        entry = destination / item["relative"]
        if not entry.is_file():
            raise RuntimeError("frozen input file missing: " + item["relative"])
        digest = hashlib.sha256(entry.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError("frozen input integrity check failed: " + item["relative"])
    return len(metadata)


def select_on_common_synthetic_probes(measured, working, device, seed, validation_tables):
    from nanomaternalpfn.priors.hdf5_replay import create_replay
    from nanomaternalpfn.priors.hdf5_replay import HDF5TaskReplay
    from nanomaternalpfn.distillation import load_teacher
    from nanomaternalpfn.validation_replay import evaluate_replay

    working = Path(working)
    probes = []
    for variant in ("v2", "v3_robust"):
        path = working / "probes" / f"{variant}_{seed}.h5"
        create_replay(
            path, num_tables=validation_tables, seed=seed,
            prior="mixture", variant=variant, maternal_probability=.7,
        )
        probes.append(path)
    contexts = [16, 32, 64, 128, 256, 512]
    rank = []
    for entry in measured:
        path = Path(entry["inference_checkpoint"])
        if not path.is_file():
            raise RuntimeError("missing best checkpoint: " + str(path))
        model = load_teacher(path, device=device)
        scores = []
        for probe in probes:
            with HDF5TaskReplay(probe) as replay:
                metrics = evaluate_replay(
                    model, replay, contexts=contexts, device=device,
                    batch_size=4, batches_per_context=2, query_rows=8,
                    draw_seed=seed+1,
                )
                scores.append(metrics["loss"])
        rank.append({
            "id":entry["id"], "checkpoint":str(path),
            "synthetic_probe_losses":scores,
            "mean_common_probe_loss":sum(scores)/len(scores),
        })
        del model
    rank.sort(key=lambda item:item["mean_common_probe_loss"])
    log_json(working / "common_probe_ranking.json", rank)
    if len(rank) < 2:
        raise RuntimeError("need at least two trained experts for ensemble distillation")
    return rank, probes


def main():
    p = argparse.ArgumentParser()
    for name,default in (
        ("variants","v2,v3_robust"),
        ("maternal-fractions","0,.3,.5,.7,1"),
        ("step-budgets","2500,5000,10000,25000"),
        ("encoders","linear,gated"),
        ("normalizations","standard,robust"),
        ("tiers","tiny"),
    ):
        p.add_argument("--"+name, default=default)
    p.add_argument("--num-tables", type=int, default=200000)
    p.add_argument("--validation-tables", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--microbatch-size", type=int, default=8)
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--train-seed", type=int, default=20261107)
    p.add_argument("--validation-seed", type=int, default=20271008)
    p.add_argument("--distill-steps", type=int, default=25000)
    a = p.parse_args()

    model_dir = Path(os.getenv("SM_MODEL_DIR","artifacts/sagemaker/c1_c6_full_model"))
    output_dir = Path(os.getenv("SM_OUTPUT_DATA_DIR","artifacts/sagemaker/c1_c6_full_output"))
    checkpoint_dir = Path("/opt/ml/checkpoints/c1_c6" if os.getenv("SM_MODEL_DIR")
                          else "artifacts/sagemaker/c1_c6_full_checkpoint")
    model_dir.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    input_dir = checkpoint_dir / "frozen"
    phase = {"synthetic_training":False,"synthetic_selection":False,
             "distillation":False,"real_data_evaluation":False,
             "all_stages_complete":False}
    log_json(output_dir/"phase_status.json",phase)
    input_channel = Path(os.getenv(
        "SM_CHANNEL_RESEARCH_INPUT", "/opt/ml/input/data/research_input",
    ))
    extract_frozen_inputs(input_channel, input_dir)
    (checkpoint_dir/"research").mkdir(parents=True,exist_ok=True)
    research = checkpoint_dir/"research"

    command = [
        sys.executable, ROOT/"scripts/23_run_c1_c6_research.py",
        "--execute","--resume","--device","cuda",
        "--output",research,
        "--num-tables",a.num_tables,
        "--validation-tables",a.validation_tables,
        "--batch-size",a.batch_size,
        "--microbatch-size",a.microbatch_size,
        "--validate-every",a.validate_every,
        "--max-context",a.max_context,
        "--train-seed",a.train_seed,
        "--validation-seed",a.validation_seed,
    ]
    for flag,value in (
        ("variants",a.variants),("maternal-fractions",a.maternal_fractions),
        ("step-budgets",a.step_budgets),("encoders",a.encoders),
        ("normalizations",a.normalizations),("tiers",a.tiers),
    ):
        toks = [str(x).strip() for x in value.split(",") if x.strip()]
        if not toks:
            raise ValueError("empty experimental factor " + flag)
        command.extend(["--"+flag,*toks])
    run_command(command)
    measured = json.loads((research/"measured_results.json").read_text())
    planned = json.loads((research/"research_plan.json").read_text())["experiments"]
    if len(measured) != len(planned):
        raise RuntimeError("Incomplete synthetic factorial study")
    phase["synthetic_training"] = True
    log_json(output_dir/"phase_status.json",phase)
    shutil.copy2(research/"measured_results.json",output_dir/"measured_synthetic_training.json")

    ranking,probes = select_on_common_synthetic_probes(
        measured,research,"cuda",a.validation_seed+1000,a.validation_tables,
    )
    candidate_dir = input_dir/"checkpoints"/"c2c3c5c6"
    candidate_dir.mkdir(parents=True,exist_ok=True)
    selected_path = candidate_dir/"v3_selected_tiny.best.inference.pt"
    shutil.copy2(ranking[0]["checkpoint"],selected_path)
    log_json(candidate_dir/"selection_provenance.json",ranking)
    shutil.copy2(research/"common_probe_ranking.json",output_dir/"common_probe_ranking.json")
    phase["synthetic_selection"] = True
    log_json(output_dir/"phase_status.json",phase)

    from nanomaternalpfn.distillation import distill
    by_id = {e["id"]:e for e in measured}
    distinct_teachers = []
    seen_trajectories = set()
    for item in ranking:
        trajectory = by_id[item["id"]]["training_trajectory_id"]
        if trajectory not in seen_trajectories:
            distinct_teachers.append(item["checkpoint"])
            seen_trajectories.add(trajectory)
        if len(distinct_teachers) == 2:
            break
    if len(distinct_teachers) != 2:
        raise RuntimeError("fewer than two independent pretrained trajectories for distillation")
    best_config = by_id[ranking[0]["id"]]
    train_replay = research/"priors"/(
        f"p{round(best_config['maternal_fraction']*100):03d}_"
        f"{best_config['variant']}_train.h5"
    )
    # Use independent synthetic replay for teacher-student training,
    # rather than reuse the common ranking probe tasks.
    from nanomaternalpfn.priors.hdf5_replay import create_replay
    distill_train = research/"distillation"/"train.h5"
    distill_val = research/"distillation"/"val.h5"
    for path,sd in ((distill_train,a.validation_seed+2000),
                    (distill_val,a.validation_seed+2001)):
        create_replay(
            path,num_tables=a.num_tables if path==distill_train else a.validation_tables,
            seed=sd,prior="mixture",variant=best_config["variant"],
            maternal_probability=best_config["maternal_fraction"],
        )
    student_dir = input_dir/"checkpoints"/"c4"
    student_dir.mkdir(parents=True,exist_ok=True)
    student_path = student_dir/"distilled_tiny.pt"
    # A partial best checkpoint is not a completed 25k-step distillation.
    # Resuming requires the saved optimizer, RNG state and completion report.
    report = distill(
        distill_train,distinct_teachers,
        student_path,steps=a.distill_steps,
        batch_size=a.batch_size,microbatch_size=a.microbatch_size,
        max_context=a.max_context,
        validation_path=distill_val,
        validate_every=a.validate_every,device="cuda",
        resume=True, checkpoint_every=25,
    )
    log_json(output_dir/"distillation_measured.json",report)
    if report["steps_completed"] != a.distill_steps or not (
        student_dir/"distilled_tiny.best.inference.pt"
    ).is_file():
        raise RuntimeError("distillation incomplete despite available partial weights")
    phase["distillation"] = True
    log_json(output_dir/"phase_status.json",phase)

    methods = [
        "adaptive_tiny","adaptive_tiny_three","adaptive_tiny_four",
        "adaptive_base","adaptive_base_three","adaptive_base_four",
        "tiny_mixture","base_mixture","v3_selected_tiny","distilled_tiny",
        "logistic","random_forest","xgboost","lightgbm",
        "tabpfn_v2_single","tabpfn_v2_default",
    ]
    run_command([
        sys.executable,ROOT/"scripts/18_run_complete_benchmark.py",
        "--input-dir",input_dir,
        "--output-dir",checkpoint_dir/"real_data_benchmark",
        "--methods",*methods,
        "--device","cuda",
    ])
    benchmark_dir=checkpoint_dir/"real_data_benchmark"
    shutil.copy2(benchmark_dir/"summary.csv",output_dir/"real_data_summary.csv")
    shutil.copy2(benchmark_dir/"all_draws.csv",output_dir/"real_data_all_draws.csv")
    shutil.copy2(benchmark_dir/"runtime_versions.json",output_dir/"runtime_versions.json")
    phase["real_data_evaluation"] = True
    phase["all_stages_complete"] = True
    log_json(output_dir/"phase_status.json",phase)
    log_json(model_dir/"phase_status.json",phase)
    # Copy small, selected inference models into model artifact.
    for source in (selected_path,student_dir/"distilled_tiny.best.inference.pt"):
        shutil.copy2(source,model_dir/source.name)
    print("COMPLETE: C1-C6 training, synthetic selection, distillation, and frozen evaluation",flush=True)


if __name__=="__main__":
    main()
