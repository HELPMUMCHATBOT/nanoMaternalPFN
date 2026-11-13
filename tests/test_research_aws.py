"""Validate C1-C6 AWS handoff without contacting AWS or creating charges."""
import json
import subprocess
import sys
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_research_aws_sources_compile_and_stage(tmp_path):
    for script in (
        ROOT/"sagemaker_research.py",
        ROOT/"scripts"/"23_run_c1_c6_research.py",
        ROOT/"scripts"/"24_distill_tiny.py",
        ROOT/"scripts"/"25_submit_research_aws.py",
    ):
        compile(script.read_text(),str(script),"exec")

    import importlib.util
    module_path=ROOT/"scripts"/"25_submit_research_aws.py"
    spec=importlib.util.spec_from_file_location("research_submit",module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    size=module.stage(ROOT,tmp_path)
    assert 0<size<10*1024*1024
    assert (tmp_path/"sagemaker_research.py").is_file()
    assert (tmp_path/"scripts"/"23_run_c1_c6_research.py").is_file()
    assert (tmp_path/"src"/"nanomaternalpfn"/"distillation.py").is_file()


def test_research_aws_launcher_dry_run_no_credentials_needed():
    cmd=[
        sys.executable,str(ROOT/"scripts"/"25_submit_research_aws.py"),
        "--bucket","example-checked-bucket",
        "--role-arn","arn:aws:iam::000000000000:role/TestRole",
    ]
    proc=subprocess.run(cmd,capture_output=True,text=True,check=True)
    assert '"experiment_count": 80' in proc.stdout
    assert '"billable_job_submission": false' in proc.stdout
    assert "DRY RUN" in proc.stdout
