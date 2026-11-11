"""Regression tests for compact SageMaker code bundles.

Gitignore does not control what is copied into SageMaker source_dir. These
tests protect against reintroducing .venv and multi-gigabyte model artifacts.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sm_submission_test", ROOT / "scripts" / "08_submit_sagemaker.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_compact_source_bundle_excludes_venv_and_results(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sagemaker_train.py").write_text("# entrypoint\n")
    (repo / "requirements.txt").write_text("numpy\n")
    package = repo / "src" / "nanomaternalpfn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "train_replay.py").write_text("print('ready')\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"invalid-bytecode")
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "07_benchmark_sagemaker_a10g.py").write_text("# benchmark\n")
    hidden = repo / ".venv" / "lib"
    hidden.mkdir(parents=True)
    (hidden / "huge.bin").write_bytes(b"x" * (1024 * 1024))
    artifacts = repo / "artifacts" / "results" / "aws_full_pretrain"
    artifacts.mkdir(parents=True)
    (artifacts / "old.pt").write_bytes(b"y" * (1024 * 1024))
    stage = tmp_path / "stage"
    total = module.stage_training_source(repo, stage)
    assert total < 1024
    assert (stage / "sagemaker_train.py").exists()
    assert (stage / "src" / "nanomaternalpfn" / "train_replay.py").exists()
    assert (stage / "scripts" / "07_benchmark_sagemaker_a10g.py").exists()
    assert not (stage / "artifacts").exists()
    assert not (stage / ".venv").exists()
    assert not (stage / "src" / "nanomaternalpfn" / "__pycache__").exists()
