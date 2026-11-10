"""Regression test: SageMaker benchmark child process can import src-layout package."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The SageMaker entrypoint lives at repository root rather than in the
# installed src-layout package. pytest's console entrypoint does not always
# put that root on sys.path, so load the explicit source path.
import importlib.util

_ENTRYPOINT = Path(__file__).resolve().parents[1] / "sagemaker_train.py"
_spec = importlib.util.spec_from_file_location("sagemaker_train", _ENTRYPOINT)
sagemaker_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sagemaker_train)


def test_benchmark_propagates_src_pythonpath(tmp_path):
    captured = {}
    def execute(command, *, check, env):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    args = SimpleNamespace(device="cuda", batch_size=8, steps=2500)
    with patch.object(sagemaker_train.subprocess, "run", side_effect=execute):
        sagemaker_train._benchmark(args, tmp_path)

    assert captured["command"][1].endswith("07_benchmark_sagemaker_a10g.py")
    assert str(sagemaker_train.ROOT / "src") in captured["env"]["PYTHONPATH"].split(os.pathsep)

    # Validate using an actual fresh child interpreter, not only a mocked call.
    env = os.environ.copy()
    env["PYTHONPATH"] = captured["env"]["PYTHONPATH"]
    result = subprocess.run(
        [sys.executable, "-c", "from nanomaternalpfn.model import NanoMaternalPFN; print('import-ok')"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "import-ok" in result.stdout
