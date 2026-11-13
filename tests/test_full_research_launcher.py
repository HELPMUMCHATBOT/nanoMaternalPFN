"""Full SageMaker research job packaging tests; no AWS calls or fake metrics."""
import importlib.util
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_full_launcher_stage_includes_all_research_code(tmp_path):
    launcher=module("full_launcher",ROOT/"scripts"/"26_launch_complete_c1_c6.py")
    source=tmp_path/"source"
    source.mkdir()
    nbytes=launcher.stage_source(source)
    assert 0<nbytes<10*1024*1024
    assert (source/"sagemaker_research.py").is_file()
    assert (source/"requirements.txt").read_text().find("tabpfn")!=-1
    for name in ("18_run_complete_benchmark.py","23_run_c1_c6_research.py","24_distill_tiny.py"):
        assert (source/"scripts"/name).is_file()
    for src in [source/"sagemaker_research.py", source/"scripts"/"18_run_complete_benchmark.py",
                ROOT/"scripts"/"26_launch_complete_c1_c6.py"]:
        compile(src.read_text(),str(src),"exec")


def test_frozen_inputs_are_checksum_verified_without_aws(tmp_path):
    launcher=module("full_launcher_2",ROOT/"scripts"/"26_launch_complete_c1_c6.py")
    entry=module("full_entry",ROOT/"sagemaker_research.py")
    helper=launcher.load_bundle_helper()
    manifest=tmp_path/"manifest"
    v1=tmp_path/"v1"; v2=tmp_path/"v2"
    for folder in (manifest,v1,v2):
        folder.mkdir()
    for name in helper.DATASETS:
        data=manifest/(name+".npz")
        data.write_bytes(b"synthetic-fixture-not-real-data")
        metadata={"sha256":helper.sha256(data)}
        (manifest/(name+".json")).write_text(json.dumps(metadata))
    for folder in (v1,v2):
        for name in ("tiny_generic","tiny_mixture","base_generic","base_mixture"):
            (folder/(name+".best.inference.pt")).write_bytes(b"dummy-checkpoint-for-bundle-test")
    files=helper.frozen_inputs(tmp_path,manifest,v1,v2)
    assert len(files)==24
    archive=tmp_path/"frozen_input.tar.gz"
    meta=helper.create_input_bundle(files,archive)
    assert len(meta["files"])==24
    channel=tmp_path/"channel"; channel.mkdir()
    archive.rename(channel/"inputs.tar.gz")
    recovered=tmp_path/"recovered"
    assert entry.extract_frozen_inputs(channel,recovered)==24
    assert (recovered/"checkpoints"/"v1"/"tiny_generic.best.inference.pt").exists()
    assert (recovered/"manifests"/(helper.DATASETS[0]+".json")).exists()


def test_full_entry_contains_all_stages_not_only_synthetic_train():
    txt=(ROOT/"sagemaker_research.py").read_text()
    for stage in ("synthetic_training", "synthetic_selection", "distillation", "real_data_evaluation"):
        assert stage in txt
    assert "all_stages_complete" in txt
    assert "18_run_complete_benchmark.py" in txt
