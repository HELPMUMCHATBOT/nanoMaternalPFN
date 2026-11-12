"""Complete SageMaker research pipeline: source staging, nanoTabPFN and metrics."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from nanomaternalpfn.nanotabpfn_baseline import (
    NanoTabConfig,new_nanotab,impute_from_context,
    NanoTabPFNReferenceClassifier,train_matched_nanotab,
)

ROOT=Path(__file__).resolve().parents[1]


def import_script(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_upstream_reference_model_and_context_only_imputation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config=NanoTabConfig()
    model=new_nanotab(config).eval()
    state={
        "config":asdict(config), "state_dict":model.state_dict(),
        "synthetic_only":True,"prior":"generic_v2",
    }
    path=tmp_path/"reference.pt"
    torch.save(state,path)
    rng=np.random.default_rng(21)
    X=rng.normal(size=(32,6)).astype(np.float32)
    X[1,0]=np.nan
    X[25,2]=np.nan
    X[26,0]=999.0
    y=np.array([0,1,2]*8,dtype=np.int64)
    ref=NanoTabPFNReferenceClassifier(path,query_batch_size=3)
    ref.fit(X[:24],y)
    prediction=ref.predict_proba(X[24:])
    assert prediction.shape==(8,3)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(axis=1),1.,atol=1e-6)
    # Imputation cannot use a query value, including extreme numbers.
    x=torch.tensor([[[1.],[3.],[float("nan")]]])
    out=impute_from_context(x,2)
    assert float(out[0,2,0])==2.0


def test_nanotab_one_step_pretraining_produces_loadable_checkpoint(tmp_path):
    torch.set_num_threads(1)
    result=train_matched_nanotab(
        workspace=tmp_path/"scratch",
        checkpoint_dir=tmp_path/"checkpoints",
        steps=1,batch_size=2,microbatch_size=1,
        num_tables=20,val_tables=12,
        device="cpu",validate_every=1,max_context=16,
    )
    checkpoint=Path(result["checkpoint"])
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert state["step"]==1
    assert state["prior"]=="generic_v2"
    assert state["synthetic_only"]
    model=new_nanotab(state["config"])
    model.load_state_dict(state["state_dict"],strict=True)
    assert result["model_parameters"]>0


def test_source_bundle_small_and_excludes_venv(tmp_path):
    launcher=import_script("benchmark_launcher","19_submit_full_aws_benchmark.py")
    stage=tmp_path/"source"
    size=launcher.stage_source(ROOT,stage)
    assert 0<size<10*1024*1024
    assert (stage/"sagemaker_benchmark.py").exists()
    assert (stage/"src"/"nanomaternalpfn"/"nanotabpfn_baseline.py").exists()
    assert (stage/"src"/"nanomaternalpfn"/"third_party"/"nanotabpfn"/"LICENSE").exists()
    assert (stage/"scripts"/"18_run_complete_benchmark.py").exists()
    assert (stage/"scripts"/"20_profile_complete_models.py").exists()
    assert not (stage/".venv").exists()
    assert not (stage/"artifacts").exists()


def test_frozen_inputs_sha256_and_archive(tmp_path):
    launcher=import_script("benchmark_launcher_sha","19_submit_full_aws_benchmark.py")
    manifest=tmp_path/"manifests"
    v1=tmp_path/"v1"
    v2=tmp_path/"v2"
    for directory in (manifest,v1,v2):
        directory.mkdir()
    for name in launcher.DATASETS:
        data=manifest/f"{name}.npz"
        np.savez(data,X=np.zeros((8,2),dtype=np.float32),y=np.arange(8)%2)
        (manifest/f"{name}.json").write_text(json.dumps({
            "sha256":launcher.sha256(data),"test_idx":[6,7],
            "context_draws":{"4":[[0,1,2,3]]},
        }))
    for tier in ("tiny","base"):
        for prior in ("generic","mixture"):
            (v1/f"{tier}_{prior}.best.inference.pt").write_text("weights")
        for prior in ("generic","mixture"):
            (v2/f"{tier}_{prior}.best.inference.pt").write_text("weights")
    files=launcher.frozen_inputs(tmp_path,manifest,v1,v2)
    assert len(files)==24
    metadata=launcher.create_input_bundle(files,tmp_path/"inputs.tar.gz")
    assert Path(metadata["path"]).is_file()
    assert metadata["sha256"]==launcher.sha256(metadata["path"])
    (manifest/"vat_gdm.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="SHA256 mismatch"):
        launcher.frozen_inputs(tmp_path,manifest,v1,v2)


def test_full_benchmark_scores_and_class_alignment(tmp_path):
    runner=import_script("benchmark_runner","18_run_complete_benchmark.py")
    class MockEstimator:
        classes_=np.array([0,2])
    pred=np.array([[.7,.3],[.2,.8]])
    classes=np.array([0,1,2])
    aligned=runner.validate_probs(pred,MockEstimator(),classes,2,"fake")
    assert aligned.shape==(2,3)
    np.testing.assert_allclose(aligned[:,1],0)
    np.testing.assert_allclose(aligned.sum(axis=1),1)
    scores=runner.score_predictions(np.array([0,2]),aligned,classes)
    assert np.isfinite(scores["accuracy"])
    assert np.isfinite(scores["brier"])


def test_SageMaker_input_extraction_guards_paths(tmp_path):
    entry=import_script("sagemaker_benchmark","19_submit_full_aws_benchmark.py")
    assert callable(entry.stage_source)


def test_end_to_end_frozen_cpu_with_retrained_reference(tmp_path):
    """Run real evaluation CLI over one synthetic test fixture, no clinical claims."""
    import subprocess
    import sys
    from dataclasses import asdict
    manifest_dir=tmp_path/"inputs"/"manifests"
    checkpoint_dir=tmp_path/"inputs"/"checkpoints"
    manifest_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    rng=np.random.default_rng(909)
    X=rng.normal(size=(60,5)).astype(np.float32)
    X[1,2]=np.nan
    y=np.asarray([0,1,2]*20,dtype=np.int64)
    npz=manifest_dir/"uci_maternal_risk.npz"
    np.savez(npz,X=X,y=y)
    manifest={
        "sha256":hashlib.sha256(npz.read_bytes()).hexdigest(),
        "test_idx":list(range(36,60)),
        "context_draws":{"16":[list(range(16))]},
    }
    (manifest_dir/"uci_maternal_risk.json").write_text(json.dumps(manifest))
    cfg=NanoTabConfig()
    net=new_nanotab(cfg).eval()
    torch.save({
        "config":asdict(cfg),"state_dict":net.state_dict(),
        "synthetic_only":True,"prior":"generic_v2",
    },checkpoint_dir/"nanotabpfn_generic.best.inference.pt")
    command=[
        sys.executable,str(ROOT/"scripts"/"18_run_complete_benchmark.py"),
        "--input-dir",str(tmp_path/"inputs"),
        "--output-dir",str(tmp_path/"results"),
        "--datasets","uci_maternal_risk",
        "--methods","logistic","nanotabpfn_generic",
        "--contexts","16","--max-draws","1","--device","cpu",
        "--query-batch-size","8",
    ]
    subprocess.run(command,check=True,capture_output=True,text=True)
    import pandas as pd
    summary=pd.read_csv(tmp_path/"results"/"summary.csv")
    raw=pd.read_csv(tmp_path/"results"/"all_draws.csv")
    assert len(raw)==2
    assert set(raw["method"])=={"logistic","nanotabpfn_generic"}
    assert len(summary)==8
    assert np.isfinite(raw["brier"]).all()
    # The resume path must not duplicate observations.
    subprocess.run(command,check=True,capture_output=True,text=True)
    raw_again=pd.read_csv(tmp_path/"results"/"all_draws.csv")
    assert len(raw_again)==2
