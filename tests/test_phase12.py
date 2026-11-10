"""Phase 1/2 unit tests; synthetic fixtures are never clinical evidence."""
import numpy as np
import pandas as pd
import pytest
import torch
from nanomaternalpfn.priors.hdf5_replay import create_replay,HDF5TaskReplay
from nanomaternalpfn.train_replay import train_replay
from nanomaternalpfn.data.open_sources import _birthwt,_gestation_weeks,SOURCES

def test_variable_replay_resume_and_class_masks(tmp_path):
    path=tmp_path/"prior.h5"
    first=create_replay(path,num_tables=12,seed=9,min_rows=33,max_rows=65,
                        min_features=5,max_features=10,limit=5)
    assert first["completed"]==5 and not first["fully_generated"]
    with pytest.raises(ValueError,match="Partial"):
        HDF5TaskReplay(path)
    done=create_replay(path,num_tables=12,seed=9,min_rows=33,max_rows=65,
                       min_features=5,max_features=10)
    assert done["fully_generated"] and len(done["sha256"])==64
    with HDF5TaskReplay(path) as replay:
        x,y,n_ctx,c=replay.sample_batch(np.random.default_rng(2),batch_size=2,
                                         context_budgets=(16,32),query_rows=4)
        assert x.shape[0]==2 and x.shape[1]==n_ctx+4 and x.shape[2]>=5
        assert int(y.max())<c
        assert len(replay)==12

def test_training_resume_and_safe_inference_checkpoint(tmp_path):
    torch.set_num_threads(1)
    replay=tmp_path/"prior.h5"
    create_replay(replay,num_tables=20,seed=13,min_rows=24,max_rows=40,
                  min_features=5,max_features=8)
    out=tmp_path/"model.pt"
    first=train_replay(replay,out,steps=1,batch_size=2,max_context=16,
                       seed=15,save_every=1)
    assert first["step"]==1 and np.isfinite(first["loss_last"])
    result=train_replay(replay,out,steps=3,batch_size=2,max_context=16,
                        seed=15,resume=True,save_every=1)
    assert result["step"]==3 and np.isfinite(result["loss_last"])
    state=torch.load(out,map_location="cpu",weights_only=False)
    assert len(state["losses"])==3 and state["synthetic_only"]
    inference=torch.load(out.with_name(out.stem+".inference.pt"),
                         map_location="cpu",weights_only=True)
    assert inference["step"]==3 and "state_dict" in inference

def test_prevent_post_outcome_birthweight_leakage(tmp_path):
    d=pd.DataFrame({"rownames":[1,2],"low":[0,1],"age":[25,19],
                    "lwt":[124,95],"race":[1,2],"smoke":[0,1],
                    "ptl":[0,2],"ht":[0,1],"ui":[0,1],"ftv":[2,0],
                    "bwt":[3000,1450]})
    path=tmp_path/"birthwt.csv";d.to_csv(path,index=False)
    x,y=_birthwt(path)
    assert not {"low","bwt","rownames"}.intersection(x)
    assert list(y)==[0,1]

def test_physionet_gestation_and_governance():
    assert _gestation_weeks("12,3")==pytest.approx(12+3/7)
    assert np.isnan(_gestation_weeks(""))
    assert all("numom2b" not in n and "dhs" not in n for n in SOURCES)
    assert SOURCES["vat_gdm"].cohort==SOURCES["vat_delivery"].cohort
    assert SOURCES["pima_diabetes"].tier=="nonmaternal_control"

def test_missing_real_dataset_not_fabricated():
    from nanomaternalpfn.data.open_sources import get_open_task
    with pytest.raises(KeyError,match="Unavailable/unverified"):
        get_open_task("fake_maternal_cohort")
