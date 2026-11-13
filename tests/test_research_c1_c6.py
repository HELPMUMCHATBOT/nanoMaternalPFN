"""C1-C6 research regression tests. Synthetic-only smoke measurements."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch

from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig
from nanomaternalpfn.priors.maternal_scm import sample_task
from nanomaternalpfn.priors.hdf5_replay import create_replay, HDF5TaskReplay
from nanomaternalpfn.distillation import distill
from nanomaternalpfn.train_replay import train_replay


torch.set_num_threads(1)


class FrozenExpert:
    def __init__(self, sign):
        self.sign = sign

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        z = np.clip(np.asarray(X)[:, 0] * self.sign * 5, -20, 20)
        p = 1 / (1 + np.exp(-z))
        return np.column_stack((1-p, p))


def test_c1_three_experts_context_only_and_winner():
    X = np.tile(np.array([[-2.], [-1.], [1.], [2.]],dtype=np.float32), (4,1))
    y = (X[:,0] > 0).astype(int)
    factory = lambda path: FrozenExpert(1 if 'good' in str(path) else -1)
    models = ['badA', 'good', 'badB']
    selector = AdaptiveNanoMaternalPFNClassifier(
        models, estimator_factory=factory, weighting='winner', calibrate=False,
    ).fit(X, y)
    assert selector.diagnostics_['context_only'] is True
    assert selector.diagnostics_['cv_folds'] == 4
    assert selector.weights_.tolist() == [0.,1.,0.]
    assert selector.predict_proba(X[:2]).shape == (2,2)
    # Query input values may change predictions but cannot change CV weights.
    before = selector.weights_.copy()
    selector.predict_proba(np.array([[1000.], [-1000.]],dtype=np.float32))
    np.testing.assert_array_equal(before, selector.weights_)


def test_c2_robust_prior_and_c6_mixture_replay(tmp_path):
    x, y, meta = sample_task(
        96,8,2,prior='maternal',seed=153,
        robust=True,augment_class_permutation=True,
    )
    assert meta['robust'] is True
    assert x.shape == (96,8) and set(np.unique(y)).issubset({0,1})
    path = tmp_path/'mixture.h5'
    state = create_replay(
        path,num_tables=16,seed=20,prior='mixture',variant='v3_robust',
        maternal_probability=.3,min_rows=24,max_rows=40,max_features=8,
    )
    assert state['fully_generated'] is True
    with HDF5TaskReplay(path) as replay:
        assert replay.settings['maternal_probability'] == pytest.approx(.3)
        assert replay.settings['variant'] == 'v3_robust'
        assert replay.settings['robust_domain_shift_and_label_noise']
        batch = replay.sample_batch(np.random.default_rng(9),2,(16,),query_rows=4)
        assert batch[0].shape[0] == 2


def test_c3_gated_robust_normalization_permutation_and_query_isolation():
    torch.manual_seed(12)
    model = NanoMaternalPFN(ModelConfig(
        24,4,1,48,architecture='pre_ln_v2',
        feature_encoder_type='gated',normalization='robust',
    )).eval()
    x=torch.randn(1,9,6)
    x[0,2,3]=float('nan')
    y=torch.tensor([[0,1,0,1,0,1]],dtype=torch.long)
    with torch.inference_mode():
        a=model(x,y,2)
        b=model(x[:,:,[2,4,0,5,1,3]],y,2)
        x2=x.clone()
        x2[:,6:]+=2000
        encoding1=model.feature_encoder.encode_inputs(x,6,'robust')
        encoding2=model.feature_encoder.encode_inputs(x2,6,'robust')
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(encoding1[:,:6],encoding2[:,:6])
    assert torch.isfinite(a).all()


def test_c4_distillation_and_c5_real_optimizer_steps(tmp_path):
    train=tmp_path/'train.h5';val=tmp_path/'val.h5'
    for path,seed in ((train,10),(val,11)):
        create_replay(
            path,num_tables=20,seed=seed,prior='mixture',
            variant='v3_robust',maternal_probability=.5,
            min_rows=24,max_rows=40,max_features=8,
        )
    teacher_cfg=ModelConfig(24,4,1,48,architecture='pre_ln_v2')
    teacher=NanoMaternalPFN(teacher_cfg).eval()
    teacher_path=tmp_path/'teacher.pt'
    torch.save({'config':asdict(teacher_cfg),'state_dict':teacher.state_dict()},teacher_path)
    out=tmp_path/'student.pt'
    result=distill(
        train,[teacher_path],out,steps=1,batch_size=2,
        microbatch_size=1,max_context=16,query_rows=4,
        validation_path=val,validate_every=1,device='cpu',
    )
    assert out.exists()
    assert result['steps_completed']==1
    assert result['task_presentations']==2
    assert result['clinical_auroc_measured'] is False
    assert result['validation_history']
    checkpoint=torch.load(out,weights_only=True)
    assert checkpoint['config']['feature_encoder_type']=='gated'
    trained=tmp_path/'ordinary.pt'
    ordinary=train_replay(
        train,trained,tier='tiny',variant='v3_robust',
        steps=1,batch_size=2,microbatch_size=1,
        max_context=16,query_rows=4,validation_path=val,
        validate_every=1,validation_batch_size=2,
        validation_batches_per_context=1,save_every=1,
        feature_encoder_type='gated',normalization='robust',
    )
    assert ordinary['step']==1
    assert ordinary['validation_history']
    assert ordinary['synthetic_only'] is True
