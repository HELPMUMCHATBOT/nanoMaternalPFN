"""Contract tests use explicitly synthetic FIXTURES, never research outcomes."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.experiments.common import load_frozen,sha256,aligned_probs,discover_missing_inputs
from scripts.experiments.predictors import build_classical,tuned_baseline,ContextOnlyExpertEnsemble
from scripts.experiments.run_predictions import perturb
from scripts.experiments.training_matrix import make_matrix,case_command
from scripts.experiments.analyze_predictions import paired_bootstrap
from scripts.experiments.status import gate


def fixture(tmp_path,name='unit_fixture',external=False):
    rng=np.random.default_rng(99)
    x=rng.normal(size=(80,6)).astype('float32')
    y=np.tile([0,1],40)
    p=tmp_path/f'{name}.npz'; np.savez_compressed(p,X=x,y=y)
    m={'sha256':sha256(p),'test_idx':list(range(60,80)),
       'context_draws':{'16':[list(range(16))]}}
    if external:m['external_validation']={'source_accession':'INDEPENDENT_TEST','population':'new cohort',
       'outcome_definition':'test binary outcome','data_license':'approved','frozen_before_model_evaluation':True,
       'model_selection_freeze_sha':'1'*40,'independent_of_development_cohorts':True,
       'independent_reviewer':'test reviewer'}
    j=tmp_path/f'{name}.json';j.write_text(json.dumps(m))
    return j


def test_hash_and_context_boundary(tmp_path):
    p=fixture(tmp_path)
    d=load_frozen(p)
    assert len(d.test)==20 and d.contexts['16'][0]==list(range(16))
    m=json.loads(p.read_text());m['context_draws']['16'][0][0]=60;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='overlaps'):load_frozen(p)


def test_reject_tampered_data(tmp_path):
    p=fixture(tmp_path);d=p.with_suffix('.npz')
    with d.open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='SHA256'):load_frozen(p)


def test_external_requires_unseen_source_and_freeze(tmp_path):
    p=fixture(tmp_path,'uci_maternal_risk',external=True)
    with pytest.raises(ValueError,match='Previously examined'):load_frozen(p,external=True)
    p=fixture(tmp_path,'independent_new',external=True)
    assert load_frozen(p,external=True).source=='INDEPENDENT_TEST'
    m=json.loads(p.read_text());m['external_validation'].pop('independent_reviewer');p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='provenance incomplete'):load_frozen(p,external=True)


def test_invalid_probs_fail_closed():
    class Est: classes_=np.array([0,1])
    assert np.allclose(aligned_probs(Est(),np.array([[.8,.2]]),[0,1]),[[.8,.2]])
    with pytest.raises(ValueError,match='Invalid predicted'):aligned_probs(Est(),np.array([[2.,.5]]),[0,1])


def test_context_only_tuning_is_operational():
    x=np.arange(60,dtype=float).reshape(20,3)
    y=np.tile([0,1],10)
    estimator,meta=tuned_baseline('logistic',x,y,seed=7,n_iter=2)
    assert meta['inner_cv_folds']==3 and set(meta['best_params'])=={'est__C','est__class_weight'}
    assert estimator.predict_proba(x[:2]).shape==(2,2)


def test_training_matrix_is_controlled():
    c=make_matrix();assert len(c)==11 and len({x['id'] for x in c})==11
    ref=next(x['factors'] for x in c if x['id']=='E2_reference')
    for x in c:
        if x['experiment'] in ('E2','E7'):
            changes=[k for k in ref if ref[k]!=x['factors'][k]]
            assert len(changes)<=1
    long=next(x for x in c if x['experiment']=='E11')
    assert long['factors']['steps']==20000
    cmd=case_command(long,'out','cuda')
    assert '--steps' in cmd and cmd[cmd.index('--steps')+1]=='20000'


def test_corruptions_are_deterministic_and_not_training_data():
    x=np.ones((20,4),dtype='float32');ref=np.arange(80).reshape(20,4)
    a=perturb(x,ref,'missing_0.25',123)
    assert np.array_equal(np.isnan(a),np.isnan(perturb(x,ref,'missing_0.25',123)))
    assert 0<np.isnan(a).sum()<len(a.flat)
    assert np.array_equal(perturb(x,ref,'clean',1),x)
    assert not np.array_equal(perturb(x,ref,'noise_0.25',1),x)


def test_patient_level_pairing_rejects_mismatched_people(tmp_path):
    y=np.tile([0,1],30)
    p=np.where(y==1,.8,.2);q=np.where(y==1,.7,.3)
    f1=tmp_path/'a.npz';f2=tmp_path/'b.npz'
    np.savez_compressed(f1,y_true=y,probs=np.column_stack([1-p,p]),classes=np.array([0,1]),test_row_index=np.arange(60))
    np.savez_compressed(f2,y_true=y,probs=np.column_stack([1-q,q]),classes=np.array([0,1]),test_row_index=np.arange(60))
    z=paired_bootstrap(f1,f2,n=100)
    assert z['n_patients']==60 and len(z['patient_bootstrap_ci'])==2
    np.savez_compressed(f2,y_true=y,probs=np.column_stack([1-q,q]),classes=np.array([0,1]),test_row_index=np.arange(60)[::-1])
    with pytest.raises(ValueError,match='exactly the same'):paired_bootstrap(f1,f2,n=100)


def test_evidence_gate_does_not_claim_unrun_experiments(tmp_path):
    r=gate(tmp_path,[],tmp_path)
    assert set(r['experiments'])=={f'E{i}' for i in range(1,13)}
    assert all(t['status']=='not_measured' for t in r['experiments'].values())
    assert len(r['local_missing_inputs']['frozen_manifest_pairs'])==16


def test_imbalance_stress_uses_context_only_and_retains_two_minority():
    from scripts.experiments.run_predictions import thin_minority_context
    ids=np.arange(30);y=np.array([0]*20+[1]*10)
    selected=thin_minority_context(ids,y,.5,42)
    assert np.sum(y[selected]==1)==5
    assert np.sum(y[selected]==0)==20
    assert np.array_equal(selected,thin_minority_context(ids,y,.5,42))
