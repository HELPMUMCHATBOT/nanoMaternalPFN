"""New paper-comparison methods: no test leakage, explicit TabPFN version, resume."""
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
import numpy as np
import pytest

from nanomaternalpfn.adaptive_classifier import AdaptiveNanoMaternalPFNClassifier
from nanomaternalpfn.evaluation.baselines import get_baseline
from nanomaternalpfn.evaluation.transfer_curves import evaluate


class FakeExpert:
    def __init__(self, name):
        self.name = name
        self.fit_indices = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fit_indices.append(np.asarray(X)[:,0].copy())
        return self

    def predict_proba(self, X):
        sign = 1 if self.name == "a" else -1
        logits = np.asarray(X)[:,1] * sign
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1-p,p])


def test_adaptive_uses_only_context_data(tmp_path):
    X = np.column_stack([np.arange(20), np.linspace(-2,2,20)]).astype(np.float32)
    y = np.array([0,1]*10)
    made=[]
    def factory(path):
        expert=FakeExpert(path.stem)
        made.append(expert)
        return expert
    clf=AdaptiveNanoMaternalPFNClassifier(
        [tmp_path/"a.pt",tmp_path/"b.pt"],estimator_factory=factory,
    ).fit(X[:16],y[:16])
    assert clf.cv_status_=="cross_fitted"
    assert clf.context_cv_folds_>=2
    assert len(made)==2
    assert all(np.concatenate(e.fit_indices).max()<16 for e in made)
    np.testing.assert_allclose(clf.weights_.sum(),1.0)
    p=clf.predict_proba(X[16:])
    assert p.shape==(4,2)
    np.testing.assert_allclose(p.sum(axis=1),1.0,atol=1e-6)
    assert clf.diagnostics_["context_only"]


def test_adaptive_rare_positive_fallback(tmp_path):
    X=np.column_stack([np.arange(16),np.linspace(0,1,16)]).astype(np.float32)
    y=np.zeros(16,dtype=int)
    y[0]=1
    clf=AdaptiveNanoMaternalPFNClassifier(
        [tmp_path/"a.pt",tmp_path/"b.pt"],
        estimator_factory=lambda path:FakeExpert(path.stem),
    ).fit(X,y)
    assert clf.cv_status_=="insufficient_minority_labels"
    assert clf.temperature_==1.0
    np.testing.assert_allclose(clf.weights_,[.5,.5])


def test_official_tabpfn_is_explicit_v2(monkeypatch):
    calls=[]
    tabpfn=ModuleType("tabpfn")
    constants=ModuleType("tabpfn.constants")
    class Versions:
        V2="v2-verified"
    class MockTabPFN:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.append((version,kwargs))
            return object()
    tabpfn.TabPFNClassifier=MockTabPFN
    constants.ModelVersion=Versions
    monkeypatch.setitem(sys.modules,"tabpfn",tabpfn)
    monkeypatch.setitem(sys.modules,"tabpfn.constants",constants)
    get_baseline("tabpfn_v2_single",seed=2)
    get_baseline("tabpfn_v2_default",seed=3)
    assert [c[0] for c in calls]==["v2-verified","v2-verified"]
    assert [c[1]["n_estimators"] for c in calls]==[1,"auto"]


def test_resume_no_duplicate_draws(tmp_path):
    rng=np.random.default_rng(11)
    X=rng.normal(size=(36,5)).astype(np.float32)
    y=np.array([0,1]*18)
    npz=tmp_path/"toy.npz"
    np.savez(npz,X=X,y=y)
    manifest={
        "sha256":hashlib.sha256(npz.read_bytes()).hexdigest(),
        "test_idx":list(range(24,36)),
        "context_draws":{"16":[list(range(16)),list(range(2,18))]},
    }
    path=tmp_path/"toy.json"
    path.write_text(json.dumps(manifest))
    output=tmp_path/"result.csv"
    first=evaluate(path,methods=("logistic",),output=output,max_draws=1)
    second=evaluate(path,methods=("logistic",),output=output,
                    max_draws=2,resume=True)
    third=evaluate(path,methods=("logistic",),output=output,
                   max_draws=2,resume=True)
    assert [len(first),len(second),len(third)]==[1,2,2]
    assert len(output.read_text().splitlines())==3
