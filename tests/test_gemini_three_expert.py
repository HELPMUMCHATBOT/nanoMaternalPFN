"""Behavioral tests for optional three-checkpoint context-only adaptation."""
import numpy as np
import pytest

from nanomaternalpfn.multi_expert_classifier import ContextOnlyMultiExpertPFN


class FakeExpert:
    def __init__(self, name):
        self.bias = { "v1": 0.25, "v2": 0.65, "generic": 0.8 }[name]
        self.fit_contexts = []

    def fit(self, X, y):
        self.fit_contexts.append(np.array(X, copy=True))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        x = np.asarray(X)
        p = np.clip(self.bias + 0.02 * np.tanh(x[:, 0]), 0.01, 0.99)
        return np.column_stack([1-p,p])


def test_three_experts_weight_only_labeled_context():
    X = np.arange(36, dtype=np.float32).reshape(12,3) / 10
    y = np.array([0,1] * 6)
    experts = {}
    def factory(path):
        name = str(path)
        experts[name] = FakeExpert(name)
        return experts[name]
    model = ContextOnlyMultiExpertPFN(
        ["v1","v2","generic"], estimator_factory=factory,
        cv_folds=3, calibrate=True,
    ).fit(X,y)
    assert model.diagnostics_["context_only"] is True
    assert model.cv_status_ == "cross_fitted"
    assert model.context_cv_folds_ == 3
    assert len(model.weights_) == 3
    assert np.all(model.weights_ >= 0)
    assert np.isclose(model.weights_.sum(), 1)
    for expert in experts.values():
        assert np.array_equal(expert.fit_contexts[-1], X)
        assert all(len(ctx) <= len(X) for ctx in expert.fit_contexts)
    # Unlabeled queries can be changed after fit without changing CV weights.
    before = model.weights_.copy()
    p = model.predict_proba(np.array([[999.,1,0],[-42.,0,2]], dtype=np.float32))
    assert p.shape == (2,2)
    assert np.allclose(p.sum(axis=1),1)
    assert np.array_equal(model.weights_, before)


def test_minority_one_defaults_to_uniform_weights():
    X = np.ones((6,2), dtype=np.float32)
    y = np.array([0,0,0,0,0,1])
    model = ContextOnlyMultiExpertPFN(
        ["v1","v2","generic"], estimator_factory=lambda x: FakeExpert(str(x)),
    ).fit(X,y)
    assert model.cv_status_ == "insufficient_minority_labels"
    assert np.allclose(model.weights_, [1/3,1/3,1/3])
    assert model.temperature_ == 1.0


def test_invalid_and_unfitted_rejected():
    with pytest.raises(ValueError):
        ContextOnlyMultiExpertPFN(["v1"])
    model = ContextOnlyMultiExpertPFN(
        ["v1","v2"], estimator_factory=lambda x: FakeExpert(str(x)),
    )
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((1,2), dtype=np.float32))
    with pytest.raises(ValueError):
        model.fit(np.ones((3,2)), np.zeros(3))
