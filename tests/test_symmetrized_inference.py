"""Properties of class symmetrization, optional calibration and CPU batching."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig
from nanomaternalpfn.symmetrized_classifier import (
    SymmetrizedNanoMaternalPFNClassifier,
    context_prior_adjustment,
)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("n_features", [4, 7])
def test_complete_class_symmetry_with_fixed_feature_permutations(
    tmp_path, n_classes, n_features
):
    torch.set_num_threads(1)
    cfg = ModelConfig(d_model=24, n_heads=4, n_layers=2, mlp_hidden=48)
    torch.manual_seed(15)
    net = NanoMaternalPFN(cfg).eval()
    ckpt = tmp_path / "weights.pt"
    torch.save({"config": asdict(cfg), "state_dict": net.state_dict()}, ckpt)
    rng = np.random.default_rng(24)
    X = rng.normal(size=(24, n_features)).astype("float32")
    X[4, 1] = np.nan
    context_y = np.tile(np.arange(n_classes), 12)[:12]
    clf = SymmetrizedNanoMaternalPFNClassifier(
        ckpt, n_feature_permutations=2, query_batch_size=3, feature_seed=42
    )
    clf.fit(X[:12], context_y)
    p = clf.predict_proba(X[12:])
    assert p.shape == (12, n_classes)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(p).all()
    assert (p >= 0).all()
    assert (p <= 1).all()
    loaded_model = clf.model
    for shift in range(1, n_classes):
        mapping = (np.arange(n_classes) + shift) % n_classes
        clf.fit(X[:12], mapping[context_y])
        q = clf.predict_proba(X[12:])
        np.testing.assert_allclose(q[:, mapping], p, atol=2e-6, rtol=2e-6)
        assert clf.model is loaded_model


def test_context_prior_adjustment_is_explicitly_opt_in():
    p = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    np.testing.assert_array_equal(
        context_prior_adjustment(p, [0, 1], alpha=0), p
    )
    adjusted = context_prior_adjustment(p, [0, 1], alpha=0.5)
    assert np.isfinite(adjusted).all()
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        context_prior_adjustment(p, [0, 1], alpha=-0.2)
