"""Smoke tests for chunked frozen-PFN inference and real-split metrics."""
import csv
import numpy as np
import torch
from dataclasses import asdict

from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig
from nanomaternalpfn.data.freeze_splits import freeze
from nanomaternalpfn.evaluation.transfer_curves import evaluate


def test_chunked_and_unchunked_inference_equivalent(tmp_path):
    torch.set_num_threads(1)
    config = ModelConfig(d_model=24, n_heads=4, n_layers=2, mlp_hidden=48)
    torch.manual_seed(8)
    model = NanoMaternalPFN(config).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)
    rng = np.random.default_rng(14)
    X = rng.normal(size=(41, 6)).astype("float32")
    y = rng.integers(0, 3, size=24)
    y[:3] = np.arange(3)
    clf = NanoMaternalPFNClassifier(path, query_batch_size=3).fit(X[:24], y)
    chunks = clf.predict_proba(X[24:])
    clf.query_batch_size = None
    together = clf.predict_proba(X[24:])
    assert chunks.shape == (17, 3)
    np.testing.assert_allclose(chunks, together, atol=1e-5)
    first_model = clf.model
    clf.fit(X[:24], y)
    assert clf.model is first_model


def test_real_split_evaluation_writes_metrics(tmp_path):
    torch.set_num_threads(1)
    config = ModelConfig(d_model=24, n_heads=4, n_layers=2, mlp_hidden=48)
    model = NanoMaternalPFN(config)
    ckpt = tmp_path / "weights.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, ckpt)
    rng = np.random.default_rng(71)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] > 0).astype(int)
    frozen = tmp_path / "real.json"
    freeze(X, y, frozen, budgets=(16, 32), n_seeds=2)
    out = tmp_path / "scores.csv"
    rows = evaluate(frozen, checkpoint=ckpt, methods=("pfn", "logistic"),
                    output=out, max_draws=1, contexts=(16,), query_batch_size=4)
    assert len(rows) == 2
    assert all(x["context"] == 16 for x in rows)
    assert all(0 <= x["brier"] <= 2 for x in rows)
    with out.open() as f:
        assert len(list(csv.DictReader(f))) == 2
