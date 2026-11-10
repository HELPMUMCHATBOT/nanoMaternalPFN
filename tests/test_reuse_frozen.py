"""Regression tests: retry failed downloads without modifying verified holdouts."""
import json
from hashlib import sha256
import numpy as np

from nanomaternalpfn.data import open_sources


def test_reuse_verified_frozen_split_without_network_or_overwrite(tmp_path, monkeypatch):
    name = "tpehg_preterm"
    manifest_path = tmp_path / (name + ".json")
    npz_path = tmp_path / (name + ".npz")
    np.savez_compressed(npz_path, X=np.ones((4, 2)), y=np.array([0, 1, 0, 1]))
    digest = sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({"sha256": digest}))
    report_path = tmp_path / "open_portfolio_report.json"
    report_path.write_text(json.dumps({
        "seed": 20261107,
        "dataset_results": {
            name: {"status": "frozen", "split_sha256": digest,
                   "source": {"rows": 4, "features": ["x", "z"]}},
            "vat_gdm": {"status": "unavailable", "reason": "temporary SSL error"},
        },
    }))
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Frozen dataset was reacquired")
    monkeypatch.setattr(open_sources, "get_open_task", unexpected_call)
    before = npz_path.read_bytes()
    result = open_sources.acquire_and_freeze(
        [name], out_dir=tmp_path, seed=20261107, reuse_frozen=True
    )
    assert result["dataset_results"][name]["status"] == "frozen"
    assert result["dataset_results"]["vat_gdm"]["status"] == "unavailable"
    assert before == npz_path.read_bytes()


def test_reuse_frozen_rejects_seed_change(tmp_path):
    path = tmp_path / "open_portfolio_report.json"
    path.write_text(json.dumps({"seed": 1, "dataset_results": {}}))
    import pytest
    with pytest.raises(ValueError, match="different seed"):
        open_sources.acquire_and_freeze(
            [], out_dir=tmp_path, seed=2, reuse_frozen=True
        )
