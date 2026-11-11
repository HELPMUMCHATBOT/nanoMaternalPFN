"""v2 architectural, prior, optimizer, inference and isolation regression tests."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch

from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig
from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
from nanomaternalpfn.priors.maternal_scm import sample_task
from nanomaternalpfn.priors.hdf5_replay import create_replay, HDF5TaskReplay
from nanomaternalpfn.train_replay import train_replay

torch.set_num_threads(1)


def test_preln_parameter_count_and_feature_permutation():
    v1 = NanoMaternalPFN(ModelConfig(24, 4, 2, 48)).eval()
    v2 = NanoMaternalPFN(ModelConfig(24, 4, 2, 48, architecture="pre_ln_v2")).eval()
    assert v1.parameter_count() == v2.parameter_count()
    x = torch.randn(1, 8, 6)
    x[0, 1, 4] = float('nan')
    labels = torch.tensor([[0, 1, 0, 1, 0]])
    with torch.inference_mode():
        original = v2(x, labels, n_classes=2)
        swapped = v2(x[:, :, [4, 0, 5, 2, 1, 3]], labels, n_classes=2)
    torch.testing.assert_close(original, swapped, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("prior", ["generic", "mixture"])
def test_v2_prior_augmentation_changes_labels_without_changing_task_shape(prior):
    # Compare same deterministic seed to original ordinal generation;
    # label permutation and feature sign flips never alter row count.
    x, y, metadata = sample_task(
        48, 8, 3, prior=prior, seed=831,
        augment_class_permutation=True, augment_feature_sign_flip=True,
    )
    assert x.shape == (48, 8)
    assert np.array_equal(np.sort(np.unique(y)), np.array([0, 1, 2]))
    assert len(metadata["class_mapping_original_to_augmented"]) == 3
    assert sorted(metadata["class_mapping_original_to_augmented"]) == [0, 1, 2]


def test_v2_replay_versioning_and_safe_training(tmp_path):
    train_file = tmp_path / "train_v2.h5"
    validation_file = tmp_path / "val_v2.h5"
    create_replay(train_file, num_tables=24, seed=30, variant="v2",
                  min_rows=24, max_rows=40, max_features=8)
    create_replay(validation_file, num_tables=24, seed=31, variant="v2",
                  min_rows=24, max_rows=40, max_features=8)
    with HDF5TaskReplay(train_file) as replay:
        assert replay.settings["variant"] == "v2"
        assert replay.settings["class_permutation_augmented"]
        assert replay.settings["sign_flip_task_probability"] == 0.5
        x, y, _, _ = replay.sample_batch(np.random.default_rng(30), 2, (16,), query_rows=4)
        assert x.shape[0] == 2 and y.shape[0] == 2
    out = tmp_path / "base_v2.pt"
    result = train_replay(
        train_file, out, tier="base", variant="v2", steps=1,
        warmup_steps=2, batch_size=2, microbatch_size=1,
        max_context=16, validation_path=validation_file,
        validate_every=1, validation_batches_per_context=1,
        validation_batch_size=2, save_every=1,
    )
    assert result["variant"] == "v2"
    assert result["peak_learning_rate"] == pytest.approx(2e-4)
    best = out.with_name("base_v2.best.inference.pt")
    checkpoint = torch.load(best, weights_only=True, map_location="cpu")
    assert checkpoint["config"]["architecture"] == "pre_ln_v2"
    model = NanoMaternalPFN(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with pytest.raises(ValueError, match="different model variant"):
        train_replay(train_file, out, variant="v1", tier="base",
                     resume=True, max_context=16, batch_size=2,
                     microbatch_size=1, steps=2)
    with pytest.raises(ValueError, match="variant"):
        train_replay(train_file, tmp_path / "bad.pt", variant="v1",
                     tier="base", max_context=16, steps=1,
                     batch_size=2, microbatch_size=1)


def test_class_symmetrization_is_equivariant_and_single_query(tmp_path):
    cfg = ModelConfig(24, 4, 2, 48, architecture="pre_ln_v2")
    torch.manual_seed(9)
    model = NanoMaternalPFN(cfg).eval()
    ckpt = tmp_path / "inference.pt"
    torch.save({"config": asdict(cfg), "state_dict":model.state_dict(),
                "variant":"v2"}, ckpt)
    rng = np.random.default_rng(17)
    X = rng.standard_normal((21, 6)).astype("float32")
    X[3, 2] = np.nan
    y = np.array([0, 1, 2] * 4, dtype=np.int64)
    clf = NanoMaternalPFNClassifier(
        ckpt, query_batch_size=2, symmetrize_classes=True,
        calibrate_context=True, calibration_folds=3,
    ).fit(X[:12], y)
    assert np.isfinite(clf.temperature_) and clf.temperature_ > 0
    p = clf.predict_proba(X[12:13])
    assert p.shape == (1, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    mapping = np.array([2, 0, 1])
    q = clf.fit(X[:12], mapping[y]).predict_proba(X[12:13])
    # Fully class-symmetrized logits are equivariant; context-only CV should
    # also be equivariant up to numeric optimizer tolerance.
    np.testing.assert_allclose(q[:, mapping], p, atol=1e-3, rtol=1e-3)


def test_v1_checkpoint_loads_under_v2_repository(tmp_path):
    old = NanoMaternalPFN(ModelConfig(24, 4, 2, 48)).eval()
    config = asdict(old.cfg)
    config.pop("architecture")  # old serialized checkpoints did not record it
    path = tmp_path / "old.pt"
    torch.save({"config": config, "state_dict": old.state_dict()}, path)
    rng = np.random.default_rng(13)
    X = rng.normal(size=(16, 5))
    y = np.array([0,1]*6)
    clf = NanoMaternalPFNClassifier(path).fit(X[:12], y)
    assert clf.model.cfg.architecture == "post_ln_v1"
    assert clf.predict_proba(X[12:]).shape == (4, 2)
