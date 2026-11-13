"""Software contract checks only: no synthetic clinical observations or smoke models."""
import json
from pathlib import Path
import pytest
from scripts.experiments.training_matrix import make_matrix
from scripts.experiments.run_complete_study import check_existing,trained_groups,full_training
from scripts.experiments.verify_campaign import verify_all


def test_all_training_arms_are_actual_full_step_budgets():
    matrix=make_matrix()
    assert len(matrix)==11
    assert sum(row['factors']['steps'] for row in matrix)==70000
    assert min(row['factors']['steps'] for row in matrix)>=5000
    assert max(row['factors']['steps'] for row in matrix)==20000
    assert not any('smoke' in row['id'].lower() for row in matrix)


def test_preflight_requires_original_eight_real_sources_and_checkpoints(tmp_path):
    result=check_existing(tmp_path,tmp_path,tmp_path)
    assert result['ready'] is False
    assert len(result['missing'])==11  # eight actual source pairs + three final model files
    assert result['frozen_sources']=={}
    assert not any(tmp_path.rglob('*.npz')) # Never creates placeholder clinical tables.


def test_controlled_stage_refuses_to_call_untrained_checkpoints_complete(tmp_path):
    groups,missing=trained_groups(tmp_path,False)
    assert len(missing)==11
    assert all(not groups[phase] for phase in ('E2','E7','E11'))
    with pytest.raises(RuntimeError,match='fully executed'):
        trained_groups(tmp_path,True)


def test_evidence_verifier_never_marks_empty_experiments_measured(tmp_path):
    result=verify_all(tmp_path/'out',tmp_path/'no_real_sources')
    for code in ('E1','E2','E3','E4','E5','E6','E7','E8','E10','E11'):
        assert result['observed_evidence_status'][code]['status']=='not_executed'
        assert not (tmp_path/'out'/code/'COMPLETE.json').is_file()


def test_training_requires_explicit_compute_acceptance(tmp_path):
    with pytest.raises(RuntimeError,match='confirm-compute'):
        full_training(tmp_path,'cuda',confirmed=False)
