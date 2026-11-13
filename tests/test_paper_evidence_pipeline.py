"""Input-integrity and measured-evidence regression checks. No fake patient data."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/paper/run_paper_evidence.py'
spec = importlib.util.spec_from_file_location('run_paper_evidence', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_zip(path, *, tamper=False):
    payloads = {p:b'test' for p in module.REQUIRED if p != 'EVIDENCE_SHA256_MANIFEST.json'}
    manifest = {'files':[{'path':p,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                         for p,data in payloads.items()]}
    if tamper:
        payloads[next(iter(payloads))] = b'tampered'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('EVIDENCE_SHA256_MANIFEST.json',json.dumps(manifest))
        for p, data in payloads.items(): z.writestr(p,data)


def test_missing_zip_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.verify_bundle(tmp_path/'missing.zip')


def test_verified_small_fixture_only_when_unlocked(tmp_path):
    p=tmp_path/'case.zip';make_zip(p)
    assert module.verify_bundle(p,locked=False)['verified_entries']==len(module.REQUIRED)-1
    with pytest.raises(ValueError,match='Wrong frozen ZIP SHA-256'):
        module.verify_bundle(p,locked=True)


def test_tampered_payload_fails(tmp_path):
    p=tmp_path/'wrong.zip'; make_zip(p,tamper=True)
    with pytest.raises(ValueError,match='ZIP member size/hash mismatch'):
        module.verify_bundle(p,locked=False)


def test_expected_measured_tables_present_and_identical_units():
    paper=ROOT/'docs/paper/tables'
    audit=json.loads((paper/'data_audit.json').read_text())
    assert audit['master_summary_rows']==1632
    assert audit['raw_draw_rows_per_current_model']==671
    assert audit['dataset_tasks']==8 and audit['independent_source_cohorts_assumed']==7
    assert audit['patient_level_predictions_in_zip'] is False
    with (paper/'leaderboard_all.csv').open() as f:
        rows={r['method']:r for r in csv.DictReader(f)}
    assert len(rows)==13
    assert float(rows['adaptive_tiny_3expert']['auroc']) == pytest.approx(0.707268179131,abs=1e-10)
    assert float(rows['tabpfn_v2_default']['auroc']) == pytest.approx(0.699226298527,abs=1e-10)
    assert float(rows['tabpfn_v2_default']['brier']) < float(rows['adaptive_tiny_3expert']['brier'])
    with (paper/'three_expert_contrasts_all_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==48
    auroc=next(r for r in rows if r['comparator']=='tabpfn_v2_default' and r['metric']=='auroc')
    assert float(auroc['equal_task_effect']) == pytest.approx(0.008041880604,abs=1e-8)
    assert float(auroc['cohort_boot_ci_lower']) < 0 < float(auroc['cohort_boot_ci_upper'])
    assert float(auroc['exact_source_signflip_p_two_sided']) == pytest.approx(0.578125)
