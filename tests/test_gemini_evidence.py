"""No synthetic fixture can be mislabeled as clinical evidence."""
import json
from pathlib import Path

import pandas as pd
import pytest

from importlib.util import module_from_spec, spec_from_file_location


def load_script(name):
    path=Path(__file__).resolve().parents[1]/"scripts"/name
    spec=spec_from_file_location(name.replace(".py",""),path)
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundler_rejects_absent_measured_artifacts(tmp_path):
    bundler=load_script("24_package_paper_evidence.py")
    with pytest.raises(FileNotFoundError):
        bundler.assemble(tmp_path,tmp_path/"no_evidence.zip")


def test_reporter_rejects_incomplete_benchmark(tmp_path):
    reporter=load_script("23_build_verified_manuscript.py")
    path=tmp_path/"summary.csv"
    pd.DataFrame([{"dataset":"uci_maternal_risk","method":"adaptive_tiny",
                   "context":16,"metric":"auroc","mean":0.7,
                   "n_valid_draws":20}]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="Required measured methods missing"):
        reporter.load_summary(path)


def test_reporter_rejects_unmeasured_aggregate(tmp_path):
    reporter=load_script("23_build_verified_manuscript.py")
    path=tmp_path/"summary.csv"
    pd.DataFrame([{"dataset":"uci_maternal_risk","method":"adaptive_tiny",
                   "context":16,"metric":"auroc","mean":float("nan"),
                   "n_valid_draws":0}]).to_csv(path,index=False)
    with pytest.raises(ValueError):
        reporter.load_summary(path,require_methods=[])


def test_verified_archive_contains_sha256_for_real_input_files(tmp_path):
    bundler=load_script("24_package_paper_evidence.py")
    base=tmp_path/"results"
    for name in bundler.REQUIRED:
        dest=base/name
        if name.endswith("profile.json") or name.endswith("memory.json"):
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text("{}")
        elif dest.suffix==".csv":
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text("metric,mean\nauroc,0.7\n")
        else:
            dest.mkdir(parents=True,exist_ok=True)
            (dest/"example.csv").write_text("metric,mean\nauroc,0.7\n")
    report=bundler.assemble(base,base/"archive.zip")
    assert len(report["sha256"]) == 64
    import zipfile
    with zipfile.ZipFile(base/"archive.zip") as z:
        manifest=json.loads(z.read("EVIDENCE_SHA256_MANIFEST.json"))
        assert manifest["status"].startswith("VERIFIED_ARCHIVE")
        assert len(manifest["files"]) >= len(bundler.REQUIRED)
