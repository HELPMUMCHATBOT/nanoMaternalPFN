#!/usr/bin/env python3
"""Run the complete P1-P15 evidence audit on a frozen, real results ZIP.

No benchmark is re-executed; no metric, p-value, effect or figure is simulated.
By default, fail closed if the review bundle differs from the audited freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

FROZEN_SHA256 = 'f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26'
EXPECTED_FILES = 256
REQUIRED = (
    'EVIDENCE_SHA256_MANIFEST.json',
    'results/paper_all_methods_summary.csv',
    'results/downstream_full/summary.csv',
    'results/downstream_v2_full/summary.csv',
    'results/paper_completion_full/summary.csv',
    'results/paper_three_expert_full/summary.csv',
    'results/edge_paper_comparison/profile.json',
    'results/adaptive_cpu_profile.json',
    'results/paper_onnx_memory.json',
)
SCRIPTS = (
    'nanoMaternalPFN_full_statistics.py',
    'nanoMaternalPFN_make_reports.py',
    'nanoMaternalPFN_cache_formulas.py',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(path: Path, *, locked: bool = True) -> dict:
    """Reject missing/changed/unchecked bundle members; return immutable provenance."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing source results ZIP: {path}')
    digest = sha256(path)
    if locked and digest != FROZEN_SHA256:
        raise ValueError(f'Wrong frozen ZIP SHA-256: {digest}')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member names')
        missing = set(REQUIRED) - set(names)
        if missing:
            raise ValueError(f'Missing required evidence: {sorted(missing)}')
        broken = z.testzip()
        if broken:
            raise ValueError(f'Corrupt ZIP member: {broken}')
        manifest = json.loads(z.read('EVIDENCE_SHA256_MANIFEST.json'))
        entries = manifest.get('files', [])
        if locked and len(entries) != EXPECTED_FILES:
            raise ValueError(f'Expected {EXPECTED_FILES} manifest entries, got {len(entries)}')
        if {v['path'] for v in entries} != (set(names) - {'EVIDENCE_SHA256_MANIFEST.json'}):
            raise ValueError('Manifest paths do not equal data members')
        for e in entries:
            data = z.read(e['path'])
            if len(data) != e['bytes'] or hashlib.sha256(data).hexdigest() != e['sha256']:
                raise ValueError(f'ZIP member size/hash mismatch: {e["path"]}')
    return {'zip_sha256': digest, 'verified_entries': len(entries),
            'locked_freeze': locked, 'has_patient_level_predictions': False}


def run_pipeline(bundle: Path, output: Path, *, locked: bool = True) -> dict:
    verified = verify_bundle(bundle, locked=locked)
    output.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    env = dict(os.environ,
               NANO_EVIDENCE_ZIP=str(bundle.resolve()),
               NANO_STATS_OUTPUT=str(output.resolve()))
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script_dir / script)],
                       check=True, env=env)
    subprocess.run([sys.executable, str(script_dir / 'write_manuscript.py'),
                    '--source',str(output),'--output',str(output/'MANUSCRIPT.md')],check=True)
    subprocess.run([sys.executable, str(script_dir / 'make_svg_figures.py'),
                    '--source',str(output),'--output',str(output/'figures')],check=True)
    subprocess.run([sys.executable, str(script_dir / 'make_architecture_figure.py'),
                    '--output', str(output/'figures/architecture_source.svg')],check=True)
    audit = json.loads((output / 'data_audit.json').read_text())
    if audit['input_zip_sha256'] != verified['zip_sha256']:
        raise RuntimeError('Input ZIP changed during analysis')
    if audit['raw_to_summary_n_mismatches'] or not audit['manifest_sha256_verified']:
        raise RuntimeError('Evidence audit reports discrepancy')
    required_outputs = ('leaderboard_all.csv', 'leaderboard_low_data.csv',
                        'three_expert_contrasts_all_metrics.csv',
                        'v1_vs_v2_development.csv', 'efficiency_measured.csv',
                        'onnx_memory_measured.csv', 'primary_by_context.csv',
                        'task_auroc_all_models.csv', 'nanoMaternalPFN_Full_Statistical_Report.md',
                        'nanoMaternalPFN_Statistical_Results.xlsx',
                        'full_auroc_leaderboard.png', 'task_auroc_effects.png',
                        'cohort_auroc_effects.png', 'MANUSCRIPT.md',
                        'figures/full_auroc_leaderboard.svg',
                        'figures/task_auroc_effects.svg',
                        'figures/cohort_auroc_effects.svg',
                        'figures/architecture_source.svg')
    missing = [p for p in required_outputs if not (output / p).is_file()]
    if missing:
        raise RuntimeError('Analysis did not produce: ' + ', '.join(missing))
    (output / 'FROZEN_EVIDENCE_PROVENANCE.json').write_text(
        json.dumps({**verified, 'analysis':'retrospective exploratory',
                    'primary_inference':'seven-source clustered sensitivity analysis; not confirmatory',
                    'not_measured':['patient-level DeLong','subgroup metrics',
                                    'Android','INT8 energy','adaptive ensemble RSS'],
                    'software': {'python':sys.version.split()[0]}}, indent=2) + '\n')
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True,
                    help='Original nanoMaternalPFN-paper_review_bundle.zip (not a transcript)')
    ap.add_argument('--output', type=Path, default=Path('artifacts/results/paper_p1_p15'))
    ap.add_argument('--allow-new-evidence', action='store_true',
                    help='Opt out of the exact frozen-ZIP hash check, not out of manifest validation')
    args = ap.parse_args()
    result = run_pipeline(args.bundle, args.output, locked=not args.allow_new_evidence)
    print(json.dumps({k:result[k] for k in ('input_zip_sha256','master_summary_rows',
                                           'all_raw_draw_rows_including_v1',
                                           'raw_to_summary_n_mismatches')},indent=2))

if __name__ == '__main__':
    main()
