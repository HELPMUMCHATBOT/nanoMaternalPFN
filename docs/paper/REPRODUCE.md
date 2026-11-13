# P1-P15: reproduce the frozen nanoMaternalPFN paper evidence

## Evidence and scope

This branch contains *derived* numerical tables, vector figures, audit scripts and an evidence-linked manuscript. It **does not** train a neural model, change the previous eight real-data tasks, create patient-level predictions, or re-run an AWS benchmark.

The **sole measurement input** is the original 256-member `nanoMaternalPFN-paper_review_bundle.zip`, uploaded on 13 November 2026, SHA-256:

`f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26`

The raw zip is deliberately **not** committed to GitHub. Obtain it from the original research archive; do not substitute a terminal transcript, another ZIP with an identical name, or a newly generated synthetic benchmark.

## Reproduction with existing real measurements

Run this on a machine with the frozen ZIP and Python 3.11+:

```bash
cd nanoMaternalPFN
python -m pip install numpy pandas matplotlib scipy openpyxl pytest
python scripts/paper/run_paper_evidence.py \
  --bundle /absolute/path/to/nanoMaternalPFN-paper_review_bundle.zip \
  --output artifacts/results/paper_p1_p15
python -m pytest -q tests/test_paper_evidence_pipeline.py
```

The command **fails closed** on the wrong ZIP hash, missing evidence members, changed byte sizes, invalid per-member SHA-256, bad CRC, mismatch between draw-level and summary data, or missing generated files. The default frozen hash is deliberate. `--allow-new-evidence` relaxes only the outer exact-ZIP hash, not the internal manifest checks; it does not turn an alternative dataset into a pre-specified benchmark.

Outputs include all-context and low-context (16/32/64) method leaderboards; dataset/context comparisons; 13-model, 17-configuration historical audit; v1-versus-v2 development tables; all four performance metrics; exact paired model comparisons; seven-source uncertainty sensitivity analyses; measured resource profiles; a 13-tab statistical workbook; tables and vector figures; and `MANUSCRIPT.md`.

For a compact version-controlled snapshot see `tables/` and `figures/` here. Generated results include `FROZEN_EVIDENCE_PROVENANCE.json` with the verified source hash and analysis status.

## Primary arithmetic

For each method and metric: average the valid context draws within `(dataset, context)`, average eligible context means **within each task**, and average eight tasks with equal weight. For paired contrasts, join identical `(dataset, context, seed)` draw keys before computing directional differences; retain both VAT tasks but treat them as **one source cohort** when resampling source groups. No missing draw is imputed.

The original 1,632-row summary covers 12 current methods; the separately archived 136-row three-expert summary adds a thirteenth evaluated method. The frozen ZIP also contains four earlier v1 arms. Current models share 671 valid paired draws across 34 task-context cells. The original source-level cluster sensitivity has only **seven** heterogeneous clusters. Its intervals and exploratory p-values do not establish external superiority.

## Source/measurement audit

- Source-cohort linkage: `docs/DATASET_PORTFOLIO.md` (VAT GDM/delivery overlap, UCI risk proxy, non-maternal Pima).
- Benchmark definition and versioning: `docs/PAPER_COMPLETION.md`, `docs/COMPLETE_AWS_PAPER_BENCHMARK.md`, `src/nanomaternalpfn/evaluation/baselines.py`.
- Standalone ONNX: measured v2 Tiny maternal-mixture graph. Do not combine its size/latency with adaptive ensemble AUROC.
- Adaptive multi-expert: distinct PyTorch CPU measured checkpoint combination, context CV fit and query time; no isolated ensemble total RSS is present.
- Brier is multiclass sum-of-squared-probability errors; ECE15 is 15-bin confidence-based ECE, **not** a calibration curve.
- No original patient-level prediction probabilities or outcomes are included in this ZIP. Consequently no patient-level ROC covariance (DeLong), independent-patient calibration curves, subgroup fairness, prospective utility, Android energy or true INT8 runtime is calculated.

## Files and provenance

The scripts are versioned, so their Git commit SHA can be linked to the fixed evidence ZIP SHA-256. The model weights themselves are not added here; preserved local original checkpoints and configurations are required to *re-execute the predictive benchmark*, which is outside the scope of this statistics rerun. Do not interpret this as checkpoint-weight provenance verification.

Dependencies deliberately do not include GPU, AWS credentials, TabPFN checkpoint downloads, or new clinical datasets. The manuscript contains placeholders for authors, institutional ethics determination, funding and conflict-of-interest declarations pending their real sign-off. Do not complete those fields by inference.
