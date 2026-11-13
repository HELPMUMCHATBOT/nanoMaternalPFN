# Gemini G1-G20 implementation: measured-only paper branch

**Branch:** `research/gemini-g1-g20-verified-paper-20261113`
**Paper title:** nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction

This is an opt-in experimental and reporting branch. Existing model weights,
frozen clinical datasets, and original evaluation outputs are never overwritten.
**Code committed to GitHub does not mean the user's Mac-only evaluation files
have been executed, examined, uploaded, or independently validated.**

## Status for every Gemini recommendation

| ID | Implementation | Evidence status |
| --- | --- | --- |
| G1 | Keep v2 and original Adaptive unchanged; opt-in experiments in separate outputs | Frozen outputs preserved; checkpoint SHA recorded only if files are local |
| G2 | Existing eight-task frozen manifests, seven cohorts including shared VAT pair | No ninth cohort required for *this exploratory paper* |
| G3 | Generate per-task/context performance tables plus measured size/latency comparisons | No formal Pareto claim until matched predictors have matching hardware metrics |
| G4 | Resource-constrained clinical PFN positioning in generated manuscript | Research only; no actual CDSS deployment claims |
| G5 | Separate contexts 16/32/64 in low-data tables | Computed from local summary |
| G6 | CTU-specific AUROC-by-context table | Computed, never inferred from aggregate |
| G7 | All 12 legacy methods, including LR/RF/XGB/LGB/TabPFN | Requires complete measured CSV |
| G8 | Explain LightGBM default min_child_samples=20 limitations | Explicit manuscript disclosure, not unfair blanket boosted-tree claim |
| G9 | Keep standalone ONNX and adaptive expert measurements separate | New adaptive CPU profiler records actual 2- and 3-expert cost |
| G10 | Preserve original v1, v2, Adaptive; import v1 per-draw results when present | Chronological comparison; not a causal one-change ablation |
| G11 | Benchmark reuse/selection disclosed as exploratory | Included in generated paper |
| G12 | Three-expert Tiny: v1 mixture + v2 mixture + v2 generic | Implemented, opt-in, never assigned an invented AUROC |
| G13 | Labeled-context-only stratified CV weights and optional temperature scaling | Implemented and tested; query labels never used |
| G14 | Full frozen evaluation vs 0.699 comparator | Only an executed result can establish an improvement; 0.705 is NOT a forecast presented as fact |
| G15 | Complete 12-method AUROC/Brier/ECE15/accuracy tables | `leaderboard_all.csv`, `leaderboard_low_data.csv` |
| G16 | Per-task low-data and CTU tables | `low_data_auroc_by_dataset.csv`, `ctu_auroc_by_context.csv` |
| G17 | Existing EdgeBench JSON and new adaptive CPU timing | Actual median/p99 and checkpoint-byte totals recorded, no simulated clinical scores |
| G18 | Deterministic bundle manifest and zip integrity check | `paper_review_bundle.zip` produced from files that actually exist |
| G19 | No new AWS training launched by this workflow | All new experiments use existing checkpoints on local CPU |
| G20 | Automatically render full manuscript draft from measured inputs | `paper_verified/manuscript_draft.md`; authors/references need verification |

## One consolidated execution command on the MacBook

First switch to the branch and install project dependencies. This local command
evaluates the optional 3-expert model on **all existing frozen task/context draws**,
profiles two- and three-expert CPU execution, generates measured-only results and
manuscript, hashes evidence, and verifies the ZIP. It does not run on AWS.

```bash
cd ~/nanoMaternalPFN
git fetch origin
git switch research/gemini-g1-g20-verified-paper-20261113
source .venv/bin/activate
python -m pip install -e ".[paper,test]"
python -m pytest -q
python scripts/26_execute_gemini.py
```

The three-expert run can be expensive on a local CPU; it is **not** a
"two-minute" evaluation. It is resumable and writes under
`artifacts/results/paper_three_expert_full/` without changing existing
v1/v2/adaptive evaluations. It never accesses labels of held-out query rows
when selecting ensemble weights. The three-expert metrics are exploratory
because the existing test sets have already influenced research decisions.

If all three-expert scores are already measured, `--skip-three-expert` can
build the original paper without claiming to run that optional comparator.
To inspect the planned commands without execution, add `--dry-run`.

## Outputs from completed local execution

- `artifacts/results/paper_three_expert_full/summary.csv`:
  **only after real evaluation**; never prepopulated
- `artifacts/results/adaptive_cpu_profile.json`: actual two- and three-expert
  fit/CV and single-query timing; synthetic timing inputs, not clinical scoring
- `artifacts/results/paper_verified/leaderboard_all.csv`: all original methods
- `artifacts/results/paper_verified/leaderboard_low_data.csv`: contexts <=64
- `artifacts/results/paper_verified/low_data_auroc_by_dataset.csv`
- `artifacts/results/paper_verified/ctu_auroc_by_context.csv`
- `artifacts/results/paper_verified/paired_effects.csv`: draw-level intervals,
  not patient/independent-hospital confidence intervals
- `artifacts/results/paper_verified/provenance.json`: SHA-256 evidence fingerprints
- `artifacts/results/paper_verified/manuscript_draft.md`
- `artifacts/results/paper_review_bundle.zip`: original and newly measured data

## Important scientific boundaries

The existing **eight tasks come from seven source cohorts**, including two
outcomes in one VAT cohort and a non-maternal Pima transfer control. The
original v1/v2/adaptive development history is openly disclosed. We do not
call a 0.001 AUROC difference statistically equivalent without an equivalence
analysis; a paired draw bootstrap is descriptive, not patient-level inference.

Standalone Tiny ONNX's reported 1.52-MB file cannot be mixed with Adaptive
Tiny's ensemble AUROC; a three-expert selector requires three checkpoints,
plus runtime/cache/fit costs beyond their on-disk bytes. A 16-24 MB
incremental process RSS measurement does not demonstrate under-32 MB total
process RAM. We do not state any unmeasured adaptive inference latency.

**No results from this new branch are claimed measured until the user's local
evaluation completes and its CSV/JSON artifacts can be audited.**
