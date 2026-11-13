# P1-P15: single-branch implementation register

All entries refer to the **same frozen measured data**, not newly fabricated or newly simulated patient results.

| ID | Implementation | Evidence and limitations |
| --- | --- | --- |
| P1 | Included historical v1 raw summary and developed v1-v2 comparison | `tables/v1_vs_v2_development.csv`; historical v1 models exist in the frozen ZIP; multi-component changes preclude mechanistic causal ablation |
| P2 | Audited 1,632 master summary rows, 136 three-expert rows, 11,407 draw records, SHA-256 manifest | `tables/data_audit.json`, pipeline fails on tampered files, mismatched means and counts |
| P3 | Joined matched dataset/context/seed draws; computed directional differences | `tables/three_expert_contrasts_all_metrics.csv`, `tables/primary_task_effects.csv` |
| P4 | Seven-source percentile bootstrap and exact sign-flip sensitivity interval | `docs/paper/STATISTICAL_REPORT.md`; VAT paired; interval not confirmatory patient uncertainty |
| P5 | Included Brier, ECE15, and accuracy throughout alongside AUROC | `tables/leaderboard_all.csv`, `tables/leaderboard_low_data.csv`; no patient calibration slope/curves possible |
| P6 | Reported task, context and low-context results | `tables/task_auroc_all_models.csv`, `tables/primary_by_context.csv` |
| P7 | Verified measured Logistic Regression and Random Forest against master data | Full 13-config leaderboard including classical comparators |
| P8 | Cited eight tasks/seven source cohorts, VAT overlap and Pima control | `docs/DATASET_PORTFOLIO.md`, `CLINICAL_REPORTING_AUDIT.md` |
| P9 | Recorded code-level comparator settings and limitations | `BENCHMARK_FAIRNESS.md` |
| P10 | Consolidated measured ONNX size/latency, p99 and RSS with method labels | `tables/efficiency_measured.csv`, `tables/onnx_memory_measured.csv`; ensemble RSS not measured |
| P11 | Separated adaptive and standalone ONNX predictor claims | `BENCHMARK_FAIRNESS.md`, `MANUSCRIPT.md` |
| P12 | Authored auditable architecture description and three data-derived SVG scientific figures and one source-derived model architecture diagram | `MANUSCRIPT.md`, `figures/*.svg`, deterministic figure script; no invented diagrams or numbers |
| P13 | Packaged frozen hash/checkpoint limitations, scripts, tests, provenance verification | `REPRODUCE.md`, `scripts/paper`, `tests/test_paper_evidence_pipeline.py`, CI |
| P14 | Documented clinical applicability and TRIPOD+AI/PROBAST+AI risk/evidence matrix | `CLINICAL_REPORTING_AUDIT.md` (not external sign-off) |
| P15 | Drafted complete evidence-linked manuscript with all numerical claims traceable to CSV | `MANUSCRIPT.md`, data-driven manuscript generator; author/ethics/funding declarations await actual sign-off |

## No invented numbers policy

Data absent from the frozen ZIP are marked **not measured**. In particular, there are no patient-level predictions, independently audited subgroup labels, DeLong covariance, Android energy profiles, INT8 deployment peaks, or an adaptive ensemble process-RSS measurement. The branch does not synthesize them. No prospective clinical validation is claimed.

## Result worth reporting, without hype

Three-expert Tiny: measured equal-task AUROC 0.707268; TabPFN v2 Default: 0.699226. Difference +0.008042, 95% seven-source bootstrap sensitivity interval [-0.018864, +0.032126], exact sign-flip p=0.578125, non-confirmatory. TabPFN had better Brier/ECE15. The three-expert method also costs more CPU/memory than standalone Tiny. Claims are version-specific and exploratory.
