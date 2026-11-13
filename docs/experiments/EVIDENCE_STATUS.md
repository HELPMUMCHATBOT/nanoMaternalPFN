# nanoMaternalPFN E1–E12 — evidence-gated execution status (13 November 2026)

**No claim of E1–E12 full experimental completion is made.** This branch provides executable code, input and privacy contracts, a prospective governance protocol, and new descriptive re-analysis of the real historical frozen benchmark ZIP. The observed E6/E7 contrasts below use **previously collected genuine scores**; they are not independent new clinical measurements.

| Experiment | Deliverable in this branch | Newly executed evidence | Still required to claim full experiment |
| --- | --- | --- | --- |
| E1 independent external | strict frozen-manifest external validator + prediction runner | none | genuinely new source, signed model freeze, real test data |
| E2 architecture | six-factor controlled training plan + executable trainer | unit-tested planner only | 6 full training runs + frozen real transfer |
| E3 tuned baseline | strict context-only CV runner | synthetic **unit test only** | 8 actual JSON/NPZ frozen source pairs |
| E4 CPU matched | isolated CPU multi-model runner | unit-tested interface only | final weights, real task data, one CPU host |
| E5 patient level | binary paired bootstrap, calibration and prediction output capture | synthetic **unit test only** | real held-out individual labels+probabilities |
| E6 expert mechanism | 4 context-only selection rules + **retrospective paired real draw comparison** | **671 matched original draws** 3-vs-2 experts, exploratory | new common-expert-pool weighting experiments on actual checkpoints |
| E7 prior mix | mixture matrix 0/25/50/70/100% + **retrospective generic-vs-mixture comparison** | **671 matched original draws** per tier, exploratory | controlled full mixed-prior pretraining + transfer |
| E8 robustness | reproducible missingness/noise/class-prevalence stress with actual frozen data | software-tested only | original real feature arrays; additional real hospital cohorts for true transport |
| E9 ARM/Android | ARM-native execution gate | not run; no device | actual ARM/Android measurements; calibrated power device if making energy claims |
| E10 quantization | ONNX FP32/INT8 parity, real-row metrics and matched CPU timing | not run | fixed-shape ONNX graphs + real frozen test source |
| E11 longer training | one 20k-step trajectory with 5k/10k/20k checkpointing | not run | full CUDA training and frozen transfer |
| E12 prospective utility | ethics-first preregistration and safety plan | not run; no participants | clinical PI, IRB/regulatory/site approval, prospective patient study |

## Recomputed real observations (historical, not new controlled ablations)

The original 256-entry evidence ZIP was verified by its fixed SHA-256 hash. Paired contrasts below use matched `(dataset, context, seed)` rows, average context-cell means within each task, and average the eight task means equally. They are **not randomized experiments** and do not establish mechanistic causality.

| Previously measured comparison | Equal-task macro AUROC difference (model A − B) | Tasks favoring A | Matched draws for each arm |
| --- | ---: | ---: | ---: |
| E6: three-expert Tiny vs two-expert Tiny | **+0.008982** | 6/8 | 671 |
| E7: v2 Tiny Generic vs v2 Tiny 70% Maternal Mixture | **+0.011194** | 5/8 | 671 |
| E7: v2 Base Generic vs v2 Base 70% Maternal Mixture | **+0.003658** | 4/8 | 671 |

Complete four-metric task-, context- and macro-level results are in `docs/experiments/measured_existing/`. A different expert-selection rule or synthetic-prior ratio cannot be evaluated accurately using these aggregate score files alone: the actual model checkpoints and real frozen source tables are required.

## Evidence disclosure

The 24 MB earlier evidence package includes four explicitly **smoke-test** inference checkpoints and two **smoke synthetic** HDF5 archives; those are not eligible for research-level predictive comparison or training-completion claims. The paper review ZIP contains real evaluated draw metrics and selected CPU profiles, but no final checkpoint binaries and no patient-by-patient predictions. The exact complete frozen dataset NPZ files were not present in the available archive. Full execution is therefore blocked by concrete missing input artifacts and, for E9/E12, device/ethics prerequisites—not by statistical analysis code.

The previous 1,632-summary-row analysis and paper are untouched. New results can be incorporated only after the relevant commands produce verified measured files from actual sources. No fabricated scores are generated.
