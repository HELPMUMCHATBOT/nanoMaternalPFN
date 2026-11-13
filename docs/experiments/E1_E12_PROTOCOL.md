# nanoMaternalPFN — E1–E12 complete experiment program

**Program status:** Infrastructure and archive-based retrospective analyses are implementable now. This protocol does **not** imply new GPU pretraining, new clinical records, prospective study, or mobile device measurement has already happened. No invented scores.

**Evidence provenance:** Original paper review ZIP SHA256 `f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26`. Original eight tasks come from **seven source cohorts** because VAT GDM and VAT delivery share participants. The later statistical audit includes 671 matched draws per current method. Models and patient tables must be supplied from the real original workspace; the archived ZIP contains metric summaries, not patient-level predictions or final checkpoint binaries.

## Unified execution

The integrated coordinator validates **all eight frozen dataset JSON/NPZ pairs and required final research checkpoints before running anything**, then executes E3, E5, E6, E8 and E4 in one sequence. It also accepts a separately attested E1 independent dataset and a matching E10 ONNX graph. It never substitutes synthetic data and does not claim E9/E12 clinical/hardware evidence. Full training on GPU requires explicit `--train-models --confirm-compute` and a CUDA accelerator.

```bash
python -m scripts.experiments.run_all --run-full \
 --manifest-dir artifacts/manifests \
 --v1-dir artifacts/results/aws_full_pretrain \
 --v2-dir artifacts/results/aws_v2_pretrain \
 --output artifacts/results/e1_e12
```

Do not run the full command without actual pretrained weights and genuine frozen real source files; it intentionally fails closed rather than generating substitute results.

All modules should be run as `python -m scripts.experiments.MODULE` from the repository root. Runtime outputs belong under ignored `artifacts/results/e1_e12/`. Do not commit raw patient files or detailed prediction arrays. Generate the full E2/E7/E11 matrix without training:

```bash
python -m scripts.experiments.training_matrix --output artifacts/results/e1_e12/training
```

Run an individual **full-budget** synthetic training case only on a real available accelerator; this can incur significant compute charges:

```bash
python -m scripts.experiments.training_matrix \
 --output artifacts/results/e1_e12/training \
 --execute-case E2_reference --device cuda --confirm-compute
```

The matrix enumerates 11 model runs (six E2 including reference, four additional E7 priors and one E11 longer run); E11 records intermediate 5k/10k/20k checkpoints along **one** 20k-step trajectory. E2 adjusts one factor at a time: pre/post normalization, class augmentation, sign augmentation, context-missingness indicators, learning-rate scheduling. E7 compares maternal fractions 0%, 25%, 50%, 70%, 100% at fixed budgets. Use the same frozen 8-task manifest files for these trained arms, but do **not** call results from reused development test sets independent external confirmation. Record both synthetic validation curves and real transfer results.

## E1. Independent external validation — gated clinical evidence

A candidate dataset must be from a genuinely independent and newly examined maternal/perinatal source, with documented participant-level split, target definition, predictor measurement time, licenses, and an **immutable model-selection freeze SHA** from before results inspection. Publish the dataset's provenance/ethics fields in the manifest. The `external_validation` object is mandatory: `source_accession`, `population`, `outcome_definition`, `data_license`, `frozen_before_model_evaluation: true`, `model_selection_freeze_sha` (40-char commit SHA), `independent_of_development_cohorts: true`, `independent_reviewer`.

Create the frozen JSON+NPZ pair using the existing `src/nanomaternalpfn/data/freeze_splits.py` after source eligibility, leakage and consent/license checks. Then add the attestation and **no new model selection after test evaluation**:

```bash
python -m scripts.experiments.run_predictions --experiment E1 \
 --manifests artifacts/manifests/independent_antenatal_cohort.json \
 --methods tiny_generic tiny_mixture tabpfn_v2_default \
 --save-predictions --output artifacts/results/e1_e12/E1
```

The command refuses any of the original eight named datasets as 'external'. This file and matching NPZ **do not exist in the archived review bundle**. A genuinely independent new source is therefore a real external dependency, not a random-split relabeling exercise.

## E2. Controlled architecture ablations

Use the training matrix. **Only one factor changes from the E2 reference**, with common seed streams, synthetic validation seeds, architecture size, dataset tasks, batch size and optimizer budgets. For the missing-indicator ablation, context-only imputation hides the NaN mask in training; the evaluation adapter reads the checkpoint metadata and uses the same context-only imputation at transfer inference. A JSON name-to-checkpoint mapping is passed to `run_predictions --checkpoint-map` for the new trained cases. `train_controlled.py` writes a model checkpoint plus complete validation evidence. Do not infer causal effects from prior v1–v2 comparisons, because those changed multiple factors together.

## E3. Fully tuned classical baselines

Use context-only inner stratified cross-validation and the same frozen held-out indices for all methods; no test score chooses hyperparameters. Search spaces are version-controlled in `predictors.py` and use deterministic `RandomizedSearchCV` with negative log loss. Supply all eight actual frozen JSON+NPZ pairs and use no `--max-draws` for a complete paper run.

```bash
python -m scripts.experiments.run_predictions --experiment E3 \
 --manifests artifacts/manifests/*.json \
 --methods logistic random_forest xgboost lightgbm \
 --n-search 8 --save-predictions --output artifacts/results/e1_e12/E3
```

**Important:** shell globs must not contain unrelated JSON manifests. Prefer an explicit list of the eight source manifests. Inner CV may be undefined on rare-class contexts; failures are recorded rather than defaulting to a fabricated result.

## E4. Matched CPU benchmark

On a single machine with real checkpoints and frozen real-table manifests, run each model in an isolated fresh process under the same repeat count, one query row, thread limits and context rows:

```bash
python -m scripts.experiments.profile_matched --all \
 --manifest artifacts/manifests/low_birth_weight.json \
 --context 64 --repeats 100 \
 --out artifacts/results/e1_e12/E4/matched_cpu.json
```

Measures load, fit/context-CV, query p50/p95/p99, process RSS snapshots, file bytes, CPU architecture and version metadata. A sampled RSS peak is a lower bound on instantaneous allocator peak. This does not produce true energy consumption or process-independent memory equivalence. **The 1.52MB ONNX file is a different predictor from the three-expert model.** For same-implementation comparisons, use the same predictor path and a matched protocol.

## E5. Patient-level discrimination and calibration

`run_predictions --save-predictions` stores class probabilities and true held-out labels in local restricted NPZ per model/task/context/draw, plus source/checkpoint digests. For **the same patients and same draw**, analyze binary paired AUROC uncertainty, Brier and calibration slope/intercept:

```bash
python -m scripts.experiments.analyze_predictions \
 --model-a artifacts/results/e1_e12/E3/restricted_predictions/TASK.MODEL_A.c64.s0.clean.npz \
 --model-b artifacts/results/e1_e12/E3/restricted_predictions/TASK.MODEL_B.c64.s0.clean.npz \
 --bootstraps 5000 --out artifacts/results/e1_e12/E5/patient_analysis.json
```

Bootstrap individuals, not repeated context draws. Calibration slope/intercept uses **binary** observations only; multiclass and clinically meaningful subgroups need a separately specified analysis with sufficient counts. No patient predictions can be reconstructed from the existing review ZIP's aggregated AUROCs/Brier/ECE.

## E6. Expert-selection mechanism

The retrospective measured **three-expert vs two-expert** audit can run now; it is not proof that the *weighting rule* caused the improvement. With real checkpoints, evaluate identical expert pools using fixed v2-mixture, uniform 1/3 weights, cross-fitted context-NLL softmax, and context-CV winner; the label-based weighting sees context only, not held-out query labels:

```bash
python -m scripts.experiments.run_predictions --experiment E6 \
 --manifests artifacts/manifests/*.json \
 --methods uniform context_cv_softmax context_cv_winner fixed_v2_mixture \
 --save-predictions --output artifacts/results/e1_e12/E6
```

## E7. Synthetic-prior composition

The original generic vs 70% maternal-mixture **measured** comparison is available as a retrospective contrast. For a controlled mixture experiment use 0/25/50/70/100% training from the matrix and evaluate each resulting checkpoint under identical frozen contexts. Report both cases where maternal mixtures help and those where generic priors lead.

## E8. Robustness and domain shift

Using actual frozen clinical tables and existing model checkpoints, the robustness runner measures clean and **simulated perturbations of real test predictors** at 10%/25% additional missingness and 10%/25% context-SD Gaussian noise, plus a labeled-context minority-class thinning stress that retains at least two minority examples. All perturbations are labeled simulation; no simulation is called real external data. Run:

```bash
python -m scripts.experiments.run_predictions --experiment E8 \
 --manifests artifacts/manifests/*.json --methods tiny_generic tiny_mixture \
 --output artifacts/results/e1_e12/E8
```

**Hospital-to-hospital validation** requires real site identifiers and multiple independent hospital cohorts; the existing review ZIP does not supply them. The class-imbalance stress deterministically thins 50% of the minority class from the labeled context, refits the model without test labels, and records the reduced effective context size. No simulated perturbation is treated as a naturally collected cohort.

## E9. Actual ARM/Android measurements

Run **E4 benchmark on the physical ARM Linux target** and capture CPU architecture, memory and timing; physical Android needs an ONNXRuntime Android app/instrumentation harness, compatible fixed-shape ONNX input, and adb access. `edge_arm.sh` fails unless running natively on ARM. Power measurements require a calibrated profiler or power meter; battery level change is not energy measurement. No such device is accessible from this conversation.

## E10. Quantization

```bash
python -m scripts.experiments.quantize_and_check \
 --onnx artifacts/results/edge_paper_comparison/v2_ctx64.onnx \
 --output-onnx artifacts/results/e1_e12/E10/v2_ctx64.int8.onnx \
 --manifest artifacts/manifests/low_birth_weight.json \
 --report artifacts/results/e1_e12/E10/quantization.json
```

This validates ONNX operator compatibility, FP32-versus-INT8 predictions and AUROC/Brier/ECE/accuracy on one real held-out test pool and one frozen context, and records inference times under the same CPU protocol. Do not extrapolate to Android memory or power. Repeat all frozen contexts/tasks for a paper-level accuracy claim. A full 8-task quantized transfer study requires individually shape-matched ONNX exports and frozen clinical test pools.

## E11. Longer pretraining

One **20,000-step** training trajectory with checkpoints at 5k, 10k, and 20k holds training architecture and prior constant; use the same synthetic validation seed to examine saturation, then evaluate each checkpoint on frozen real tasks. Do not label shorter experimental checkpoints fully trained models or claim convergence without validation curves. A cosine decay fixed to different total lengths can confound step-budget comparisons; the default controlled E11 schedule is constant.

## E12. Prospective clinical utility (not runnable from historical ZIP)

**Real participant-facing research requires independent approvals and clinical implementation.** See `PROSPECTIVE_CLINICAL_PROTOCOL.md`. A prospective trial must finalize outcome ascertainment, clinical workflow, consent/waiver/ethics review, data protection and the action threshold before collecting/analyzing trial outcomes. The current code does not enroll patients, provide care advice, send notifications or change clinical decisions.

## Execution statuses / strict completion

```bash
python -m scripts.experiments.status --manifest-dir artifacts/manifests \
 --output artifacts/results/e1_e12/status.json
```

For E1–E12, status is `measured_verified` **only** when a corresponding experiment has a `COMPLETE.json` containing explicit `measured=true`, `verified=true`, `source_sha256`, and reviewable evidence; a runnable script alone is insufficient. This status is an evidence ledger, not a scheduler. Note that synthetically prepared smoke fixtures and smoke checkpoints test code, but **never** count as final research measurements.
