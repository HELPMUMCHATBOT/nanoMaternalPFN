# Fairness and interpretation of the existing benchmark

## What is and is not comparable

The eight real tasks were originally collected from seven **source cohorts**, not eight independent maternal populations. PhysioNet VAT gestational diabetes and VAT delivery-type tasks share the same cohort. Pima is non-maternal. UCI maternal-risk categories are a risk proxy, not a prospectively observed morbidity event. Dataset/context/seed were matched across models wherever valid; a dataset with more context sizes or raw draws is not assigned greater **primary task weight**.

An overall 0.707268 versus 0.699226 measured AUROC for three-expert Tiny versus **official TabPFN v2 Default** means an observed macro lead of +0.008042. It does **not** mean tested statistical equivalence, general superiority to TabPFN versions beyond v2, superiority on all eight tasks, or zero-shot *model selection* on an untouched independent test benchmark. The real rows were not used for synthetic weight pretraining, but the model choices were influenced by previously inspected benchmark outcomes.

## Recorded baseline defaults

Source: `src/nanomaternalpfn/evaluation/baselines.py` on the parent branch and frozen runtime configurations.

| Comparator | Observed configuration | Interpretation |
| --- | --- | --- |
| Logistic regression | SimpleImputer(add_indicator), StandardScaler, LogisticRegression(max_iter=1000) | Default regularization, no nested tuning |
| Random forest | 200 trees, single-thread, fixed seed | No hyperparameter search |
| XGBoost | 200 histogram-method estimators, single-thread, fixed seed, class-specific objective | No nested hyperparameter search |
| LightGBM | 200 estimators, single-thread, otherwise library defaults | `min_child_samples=20` default may inhibit splits with N=16/32 |
| TabPFN v2 single | Official `TabPFNClassifier`, explicit `ModelVersion.V2`, `n_estimators=1` | Separate model from ensemble default |
| TabPFN v2 Default | Official `ModelVersion.V2`, `n_estimators=auto` | Compare only against precisely named v2 version |
| nanoMaternalPFN v2 | Existing pretrained synthetic weights, task-context inference | Standalone Tiny/Base × generic/maternal priors |
| Adaptive 2-/3-expert | Existing pretrained checkpoints, labeled-context selection/temperature | Multi-checkpoint model and extra fit/query cost |

Use the phrase **outperformed the recorded default-configuration boosted-tree baselines on the observed macro average**, never "proved superiority to tuned gradient boosting". Pima and VAT outcomes may show distinct patterns. In particular, LightGBM's small-context behavior is configuration-sensitive and is documented instead of extrapolated.

## What the development comparison cannot prove

v1 -> v2 changed Pre-LN/normalization behavior, class-index randomization, synthetic training augmentation and optimization scheduling together. The v1-v2 numerical table is **historical development evidence**, not a surgical causal ablation isolating one architecture mechanism. Two- and three-expert variants also differ in expert count, checkpoints, cross-validation work and selection opportunity. The three-expert pool was suggested *after* seeing earlier task scores; its paired inferential tests are exploratory.

## Calibration and multiplicity

Compare all four measured metrics: AUROC and accuracy (higher better), multiclass Brier and ECE15 (lower better). In this archive, TabPFN v2 Default has better aggregate Brier and ECE15 than three-expert Tiny, even as it has slightly lower observed AUROC. Paired tests resample **source-cohort** clusters, keeping VAT paired, and include adjustment for 12 comparator AUROC contrasts. They are not confirmatory tests of external clinical transportability.

## Edge and hardware limitations

Standalone v2 Tiny maternal-mixture ONNX is measured at ~1.52 MB and 2.649 ms median per single CPU query at context 64. The three-expert adaptive Tiny checkpoint collection is ~4.348 MB with a separately measured warmed PyTorch CPU query median ~17.76 ms, plus ~130.58 ms context CV fit at context 64. These are **distinct predictors and timing protocols**, and their ratio is not a matched end-to-end benchmark. Measured standalone ONNX total-process peak RSS exceeds 32 MB. No Android/device/INT8 energy conclusions are available. Do not substitute incremental RSS for full process memory.

## Remaining evidence gaps

This package cannot determine patient-level confidence intervals, subgroup impact, causal benefit, threshold-specific net benefit, clinically representative transportability or audited source labels. Those gaps are reported openly in the manuscript; no fabricated scores or retrospective test-set reoptimization is allowed in this branch.
