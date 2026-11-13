# Full Statistical Analysis — nanoMaternalPFN

**Evidence freeze:** uploaded 256-file `nanoMaternalPFN-paper_review_bundle.zip`  
**Analysis date:** 13 November 2026  
**Primary comparison:** three-expert Adaptive Tiny versus official TabPFN v2 Default  
**Classification:** retrospective exploratory model benchmark; no patient-level validation or prospective decision support testing.

## 1. Executive conclusion

The three-expert nanoMaternalPFN-Tiny model **ranks first descriptively** among 13 current evaluated models by equal-task macro AUROC: **0.707268** versus **0.699226** for TabPFN v2 Default. The observed difference is **+0.008042 AUROC**, or **0.804 percentage points of AUROC**. This is an *observed benchmark lead*, not an independently established superiority result.

Source-aware sensitivity analysis across seven source cohorts (eight prediction tasks; both VAT outcomes kept in one cluster) yields a **95% seven-source percentile-bootstrap interval [-0.0189, +0.0321]** and an **exact two-sided source sign-flip p-value of 0.578125**. The interval spans zero, and the p-value is not evidence of superiority. It beats TabPFN on **4/8 tasks** and **3/7 source-cohort averages**. The equal-source-weighted observed difference is **+0.004223**, lower than the task-weighted effect because VAT provides two outcomes.

At low labeled-context sizes (16/32/64), the three-expert model leads by **+0.011540 AUROC**, but its seven-source interval **[-0.0131, +0.0343]** also crosses zero (exact sign-flip p=0.468750).

**Correct current claim:** “Three-expert nanoMaternalPFN-Tiny achieved the highest observed macro AUROC in an exploratory eight-task benchmark, with performance heterogeneity and no established statistical superiority over TabPFN v2 Default.” This should not be stated as prospective clinical superiority, validated equivalence, or a definitive superiority claim.

## 2. Integrity, completeness and estimand

- **Manifest audit:** 256 SHA-256 file hashes and byte sizes all match; ZIP CRC validation succeeds. Full input ZIP SHA-256: `f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26`.
- **Raw evidence:** 11,407 measured draw records across 17 configurations (13 current model arms plus 4 historical v1 arms). All 13 current arms have **671 paired draws** over **34 matched dataset/context cells** (total model/metric aggregate rows = 1,768).
- **Replicated summaries:** every numerical summary mean recomputed from its own per-draw CSV agrees within at most **3.33e-16**; no missing or mismatched draw counts.
- **Study structure:** eight prediction tasks from seven source cohorts. VAT GDM and VAT Delivery are two outcomes from one source; Pima Diabetes is the nonmaternal transfer control and UCI maternal-risk category is a risk proxy.
- **Primary reported average:** average the eligible context-draw means within each task and average the eight task scores equally. Each of the 34 task/context cells contributes within its own task, rather than treating the number of draws or eligible contexts as a task's relative importance.
- **Matched comparison:** every model pair is joined by exact dataset, context size, and frozen draw seed; all contrasts use matching records. No synthetic clinical performance numbers are inserted.
- **Known data limitation:** the ZIP has raw *performance metrics per context draw*, but **does not have held-out patient-by-patient predictions or true test labels**. Therefore patient-level DeLong AUROC intervals, patient-block bootstrap, calibration curves by observation and subgroup/decision-curve statistics cannot be computed from this bundle.

## 3. Entire 13-method leaderboard (all eligible contexts)

All values below are directly recomputed from measured summary and draw-level CSVs. For AUROC and accuracy, higher is better. For Brier score and ECE15, lower is better. Brier is the recorded multiclass probability score.

| method | auroc | brier | ece15 | accuracy |
| --- | --- | --- | --- | --- |
| adaptive_tiny_3expert | 0.7073 | 0.3358 | 0.1048 | 0.7703 |
| tiny_generic | 0.7028 | 0.3423 | 0.1153 | 0.7584 |
| tabpfn_v2_default | 0.6992 | 0.3165 | 0.0930 | 0.7864 |
| adaptive_tiny | 0.6983 | 0.3434 | 0.1057 | 0.7645 |
| adaptive_base | 0.6950 | 0.3385 | 0.0975 | 0.7670 |
| base_generic | 0.6949 | 0.3445 | 0.1254 | 0.7632 |
| tabpfn_v2_single | 0.6949 | 0.3211 | 0.0952 | 0.7841 |
| tiny_mixture | 0.6916 | 0.3504 | 0.1182 | 0.7602 |
| base_mixture | 0.6912 | 0.3433 | 0.1101 | 0.7591 |
| random_forest | 0.6892 | 0.3335 | 0.1093 | 0.7668 |
| logistic | 0.6803 | 0.3468 | 0.1221 | 0.7680 |
| xgboost | 0.6632 | 0.3940 | 0.1646 | 0.7472 |
| lightgbm | 0.6085 | 0.3652 | 0.1075 | 0.7562 |


**Important:** the standalone *v2 Tiny Generic* model also has higher observed macro AUROC (**0.702838**) than the TabPFN v2 Default macro (**0.699226**). That is an observed number, not a significance claim. The three-expert Adaptive model must not inherit Tiny Generic's single-model deployment size or latency.

## 4. Low-data leaderboard (labeled contexts 16, 32 and 64)

| method | auroc | brier | ece15 | accuracy |
| --- | --- | --- | --- | --- |
| adaptive_tiny_3expert | 0.6955 | 0.3453 | 0.1099 | 0.7646 |
| tiny_generic | 0.6894 | 0.3531 | 0.1189 | 0.7506 |
| adaptive_tiny | 0.6877 | 0.3523 | 0.1114 | 0.7606 |
| adaptive_base | 0.6848 | 0.3476 | 0.1031 | 0.7623 |
| tabpfn_v2_default | 0.6839 | 0.3341 | 0.0997 | 0.7748 |
| base_mixture | 0.6819 | 0.3509 | 0.1102 | 0.7564 |
| base_generic | 0.6813 | 0.3559 | 0.1273 | 0.7543 |
| tiny_mixture | 0.6807 | 0.3573 | 0.1172 | 0.7552 |
| tabpfn_v2_single | 0.6804 | 0.3387 | 0.1023 | 0.7727 |
| random_forest | 0.6726 | 0.3490 | 0.1141 | 0.7574 |
| logistic | 0.6591 | 0.3682 | 0.1327 | 0.7546 |
| xgboost | 0.6421 | 0.4117 | 0.1672 | 0.7306 |
| lightgbm | 0.5580 | 0.3845 | 0.0953 | 0.7377 |


In this subset, the three-expert model is descriptively first, but it does not establish consistent superiority in source-level resampling. All eight tasks contribute to this low-data aggregate.

## 5. Primary inference: three-expert versus TabPFN v2 Default

### 5.1 Across-source effect and formal exploratory tests

- **Mean paired AUROC effect:** +0.008042 (three-expert minus TabPFN, equal task weight).
- **95% source-cluster percentile-bootstrap interval:** [-0.018864, +0.032126] using 100,000 replicates of seven source units, keeping VAT outcomes together.
- **Exact two-sided source-cohort sign-flip test:** p=0.578125; exact one-sided sign-flip p=0.289062, under the sign-symmetry null. These are not confirmatory because three-expert design/model selection used earlier views of these same benchmarks.
- **Paired, source-averaged t-test:** p=0.775186 on only seven cohort means; **source Wilcoxon exact signed rank:** p=0.937500; **source sign test:** p=1.000000. These are weak, small-sample sensitivity tests, not proof of equal performance.
- **Holm familywise p-value** over 12 exploratory AUROC comparator contrasts: 1.000000.
- **Task wins:** 4 of 8; cohort-average wins: 3 of 7.

### 5.2 Why 671 draws do not make the p-value small

With the patients in each frozen test pool reused across context draws, the draw is **not an independent patient or hospital cohort**. Resampling the paired context draws while fixing the same patients produces a narrower **conditional** 95% interval of **[+0.004364, +0.011792]**. It describes sensitivity to context sampling, *not* generalization to independent cohorts. The far wider seven-source interval is much more revealing of real between-task heterogeneity, although seven convenience-source cohorts still do not provide a definitive population generalization interval. **Do not present the draw-level interval as a clinical superiority confidence interval.**

### 5.3 Dataset-level AUROC differences

| dataset | adaptive_tiny_3expert | tabpfn_v2_default | adaptive_tiny | tiny_generic | Δ_3expert_minus_TabPFN | Δ_3expert_minus_2expert |
| --- | --- | --- | --- | --- | --- | --- |
| uci_maternal_risk | 0.7748 | 0.8132 | 0.7726 | 0.7555 | -0.0385 | +0.0022 |
| vat_gdm | 0.7117 | 0.6834 | 0.7150 | 0.7056 | +0.0283 | -0.0033 |
| vat_delivery | 0.6176 | 0.5763 | 0.5775 | 0.6514 | +0.0413 | +0.0400 |
| low_birth_weight | 0.7469 | 0.6759 | 0.7437 | 0.7341 | +0.0710 | +0.0032 |
| cardiotocography | 0.8965 | 0.9182 | 0.8859 | 0.8757 | -0.0217 | +0.0106 |
| pima_diabetes | 0.8217 | 0.8294 | 0.8040 | 0.8232 | -0.0077 | +0.0177 |
| tpehg_preterm | 0.5345 | 0.5490 | 0.5188 | 0.5372 | -0.0146 | +0.0157 |
| ctu_cord_acidemia | 0.5545 | 0.5483 | 0.5688 | 0.5399 | +0.0062 | -0.0143 |


The three-expert model exceeds TabPFN on Low Birth Weight, VAT Delivery, VAT GDM, and CTU Cord Acidemia, while TabPFN leads on UCI Maternal Risk, Cardiotocography, TPEHG Preterm and Pima Diabetes. The largest positive effect is Low Birth Weight (+0.0710), whereas UCI Maternal Risk is a large negative contribution. Dataset selection strongly influences the aggregate.

## 6. Sensitivity to task and cohort inclusion

| scenario | n_tasks | n_sources | equal_task_effect | cohort_ci_lower | cohort_ci_upper | exact_signflip_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| all_tasks | 8 | 7 | +0.0080 | -0.0190 | +0.0322 | 0.5781 |
| maternal_only_no_Pima | 7 | 6 | +0.0103 | -0.0209 | +0.0367 | 0.5312 |
| exclude_VAT_pair | 6 | 6 | -0.0009 | -0.0238 | +0.0305 | 0.9688 |
| exclude_low_birth_weight | 7 | 6 | -0.0010 | -0.0242 | +0.0198 | 0.9688 |
| exclude_UCI_proxy | 7 | 6 | +0.0147 | -0.0123 | +0.0382 | 0.4688 |
| strict_no_Pima_or_UCI | 6 | 5 | +0.0184 | -0.0133 | +0.0447 | 0.4375 |


Excluding the shared VAT pair changes the observed lead to **-0.000868**. Excluding Low Birth Weight changes it to **-0.000958**. Neither is a preregistered primary analysis; both are transparent influence checks. Leave-one-source-out observed macro effects range from **-0.000958** to **+0.014684**.

## 7. Context-size analysis

Each row averages only tasks eligible for that context; larger budgets evaluate fewer tasks. Do **not** interpret differences across context budgets as controlled learning curves over an unchanged set of tasks.

| contexts | n_tasks | n_sources | paired_draws | three_macro | other_macro | equal_task_effect | cohort_ci_lower | cohort_ci_upper | exact_signflip_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16,32,64 | 8 | 7 | 471 | 0.6955 | 0.6839 | +0.0115 | -0.0131 | +0.0343 | 0.4688 |
| 16 | 8 | 7 | 152 | 0.6631 | 0.6552 | +0.0079 | -0.0128 | +0.0304 | 0.5469 |
| 32 | 8 | 7 | 159 | 0.7017 | 0.6864 | +0.0153 | -0.0129 | +0.0409 | 0.4375 |
| 64 | 8 | 7 | 160 | 0.7216 | 0.7101 | +0.0114 | -0.0180 | +0.0358 | 0.5781 |
| 128 | 5 | 5 | 100 | 0.7787 | 0.7860 | -0.0073 | -0.0358 | +0.0319 | 0.7500 |
| 256 | 3 | 3 | 60 | 0.8617 | 0.8942 | -0.0325 | -0.0464 | -0.0101 | 0.2500 |
| 512 | 2 | 2 | 40 | 0.8948 | 0.9242 | -0.0294 | -0.0513 | -0.0074 | 0.5000 |
| all | 8 | 7 | 671 | 0.7073 | 0.6992 | +0.0080 | -0.0190 | +0.0322 | 0.5781 |


Three-expert Tiny leads at context 16, 32 and 64 in point estimates. It trails TabPFN at context 128, 256 and 512, but these larger-context comparisons include fewer source cohorts. No individual context-budget test establishes source-level superiority.

## 8. Four-metric assessment: discrimination, accuracy and calibration

A positive “effect” always favors three-expert Tiny; for Brier and ECE15 the effect is **comparator minus three-expert** (because lower is better). This convention prevents accidentally interpreting worse calibration as improvement.

| metric | macro_three | macro_competitor | equal_task_effect | cohort_boot_ci_lower | cohort_boot_ci_upper | exact_source_signflip_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| auroc | 0.7073 | 0.6992 | +0.0080 | -0.0189 | +0.0321 | 0.5781 |
| brier | 0.3358 | 0.3165 | -0.0192 | -0.0409 | -0.0018 | 0.1094 |
| ece15 | 0.1048 | 0.0930 | -0.0118 | -0.0210 | -0.0042 | 0.0625 |
| accuracy | 0.7703 | 0.7864 | -0.0161 | -0.0397 | +0.0023 | 0.2500 |


TabPFN's Brier score is lower by **0.019235**, ECE15 lower by **0.011840**, and accuracy higher by **0.016100** in the observed equal-task aggregates. This means the accuracy–efficiency narrative is **multiobjective**, not “three-expert wins all endpoints.” Because source-level p-tests and bootstrap intervals have different small-sample assumptions, treat conflicting significance indicators cautiously; none of these endpoints establish confirmatory model superiority.

Against the original *two-expert* Adaptive Tiny, adding the generic expert improved observed AUROC by **+0.008982** and Brier by **+0.007665**, with ECE15 changing by **+0.000886** (positive numbers favor three-expert). This suggests a useful exploratory incremental model improvement, but the cohort-bootstrap AUROC interval crosses zero, and the AUROC exact sign-flip p-value is **0.125000**.

## 9. Every AUROC comparator contrast, source-aware inference and multiplicity

The following table compares three-expert Tiny against *each* of the 12 other present model configurations. Holm adjustments were applied across these 12 exploratory AUROC comparisons. Additional metrics are fully listed in `three_expert_contrasts_all_metrics.csv`; they are not treated as preregistered coprimary tests.

| comparator | equal_task_effect | equal_source_effect | cohort_boot_ci_lower | cohort_boot_ci_upper | exact_source_signflip_p_two_sided | holm_p_12_competitors | task_wins | source_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny_generic | +0.0044 | +0.0070 | -0.0055 | +0.0154 | 0.4375 | 1.0000 | 5 | 4 |
| tabpfn_v2_default | +0.0080 | +0.0042 | -0.0189 | +0.0321 | 0.5781 | 1.0000 | 4 | 3 |
| adaptive_tiny | +0.0090 | +0.0076 | -0.0004 | +0.0156 | 0.1250 | 0.8750 | 6 | 6 |
| adaptive_base | +0.0123 | +0.0081 | -0.0042 | +0.0274 | 0.2969 | 1.0000 | 6 | 5 |
| base_generic | +0.0124 | +0.0108 | -0.0003 | +0.0217 | 0.1250 | 0.8750 | 6 | 5 |
| tabpfn_v2_single | +0.0124 | +0.0092 | -0.0143 | +0.0377 | 0.4531 | 1.0000 | 4 | 3 |
| tiny_mixture | +0.0156 | +0.0142 | +0.0071 | +0.0220 | 0.0312 | 0.3438 | 7 | 6 |
| base_mixture | +0.0160 | +0.0137 | +0.0049 | +0.0255 | 0.0625 | 0.5625 | 7 | 6 |
| random_forest | +0.0180 | +0.0132 | -0.0137 | +0.0441 | 0.3750 | 1.0000 | 5 | 4 |
| logistic | +0.0270 | +0.0214 | +0.0023 | +0.0486 | 0.1094 | 0.8750 | 7 | 6 |
| xgboost | +0.0440 | +0.0403 | +0.0141 | +0.0701 | 0.0469 | 0.4688 | 7 | 6 |
| lightgbm | +0.0988 | +0.0956 | +0.0611 | +0.1292 | 0.0156 | 0.1875 | 8 | 7 |


**None of the 12 AUROC comparisons meets p<0.05 after Holm correction.** For example, the exploratory unadjusted exact p-value for LightGBM is 0.015625, but adjusted Holm p=0.187500. A lower LightGBM macro under its recorded default settings does not justify a broad claim against tuned gradient boosting.

## 10. What the three-expert gate actually learned

The expert gates were cross-fitted using labeled-context data only. Diagnostic JSONLs include three expert weights, CV log losses, temperature, status and context seed for all **671** matched draws. Their weights sum to one within numerical precision. Expert 3 is the pretrained **v2 Tiny Generic** checkpoint; the first two are v1 and v2 Tiny Maternal-Mixture checkpoints. Average weights are descriptive and do not by themselves prove causal benefit from a prior.

| dataset | generic_mean_weight | fraction_generic_top | delta_auroc_vs_two | mean_temperature |
| --- | --- | --- | --- | --- |
| cardiotocography | 0.4037 | 0.4322 | +0.0106 | 0.5716 |
| ctu_cord_acidemia | 0.3090 | 0.3276 | -0.0143 | 1.0423 |
| low_birth_weight | 0.2805 | 0.0750 | +0.0032 | 1.1319 |
| pima_diabetes | 0.3793 | 0.7667 | +0.0177 | 0.9076 |
| tpehg_preterm | 0.3171 | 0.3467 | +0.0157 | 0.8987 |
| uci_maternal_risk | 0.4365 | 0.5500 | +0.0022 | 0.8208 |
| vat_delivery | 0.2985 | 0.3333 | +0.0400 | 1.2423 |
| vat_gdm | 0.3647 | 0.6500 | -0.0033 | 0.9358 |


Importantly, the generic expert was **not** universally the highest-weight model on VAT Delivery, despite VAT Delivery exhibiting the biggest three-expert performance increase over original Adaptive Tiny. This corrects the earlier unsupported claim that context CV must reliably pick the generic expert on every favored dataset. The selector improves ensemble behavior through the full weighting/calibration system, not a guaranteed per-task winner choice.

## 11. v1 versus v2 longitudinal development comparisons

These v1/v2 estimates compare historical arms on the same frozen test tasks. Changes between versions were not all isolated one factor at a time; therefore the comparisons describe **development progression**, not causal experiments attributing effects specifically to Pre-LN or permutation fixes.

| arm | v2_improvement_positive | v2_task_wins | cohort_ci_lower | cohort_ci_upper | source_signflip_p_two_sided |
| --- | --- | --- | --- | --- | --- |
| tiny_generic | +0.0149 | 5 | +0.0050 | +0.0248 | 0.0625 |
| tiny_mixture | +0.0428 | 5 | -0.0167 | +0.1355 | 0.4062 |
| base_generic | +0.2243 | 8 | +0.1333 | +0.3317 | 0.0156 |
| base_mixture | +0.0289 | 7 | -0.0130 | +0.0757 | 0.2656 |


Full four-metric v1-versus-v2 effects appear in `v1_vs_v2_development.csv`.

## 12. Measured efficiency and memory — no cross-model mixing

The new adaptive CPU profile uses **warmed PyTorch inference**, single synthetic query, 6 features, and seven repeats for each context. The EdgeBench baseline profile uses a different routine (five repeats for classical/TabPFN, 100 for standalone ONNX). As a result, measured timings can be reported but should not be characterized as a controlled, equal-protocol system-speedup ratio. Fit/context CV time is separate from query time.

| method | context | runtime | fit_cv_p50_ms | query_p50_ms | query_p99_ms | combined_checkpoint_mb | repeats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_tiny | 16 | PyTorch CPU | 22.7003 | 5.6629 | 5.8378 | 2.8986 | 7 |
| adaptive_tiny_3expert | 16 | PyTorch CPU | 31.9132 | 7.8858 | 8.3073 | 4.3480 | 7 |
| adaptive_tiny | 64 | PyTorch CPU | 90.8820 | 12.0872 | 12.7203 | 2.8986 | 7 |
| adaptive_tiny_3expert | 64 | PyTorch CPU | 130.5791 | 17.7597 | 20.8193 | 4.3480 | 7 |
| adaptive_tiny | 128 | PyTorch CPU | 264.1317 | 19.0558 | 20.3978 | 2.8986 | 7 |
| adaptive_tiny_3expert | 128 | PyTorch CPU | 396.5519 | 25.3085 | 31.1511 | 4.3480 | 7 |
| standalone_tiny_mixture_onnx | 16 | ONNX CPU | — | 0.9427 | 0.9653 | 1.5199 | 100 |
| standalone_tiny_mixture_onnx | 64 | ONNX CPU | — | 2.6485 | 2.9235 | 1.5222 | 100 |
| standalone_tiny_mixture_onnx | 128 | ONNX CPU | — | 5.0977 | 5.8559 | 1.5253 | 100 |
| tabpfn_v2_default | 16 | Baseline CPU | 28.0325 | 138.1190 | — | — | 5 |
| tabpfn_v2_default | 64 | Baseline CPU | 28.4914 | 339.8871 | — | — | 5 |
| tabpfn_v2_default | 128 | Baseline CPU | 32.2517 | 714.9843 | — | — | 5 |


The three-expert combined measured PyTorch checkpoint size is **4.347998 MB** (2-expert = **2.898644 MB**). At context 64, it measured **17.760 ms** warmed CPU query p50 and **130.579 ms** context fit/CV p50. This is slower than original Adaptive Tiny's **12.087 ms** query p50. The **1.522 MB ONNX** size at context 64 belongs to the **standalone v2 Tiny Mixture model**, not the ensemble that scored 0.707268. It cannot be used as the three-expert file footprint.

Memory measurements are for isolated **standalone ONNX** processes and do not estimate three-expert peak runtime RAM. The v2 Tiny Mixture process at context 64 had **71.359 MB total process peak RSS** and **24.359 MB incremental peak over import baseline**. All reported `meets_32mb_total_process_target` flags are false; no under-32MB total RAM claim is supported. No Android, power, INT8 or mobile-native memory study was available in the archive.

## 13. Statistical limitations and interpretation

1. **No patient-level predictions:** The ZIP contains aggregate AUROC/Brier/ECE15/accuracy for each frozen context draw, not arrays of held-out patient labels and model probabilities. Consequently, no valid DeLong test of paired per-patient AUROC, patient-block bootstrap, calibration slope/intercept, clinical decision-curve analysis, patient subgroup testing or per-patient paired discordance tests can be performed from this bundle alone.
2. **Only seven source cohorts:** Two VAT outcomes share a source; the source-cluster bootstrap's 95% interval is a sensitivity summary with weak small-sample precision, not an external population validation interval. All sources are convenience benchmarks, not a probability sample of hospitals.
3. **Reuse and adaptivity:** The eight datasets remained unseen by neural pretraining weights, but earlier test metrics informed v2/adaptive research choices and the optional third expert was chosen after benchmark inspection. That creates adaptive benchmark selection. Post hoc p-values and CIs do not eliminate selection bias.
4. **No prospective clinical validation:** Dataset labels, class prevalence, quality, time of predictor availability, data collection mechanisms, and external reproducibility differ. The Pima task is a nonmaternal transfer control; UCI risk category is a proxy label.
5. **Many comparisons and metrics:** There are 13 current evaluated models and four metrics, with repeated contexts. We supply 12-comparison Holm corrections *within AUROC* but do not claim that these solve the broader multiple-selection and multi-endpoint problem.
6. **Configured vs fully tuned baselines:** XGBoost and LightGBM used recorded configurations. LightGBM defaults can produce constant prediction (AUROC 0.5) in very small contexts. Claims of categorical superiority over fully tuned classical ML are unsupported.
7. **Efficiency protocols:** Standalone Tiny ONNX and adaptive multi-checkpoint models have different code paths, checkpoint sizes, profiling repeats and context adaptation overhead; a strict head-to-head Pareto point requires the same predictor configurations and profiling protocol.
8. **Source scope:** SHA-256 checks establish consistency of the submitted archive, not external verification of the truth/representativeness of the datasets or an independent audit of the originally executed model weights.

## 14. Manuscript-safe results wording

> Across eight previously inspected frozen real-world prediction tasks derived from seven source cohorts, a context-weighted combination of three pretrained nanoMaternalPFN Tiny checkpoints achieved a descriptive equal-task macro AUROC of **0.7073**, compared with **0.6992** for TabPFN v2 Default. The paired mean difference was **+0.0080**. A seven-source cluster bootstrap yielded an exploratory 95% interval of **[-0.0189, +0.0321]**, and an exact two-sided source sign-flip sensitivity test produced **p=0.578**. Three-expert Tiny led on four of eight tasks. Although it had the highest observed AUROC among the tested configurations, its superiority to TabPFN v2 was **not statistically established** under source-aware sensitivity analyses. TabPFN had better aggregate Brier score, ECE15 and accuracy. The analysis is exploratory because benchmark results had informed iterative model development.

## 15. Files and reproducibility

- `nanoMaternalPFN_full_statistics.py`: analysis implementation; same seed and 100,000-replicate source bootstrap for reproducibility.
- `leaderboard_all.csv`, `leaderboard_low_data.csv`, `task_means_all_metrics.csv`, `task_auroc_all_models.csv`.
- `three_expert_contrasts_all_metrics.csv`: 48 paired comparisons, each with source-cluster intervals, exact sign-flip p-values, Welch-free source t-test, Wilcoxon, task wins and familywise adjustments.
- `primary_task_effects.csv`, `primary_context_effects.csv`, `primary_by_context.csv`, `robustness_exclusions.csv`.
- `three_expert_weights_raw.csv`, `three_expert_weight_diagnostics.csv`, `v1_vs_v2_development.csv`.
- `efficiency_measured.csv`, `onnx_memory_measured.csv`, `raw_summary_consistency.csv`, `integrity_manifest_audit.csv`, `draw_counts.csv`, `data_audit.json`.
- `nanoMaternalPFN_Statistical_Results.xlsx` — curated, labeled, formatted analysis workbook.

**Citable origin:** user-supplied `nanoMaternalPFN-paper_review_bundle.zip`; exact SHA-256 above. Relevant development code: [nanoMaternalPFN research branch](https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/tree/research/gemini-g1-g20-verified-paper-20261113).
