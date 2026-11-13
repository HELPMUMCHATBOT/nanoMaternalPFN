# nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction

**Selected submission-facing manuscript — 13 November 2026**  
**Subtitle:** Synthetic pretraining, context-adaptive prediction, and measured CPU efficiency for data-scarce maternal and perinatal tasks  
**Status:** Scientific draft for author review; quantitative findings are traceable to the frozen benchmark archive. Authorship, affiliations, ethics and funding declarations require author confirmation.

## Abstract

**Background:** Maternal and perinatal prediction frequently relies on small, heterogeneous tabular datasets. Prior-data fitted networks (PFNs) can learn across synthetic prediction tasks and subsequently adapt to new problems from a small labeled context, without downstream neural-network weight updates.

**Methods:** We developed nanoMaternalPFN, a family of compact tabular PFNs pretrained on synthetic generic and maternal-informed task distributions. The evaluated configurations include standalone Tiny and Base networks and context-adaptive combinations of pretrained experts. We compared these systems with classical tabular methods and official TabPFN v2 across eight publicly sourced maternal, perinatal and related prediction tasks representing seven source cohorts. Evaluations used fixed context draws, with AUROC, accuracy, Brier score, and 15-bin expected calibration error (ECE15). Separately, we characterized CPU inference of standalone ONNX and adaptive PyTorch implementations.

**Results:** A three-expert Tiny configuration achieved the highest observed equal-task macro AUROC of **0.7073**, compared with **0.6992** for TabPFN v2 Default, **0.6892** for Random Forest, and **0.6632** for the recorded XGBoost configuration. At labeled contexts of 16–64 examples, the corresponding AUROCs were **0.6955** and **0.6839** for three-expert Tiny and TabPFN v2 Default. The observed overall paired AUROC difference was **+0.0080**; source-aware uncertainty included zero (95% exploratory interval **−0.0189 to +0.0321**). TabPFN v2 Default achieved lower aggregate Brier score and ECE15. The separately profiled standalone v2 Tiny maternal-mixture ONNX graph occupied **1.522 MB** and had **2.649 ms** median single-query CPU latency at context 64. The three-expert system occupied **4.348 MB** in combined checkpoint files and had **17.760 ms** median warmed PyTorch CPU query latency under a different profiling protocol.

**Conclusions:** Compact PFNs trained on synthetic task distributions achieved competitive observed discrimination on heterogeneous low-data prediction tasks. nanoMaternalPFN offers complementary single-model and adaptive-expert configurations with distinctly measured computational costs, supporting further work on resource-aware maternal and perinatal prediction. The present retrospective benchmark does not establish statistical superiority over TabPFN v2 or clinical deployment effectiveness.

**Keywords:** maternal health; perinatal prediction; tabular foundation model; prior-data fitted network; in-context learning; synthetic pretraining; low-data prediction; resource efficiency.

## 1. Introduction

Maternal and perinatal predictive modeling presents a practical challenge: useful clinical tables can be small, variables and measurements differ across studies, and developing a separate high-capacity neural model for each outcome may be inefficient. These conditions motivate systems that can transfer learned inductive biases to new tables while conditioning on a small number of labeled examples.

Prior-data fitted networks learn a predictor over sampled tabular tasks; after pretraining, a model can infer predictions for a new task by conditioning on observed feature–label examples rather than updating weights. The original TabPFN work established the promise of this approach for small tables [1]. Healthcare applications nevertheless require attention to dataset heterogeneity, evaluation design, calibration, and computational resources.

We developed **nanoMaternalPFN** to investigate whether compact synthetic-pretrained PFNs can provide a useful accuracy–resource operating point in maternal, perinatal, and closely related tabular classification. The family combines feature-oriented attention, context-restricted patient attention, missingness-aware inputs, and generic or maternal-shaped synthetic task priors. We also investigated context-only weighting of existing pretrained experts, without downstream weight updates.

This paper makes four integrated contributions:

1. A compact, reproducible PFN architecture and synthetic-pretraining workflow, with Tiny and Base configurations and distinct generic and maternal-informed priors.
2. A multi-task study of in-context transfer over eight open prediction tasks, including explicit low-context results and comparison against recorded classical-model and official TabPFN v2 configurations.
3. Measured context-adaptive two- and three-expert configurations, with evaluation against single-checkpoint models and attention to differing task behavior.
4. Separately measured standalone ONNX and adaptive PyTorch CPU resource profiles, enabling an evidence-based description of efficiency without attributing one predictor's size or speed to another.

The central question is not whether one model wins on every outcome, but whether a compact, reusable pretrained family offers competitive prediction across clinically relevant small-table tasks with documented resource requirements.

## 2. Methods

### 2.1 Study design and task portfolio

We evaluated eight publicly sourced prediction tasks representing **seven source cohorts**: UCI Maternal Health Risk; PhysioNet maternal visceral-adipose gestational diabetes (VAT GDM); delivery outcome from the same VAT cohort; MASS low birth weight; UCI cardiotocography; Pima Diabetes; term/preterm electrohysterography (TPEHG); and CTU-UHB cord-pH classification. VAT GDM and delivery are two outcomes measured within one cohort. Pima is a non-maternal metabolic transfer control. The UCI maternal-risk categories are proxy risk labels, and the CTU-UHB cord-pH threshold is an exploratory classification endpoint.

The original study used fixed, group-safe splits and labeled-context draws. Eligible context budgets included 16, 32, 64, 128, 256, and 512 rows; larger budgets were not available for every task. Contexts with insufficient observed class diversity were excluded consistently across compared methods. Dataset acquisition, endpoint handling and split provenance are described in the repository's DATASET_PORTFOLIO and experiment protocol documents.

### 2.2 nanoMaternalPFN design and synthetic pretraining

nanoMaternalPFN processes tabular cells with feature-attention and context-restricted patient-attention operations. Missing-value indicators and context-based preprocessing support tables with heterogeneous features. Pretraining used synthetic tables, not rows from the real-world evaluation datasets. The training study investigated a generic synthetic prior and a mixture containing 70% maternal-shaped and 30% generic synthetic tasks in expectation.

The model family includes historical v1 and improved v2 implementations, with Tiny and Base sizes. Version 2 introduced changes to normalization, class-index augmentation and optimization that are documented in the repository. The historical v1–v2 comparison measures the cumulative development result rather than isolating the causal effect of any one change.

### 2.3 Standalone and adaptive inference

A standalone PFN makes predictions using its fixed pretrained checkpoint and the labeled context. The two-expert Adaptive Tiny configuration combines historical v1 Tiny maternal-mixture and v2 Tiny maternal-mixture experts. The evaluated three-expert configuration additionally includes v2 Tiny Generic. Expert weights and optional temperature are selected from folds **within the labeled context**; the target query labels are not used to fit task-specific weights.

Because experts and model configurations have distinct footprints, we evaluate predictive performance and computational requirements at the **configuration level**, not by substituting the resource metrics of one system for those of another.

### 2.4 Comparators

The benchmark included Logistic Regression, Random Forest, XGBoost, LightGBM, official TabPFN v2 Single and Default, standalone nanoMaternalPFN variants and adaptive variants. Baseline implementations and settings were kept as recorded: Logistic Regression with imputation, standardization and 1,000-iteration limit; 200-tree Random Forest; 200-estimator XGBoost; and 200-estimator LightGBM with other library defaults. These are **fixed recorded baseline configurations**, not an exhaustive nested hyperparameter-tuning study. The TabPFN comparator is explicitly **v2**, not a claim against all TabPFN generations.

### 2.5 Endpoints, aggregation and statistical analysis

Predictive outcomes were AUROC, accuracy, multiclass Brier score and 15-bin ECE15. Higher AUROC and accuracy are favorable; lower Brier and ECE15 are favorable. Macro results average valid context draws within each dataset/context cell, then eligible contexts within each task, and finally give each of the eight tasks equal weight. Paired contrasts match dataset, context and draw seed before comparison.

The archived evaluation comprises **671 matched valid draws per current model** across **34 eligible task/context cells**. For exploratory uncertainty analysis we resampled seven source groups, keeping the two VAT outcomes in one cluster, and computed an exact source-group sign-flip sensitivity test. These intervals describe variation across the available sources and are not patient-level confidence intervals or confirmatory external-validation tests. The experiment history includes iterations informed by earlier benchmark results, so comparative inference is interpreted as exploratory.

### 2.6 CPU and model-size measurements

A standalone v2 Tiny maternal-mixture ONNX graph was profiled with 100 repeated single-query CPU inferences at selected context budgets. Separately, the two- and three-expert implementations were profiled using warmed PyTorch CPU inference and context-cross-validation fit timing, with seven repeated measurements. Other baseline timings used a further protocol. We report file size, context-fit time, query latency and process memory with their original configuration labels; unlike a matched system-level timing trial, these separate measurement protocols do not establish a head-to-head speedup factor.

## 3. Results

### 3.1 Benchmark integrity

The frozen 256-file evidence archive passed SHA-256 and byte-count verification. The master original 12-configuration table comprised 1,632 metric summary rows; an additional 136 rows describe the evaluated three-expert arm. Reconstructed draw means agreed with archived summaries to numerical precision, with no draw-count discrepancies. Complete machine-readable tables and validation routines are provided in the linked research repository.

### 3.2 Predictive discrimination

**Table 1. Selected measured configurations, equal-task macro averages across eligible contexts.** Full 13-configuration and historical v1 results appear in the reproducibility supplement.

| Configuration | AUROC ↑ | Brier ↓ | ECE15 ↓ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: |
| nanoMaternalPFN Adaptive Tiny, 3 experts | **0.7073** | 0.3358 | 0.1048 | 0.7703 |
| nanoMaternalPFN v2 Tiny Generic | 0.7028 | 0.3423 | 0.1153 | 0.7584 |
| Official TabPFN v2 Default | 0.6992 | **0.3165** | **0.0930** | **0.7864** |
| nanoMaternalPFN Adaptive Tiny, 2 experts | 0.6983 | 0.3434 | 0.1057 | 0.7645 |
| nanoMaternalPFN Adaptive Base, 2 experts | 0.6950 | 0.3385 | 0.0975 | 0.7670 |
| nanoMaternalPFN v2 Tiny Maternal Mixture | 0.6916 | 0.3504 | 0.1182 | 0.7602 |
| Random Forest | 0.6892 | 0.3335 | 0.1093 | 0.7668 |
| Logistic Regression | 0.6803 | 0.3468 | 0.1221 | 0.7680 |
| XGBoost | 0.6632 | 0.3940 | 0.1646 | 0.7472 |
| LightGBM | 0.6085 | 0.3652 | 0.1075 | 0.7562 |

The three-expert Tiny configuration had the highest observed macro AUROC (0.7073) and exceeded TabPFN v2 Default by **0.0080** on the paired equal-task summary. The seven-source exploratory bootstrap interval for this contrast was **[−0.0189, 0.0321]**, and the exact two-sided source sign-flip sensitivity test yielded **p = 0.578**. The difference did not establish statistical superiority. TabPFN v2 Default had stronger aggregate Brier, ECE15 and accuracy scores.

The standalone **v2 Tiny Generic** checkpoint also achieved competitive observed discrimination (0.7028 AUROC). Its clinical benchmark result must not be combined with the separately measured **v2 Tiny Maternal Mixture ONNX** file size and latency as if they described one identical predictor.

### 3.3 Low-context performance

**Table 2. Measured macro AUROC, contexts 16, 32 and 64 only.**

| Configuration | AUROC ↑ |
| --- | ---: |
| Adaptive Tiny, 3 experts | **0.6955** |
| v2 Tiny Generic | 0.6894 |
| Adaptive Tiny, 2 experts | 0.6877 |
| Adaptive Base, 2 experts | 0.6848 |
| TabPFN v2 Default | 0.6839 |
| Random Forest | 0.6726 |
| Logistic Regression | 0.6591 |
| XGBoost | 0.6421 |
| LightGBM | 0.5580 |

The observed low-context paired AUROC difference between three-expert Tiny and TabPFN v2 Default was **+0.0115**, with an exploratory source interval **[−0.0131, 0.0343]**. Patterns varied substantially between tasks. The three-expert approach had a positive per-task AUROC contrast on low birth weight (+0.0710 across eligible contexts), VAT delivery (+0.0413), VAT GDM (+0.0283), and CTU cord acidemia (+0.0062), but not every task. The corresponding full task-by-context table is included in the supplement.

### 3.4 Compact-model and adaptive-model efficiency

**Table 3. Measured CPU results for distinctly configured predictor implementations.** Timings arise from different profiling protocols and are not a controlled cross-method acceleration ratio.

| Predictor implementation | Context | Stored artifact size | Median CPU query | Context CV fit |
| --- | ---: | ---: | ---: | ---: |
| Standalone v2 Tiny Maternal Mixture, ONNX | 16 | 1.520 MB | 0.943 ms | Not measured in this profiling path |
| Standalone v2 Tiny Maternal Mixture, ONNX | 64 | **1.522 MB** | **2.649 ms** | Not measured in this profiling path |
| Standalone v2 Tiny Maternal Mixture, ONNX | 128 | 1.525 MB | 5.098 ms | Not measured in this profiling path |
| Adaptive Tiny, 2 experts, PyTorch | 64 | 2.899 MB combined checkpoints | 12.087 ms | 90.882 ms |
| Adaptive Tiny, 3 experts, PyTorch | 64 | **4.348 MB combined checkpoints** | **17.760 ms** | 130.579 ms |

At context 64, standalone ONNX total-process peak RSS was **71.359 MB**, and the increment over the process import baseline was **24.359 MB**. These are different quantities; the measurements do not demonstrate a 32 MB total-process memory footprint. The archive does not contain an adaptive three-expert process-RSS profile or a mobile-device deployment trial.

The results identify two useful operating points within the same model family: a very compact standalone network and a context-adaptive multi-checkpoint predictor with higher observed AUROC but greater computational cost.

## 4. Discussion

This study presents a resource-aware PFN design and a measured comparison across heterogeneous maternal, perinatal, and related tabular prediction tasks. Three observations define its contribution.

First, **small synthetic-pretrained models were competitive without real-row neural-weight pretraining**. The measured v2 Tiny Generic configuration achieved 0.7028 macro AUROC, while the three-expert approach reached 0.7073. This is meaningful even when a test benchmark does not support a definitive ranking between top methods.

Second, **synthetic prior diversity and context-adaptive inference are useful design dimensions**. The generic and maternal-mixture experts produced differing performance patterns across tasks; it would be inaccurate to claim that the maternal-mixture prior was universally better. The three-expert configuration uses labeled-context cross-validation to combine these existing representations, rather than retraining neural weights for each real-world dataset. Its observed average improvement over the original two-expert version was +0.0090 AUROC, with source-level uncertainty that crossed zero.

Third, **computational requirements must follow the specific predictor**. The standalone maternal-mixture Tiny ONNX graph was approximately 1.52 MB and provided millisecond-scale single-query CPU inference at contexts 16 and 64. The three-expert method provided a different predictive operating point, with 4.348 MB of combined checkpoint files and an additional inference and context-fit cost. These results support a practical model-family approach to future deployment research without conflating a single-network resource profile with multi-expert predictive results.

The benchmark also illustrates why multiple endpoints matter: TabPFN v2 Default delivered better aggregate probability calibration scores even though three-expert Tiny had the highest observed AUROC. For clinical research, ranking and calibrated risk estimation are related but distinct objectives.

### 4.1 Scope and limitations

The study is a retrospective, open-data evaluation with eight tasks representing seven cohorts, including two outcomes from one VAT cohort and one non-maternal transfer control. The task mix includes risk proxies and outcome definitions that do not map to one prospective clinical decision. Neural weights were pretrained only on synthetic tasks; however, version and expert-pool development was informed by earlier examination of benchmark outcomes, so the comparative findings are exploratory rather than independent confirmation of a prespecified final model. The seven-source sensitivity interval is correspondingly imprecise. Baseline configurations were fixed rather than exhaustively tuned, and CPU procedures differed by implementation. The archive does not provide patient-level predictions needed for patient-level confidence intervals, calibration plots, subgroup analysis or threshold-specific decision analysis; device deployment and prospective patient outcomes were not evaluated. These boundaries inform follow-up evaluation and do not alter the observed benchmark measurements.

## 5. Conclusion

**nanoMaternalPFN demonstrates a compact, synthetic-pretrained approach to tabular in-context prediction for data-scarce maternal and perinatal research.** Among the tested configurations, a three-expert Tiny predictor produced the highest observed macro AUROC, while the distinct standalone Tiny ONNX configuration offered a small file footprint and measured millisecond-scale CPU inference. Together, these complementary configurations motivate resource-aware clinical prediction research in settings where labeled datasets and compute are constrained. Further independent and deployment-oriented evaluations are needed to determine clinical transportability and operational utility.

## Data availability, reproducibility and declarations

Exact result tables, the source-aware analysis, figure scripts, benchmark implementation, and SHA-256 verification procedures are in the accompanying nanoMaternalPFN GitHub research branch, under docs/paper and scripts/paper. The frozen evidence archive has SHA-256 **f67a41dab4d35b7cc09b9a0a353bbe09928833669ea9547678b804a973167a26**. This manuscript does not contain invented raw patient predictions or unmeasured device metrics.

**Declarations awaiting author confirmation:** author names and contributions; affiliations; funding sources; conflicts of interest; institutional ethics determination and source-specific data-use/license review. These should be completed from actual records before submission.

### References

1. Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature*. 2025;637:319–326. https://doi.org/10.1038/s41586-024-08328-6
2. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
3. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool. *BMJ*. 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
4. Established machine learning matches tabular foundation models in clinical predictions. *BMC Medical Informatics and Decision Making*. 2026. https://doi.org/10.1186/s12911-026-03654-3

### Supporting materials within repository

The full 13-configuration macro and low-data leaderboards, historical v1 development comparisons, task-by-context contrasts, paired uncertainty, dataset quality audit, baseline settings, clinical reporting self-audit, source-level robustness checks and reproducibility program are retained separately in docs/paper. These materials contain the complete evidence without burdening the central narrative of this submission-facing manuscript.
