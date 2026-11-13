#!/usr/bin/env python3
"""Assemble the evidence-grounded manuscript from recalculated numerical tables.

Requires the outputs of run_paper_evidence.py. Missing measurements remain
explicitly missing; this program does not fit, simulate or impute any model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def md(df, *, decimals=4):
    columns=list(df.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for rec in df.itertuples(index=False,name=None):
        cells=[]
        for item in rec:
            if pd.isna(item): cells.append('—')
            elif isinstance(item,float): cells.append(f'{item:.{decimals}f}')
            else: cells.append(str(item).replace('|','/'))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def write_manuscript(source: Path, target: Path):
    req=['leaderboard_all.csv','leaderboard_low_data.csv', 'three_expert_contrasts_all_metrics.csv',
         'primary_task_effects.csv','v1_vs_v2_development.csv','efficiency_measured.csv',
         'primary_by_context.csv','robustness_exclusions.csv', 'data_audit.json']
    missing=[s for s in req if not (source/s).is_file()]
    if missing: raise FileNotFoundError('Cannot publish unmeasured manuscript: '+str(missing))
    all_=pd.read_csv(source/'leaderboard_all.csv').sort_values('auroc',ascending=False)
    low=pd.read_csv(source/'leaderboard_low_data.csv').sort_values('auroc',ascending=False)
    p=pd.read_csv(source/'three_expert_contrasts_all_metrics.csv')
    task=pd.read_csv(source/'primary_task_effects.csv')
    v1=pd.read_csv(source/'v1_vs_v2_development.csv')
    edge=pd.read_csv(source/'efficiency_measured.csv')
    context=pd.read_csv(source/'primary_by_context.csv')
    robust=pd.read_csv(source/'robustness_exclusions.csv')
    audit=json.loads((source/'data_audit.json').read_text())
    assert audit['manifest_sha256_verified'] and audit['raw_to_summary_n_mismatches']==0
    assert audit['patient_level_predictions_in_zip'] is False
    assert len(all_)==13 and audit['master_summary_rows']==1632
    t='adaptive_tiny_3expert';b='tabpfn_v2_default'
    comp=p[(p.comparator==b)&(p.metric=='auroc')].iloc[0]
    brier=p[(p.comparator==b)&(p.metric=='brier')].iloc[0]
    ece=p[(p.comparator==b)&(p.metric=='ece15')].iloc[0]
    other=p[(p.comparator=='adaptive_tiny')&(p.metric=='auroc')].iloc[0]
    lo=context[(context.comparator==b)&(context.metric=='auroc')&(context.contexts=='16,32,64')].iloc[0]
    ttask=task[(task.comparator==b)&(task.metric=='auroc')][['dataset','effect']].sort_values('dataset')
    ttask=ttask.rename(columns={'effect':'AUROC difference (3-expert − TabPFN v2)'})
    v1=v1[v1.metric=='auroc'][['arm','v2_improvement_positive','v2_task_wins','cohort_ci_lower','cohort_ci_upper']]
    ed=edge[edge.context.isin([16,64,128])][['method','context','runtime','fit_cv_p50_ms','query_p50_ms','query_p99_ms','combined_checkpoint_mb','repeats']]
    by_context=context[(context.comparator==b)&(context.metric=='auroc')][['contexts','n_tasks','n_sources','three_macro','other_macro','equal_task_effect']]
    sens=robust[(robust.comparator==b)&(robust.metric=='auroc')][['scenario','n_tasks','equal_task_effect','cohort_ci_lower','cohort_ci_upper']]
    m=[]
    def emit(s=''): m.append(s)
    emit('# nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction')
    emit('')
    emit('**Research manuscript — evidence-linked draft, 13 November 2026.**  ')
    emit('**Authors, affiliations, ethics determination and funding declarations: to be confirmed by the team.**  ')
    emit('**Study classification:** retrospective exploratory synthetic-to-real benchmark; not a validated clinical decision-support device.')
    emit('')
    emit('## Abstract')
    emit(f'''**Background:** Clinical prediction from small heterogeneous tabular datasets is challenging, and deployable models may face tight resource limits. **Objective:** To characterize discrimination, calibration and computation of compact prior-data fitted networks pretrained on synthetic tables and evaluated on obstetric and related tasks. **Methods:** We analyzed eight frozen real-world prediction tasks from seven source cohorts using {audit['raw_draw_rows_per_current_model']} matched context-draw evaluations per current model. Comparators comprised logistic regression, random forest, XGBoost, LightGBM, official TabPFN v2 single and default configurations, standalone nanoMaternalPFN variants, and two- or three-expert context-adaptive variants. Metrics were AUROC, accuracy, multiclass Brier and 15-bin expected calibration error. Paired AUROC differences were averaged within context and task and assessed using exploratory source-cluster resampling that kept both VAT outcomes together. **Results:** Three-expert Tiny achieved the highest observed equal-task macro AUROC ({float(all_.set_index('method').loc[t,'auroc']):.4f}) compared with TabPFN v2 Default ({float(all_.set_index('method').loc[b,'auroc']):.4f}); the paired difference was {float(comp.equal_task_effect):+.4f} (95% seven-source bootstrap sensitivity interval {float(comp.cohort_boot_ci_lower):+.4f} to {float(comp.cohort_boot_ci_upper):+.4f}; exact sign-flip p={float(comp.exact_source_signflip_p_two_sided):.3f}). TabPFN performed better on aggregate Brier and ECE15. **Conclusion:** The measured benchmark suggests a promising compact-model accuracy–resource trade-off, but it does not establish independent clinical superiority, equivalence, calibration advantage or deployment readiness.''')
    emit('')
    emit('## 1. Introduction')
    emit('Clinical risk prediction frequently uses tabular measurements assembled from relatively small cohorts, often with missing values, inconsistent feature availability and shifting outcome definitions. Foundation-style approaches pretrained on synthetic tasks may reduce the need to fit neural-network weights on every new clinical dataset, but their accuracy, calibration and runtime should be assessed separately. A small model is not necessarily clinically reliable, and its resource requirements must correspond to the actual predictor evaluated.')
    emit('')
    emit('Prior-data fitted networks (PFNs) learn to infer class probabilities conditional on labeled examples sampled from prior-generated tasks. TabPFN demonstrated the general potential of this strategy on small tabular datasets [1]. In maternal and fetal prediction research, small sample sizes and endpoint diversity make performance heterogeneity particularly important. Our study examines nanoMaternalPFN, a compact PFN family with generic and maternal-informed synthetic pretraining; the goal is to report what the locked benchmark actually measured rather than infer clinical utility from test-set rankings.')
    emit('')
    emit('The contributions are: (i) eight-task, seven-source frozen-context evaluation; (ii) a transparent longitudinal comparison of historical v1 and current v2 models; (iii) a context-only two- and three-expert adaptive analysis; (iv) paired source-aware uncertainty and calibration assessment; and (v) separately recorded single-model ONNX and multi-expert CPU runtime results. Every statistical claim here is traceable to the frozen ZIP and audit scripts.')
    emit('')
    emit('## 2. Materials and methods')
    emit('### 2.1 Design and sources')
    emit('The study retrospectively benchmarks eight already-inspected open prediction tasks. The two VAT outcomes derive from the same 133-pregnancy source cohort; they are distinct tasks but not independent participant populations. The Pima Diabetes dataset is a non-maternal metabolic transfer control. The UCI maternal-risk label is a categorized proxy rather than a prospectively adjudicated morbidity endpoint. Additional outcome-specific cautions and source URLs are recorded in `docs/DATASET_PORTFOLIO.md`.')
    emit('')
    emit('**Evaluated tasks:** UCI Maternal Health Risk; PhysioNet VAT gestational diabetes; PhysioNet VAT delivery outcome; MASS low birth weight; UCI cardiotocography; Pima Diabetes; PhysioNet term/preterm electrohysterography; and CTU-UHB cord acidemia. The CTU-UHB threshold in the repository dataset adapter is a cord-pH classification proxy, not a prospective clinical treatment endpoint.')
    emit('')
    emit('### 2.2 Models and synthetic pretraining')
    emit('nanoMaternalPFN processes tabular feature and missingness information using feature-oriented and context-restricted patient attention, and predicts query labels from a labeled context. Training comprises generic synthetic prior tables and a maternal-mixture prior (70% maternal-shaped / 30% generic in expectation). The codebase distinguishes post-LN v1 and pre-LN v2 variants; v2 also changed class-index augmentation and training configuration. These historical changes occurred together, so the v1-to-v2 comparison is descriptive development history, **not** a controlled single-component causal ablation.')
    emit('')
    emit('Adaptive Tiny combines the historical v1 Tiny Maternal-Mixture and v2 Tiny Maternal-Mixture checkpoints. The evaluated three-expert variant adds v2 Tiny Generic. Its expert weights and optional temperature are fitted by cross-validation on **labeled context only**; no query/test label is used for within-task adaptation. However, the *choice to test this three-expert configuration* was informed by previously inspected benchmark results, making its reported performance exploratory and subject to adaptive benchmark selection.')
    emit('')
    emit('![Architecture traced from the pre-LN v2 model source](figures/architecture_source.svg)')
    emit('')
    emit('### 2.3 Benchmark comparators and frozen splits')
    emit('The 13 current evaluated configurations are listed in Table 1. Classical baselines use the repository’s recorded defaults, not a clinical hyperparameter-optimization campaign: Logistic Regression (imputation, scaling, max_iter=1000), Random Forest (200 trees), XGBoost (200 estimators; histogram method) and LightGBM (200 estimators, otherwise library defaults). Official TabPFN v2 is explicitly selected with `ModelVersion.V2`, either one estimator or its default ensemble configuration. Predictive comparisons use the same frozen context/test indices and seeded draws where valid. Ineligible class-deficient contexts are recorded and uniformly omitted across methods.')
    emit('')
    emit(f'''The evidence archive contains {audit['all_raw_draw_rows_including_v1']:,} draw-level metric records over 17 historical/current configurations and {audit['matched_context_cells_per_model']} matched dataset/context cells per current method. There are {audit['raw_draw_rows_per_current_model']} matched valid draws per current configuration. The primary summary comprises {audit['master_summary_rows']:,} rows for the original 12-current-method design; the additional three-expert arm contributes {audit['three_expert_summary_rows']} rows. The ZIP also contains four historical v1 arms.''')
    emit('')
    emit('### 2.4 Endpoints and estimand')
    emit('The metrics are AUROC (binary or one-vs-rest multiclass according to the recorded evaluation routine), accuracy, multiclass Brier score (sum over class probabilities, averaged over records) and 15-bin confidence-based expected calibration error (ECE15). Higher is better for AUROC/accuracy; lower is better for Brier/ECE15. Neither Brier nor ECE15 is a patient-level calibration curve, and neither directly estimates clinical benefit.')
    emit('')
    emit('For each method, we first average valid draws at each dataset/context cell, then average eligible contexts within each task, then average the eight task means equally. The availability of larger contexts differs by dataset; cross-context comparisons are therefore not fixed-cohort learning curves. For pairwise results we join by the exact dataset, context and seed before taking differences. The directional effect is always positive if the three-expert model is better; for Brier/ECE15 the difference is reversed.')
    emit('')
    emit('### 2.5 Uncertainty and multiplicity')
    emit('Uncertainty analyses were **exploratory sensitivities** because model design and expert-pool selection followed earlier examination of these eight datasets. We resampled seven source groups with replacement (100,000 replicates), keeping both VAT endpoints together and preserving task weights. We also enumerated all 2^7 source-group sign flips under a symmetry assumption; p-values are descriptive, not independent confirmatory evidence. Twelve AUROC comparator tests were Holm adjusted. Repeated context draws re-use held-out patients, so a within-draw bootstrap cannot substitute for patient or source-level external uncertainty. No equivalence or noninferiority margin was prespecified.')
    emit('')
    emit('### 2.6 Model efficiency and hardware')
    emit('The paper reports two different profiled objects: a **standalone v2 Tiny Maternal-Mixture ONNX graph** (100 CPU query repetitions at each measured context) and warmed **PyTorch adaptive multi-expert predictors** (seven repetitions plus a separately reported context cross-validation fit cost). Baseline CPU timing was collected under a further routine with five repetitions. Cross-protocol ratios are not claimed as controlled head-to-head speedups. File size, combined checkpoint size, query time, context-fit time, total process RSS and incremental RSS remain separately identified. No Android, INT8 energy, mobile device RAM or prospective clinical performance is inferred.')
    emit('')
    emit('## 3. Results')
    emit('### 3.1 Source integrity')
    emit(f'''The evidence ZIP SHA-256 is `{audit['input_zip_sha256']}`. All {audit['manifest_file_count']} manifest-listed members passed byte-count and SHA-256 verification. Recalculated raw-draw means matched archived summaries with maximum absolute discrepancy {audit['raw_to_summary_recomputed_max_abs_error']:.2e}; there were {audit['raw_to_summary_n_mismatches']} draw-count mismatches. These checks verify the archive’s internal integrity, not independent patient-data provenance or model-weight authenticity.''')
    emit('')
    emit('### 3.2 Primary discrimination and calibration')
    emit('**Table 1. Full measured macro leaderboard (equal task weight; eligible contexts).**')
    emit(md(all_[['method','auroc','brier','ece15','accuracy']]))
    emit('')
    emit('![Measured AUROC leaderboard](figures/full_auroc_leaderboard.svg)')
    emit('')
    emit(f'''The three-expert model ranked first by observed macro AUROC ({float(all_.set_index('method').loc[t,'auroc']):.4f}) and led official TabPFN v2 Default ({float(all_.set_index('method').loc[b,'auroc']):.4f}) by {float(comp.equal_task_effect):+.4f}. The seven-source percentile bootstrap interval was [{float(comp.cohort_boot_ci_lower):+.4f}, {float(comp.cohort_boot_ci_upper):+.4f}]; the exact two-sided source sign-flip result was p={float(comp.exact_source_signflip_p_two_sided):.3f}. The interval spans zero. Three-expert Tiny won {int(comp.task_wins)}/{int(comp.tasks)} tasks and {int(comp.source_wins)}/{int(comp.sources)} source averages. No superiority or practical equivalence is established.''')
    emit('')
    emit(f'''TabPFN v2 Default showed better observed aggregate calibration: three-expert Tiny Brier {float(all_.set_index('method').loc[t,'brier']):.4f} vs TabPFN {float(all_.set_index('method').loc[b,'brier']):.4f}, and ECE15 {float(all_.set_index('method').loc[t,'ece15']):.4f} vs {float(all_.set_index('method').loc[b,'ece15']):.4f}. The three-expert Brier directional effect was {float(brier.equal_task_effect):+.4f}, and ECE15 was {float(ece.equal_task_effect):+.4f} (positive would favor three-expert). Thus, AUROC is **not** an all-metric victory.''')
    emit('')
    emit('### 3.3 Low-data performance and heterogeneity')
    emit('**Table 2. Macro leaderboard restricted to labeled contexts 16, 32 and 64.**')
    emit(md(low[['method','auroc','brier','ece15','accuracy']]))
    emit('')
    emit(f'''Across those three context budgets, the three-expert model had observed AUROC {float(lo.three_macro):.4f} compared with TabPFN v2 Default {float(lo.other_macro):.4f}, a paired difference of {float(lo.equal_task_effect):+.4f} with source interval [{float(lo.cohort_ci_lower):+.4f}, {float(lo.cohort_ci_upper):+.4f}] and exact sign-flip p={float(lo.exact_signflip_p_two_sided):.3f}. This is descriptive low-data performance, not an independent significance result.''')
    emit('')
    emit('**Table 3. Paired task-level AUROC differences for three-expert Tiny vs TabPFN v2 Default.**')
    emit(md(ttask))
    emit('')
    emit('![Measured dataset-level AUROC differences](figures/task_auroc_effects.svg)')
    emit('')
    emit('**Table 4. By-context differences; larger context budgets evaluate fewer tasks.**')
    emit(md(by_context))
    emit('')
    emit('### 3.4 Sensitivity analysis and multiple comparisons')
    emit('**Table 5. Influential inclusion/exclusion choices (all post hoc, not alternative primary outcomes).**')
    emit(md(sens))
    emit('')
    emit('Excluding the VAT pair or the Low Birth Weight task changes the observed average lead to a slight deficit. This is evidence of dataset heterogeneity and argues against an unconditional generalization claim. Across twelve exploratory AUROC comparator contrasts, no comparison has Holm-adjusted p below 0.05. The source-cluster intervals are based on only seven convenience-source units and should be interpreted cautiously.')
    emit('')
    emit('![Exploratory source-cluster bootstrap contrasts](figures/cohort_auroc_effects.svg)')
    emit('')
    emit('### 3.5 v1/v2 historical development and adaptive increment')
    emit('**Table 6. v2 minus v1 measured macro AUROC by architectural/prior arm.**')
    emit(md(v1))
    emit('')
    emit(f'''Adding the third expert improved observed macro AUROC relative to the original two-expert Adaptive Tiny by {float(other.equal_task_effect):+.4f}; however the cohort-bootstrap interval [{float(other.cohort_boot_ci_lower):+.4f}, {float(other.cohort_boot_ci_upper):+.4f}] crosses zero. The mixed architecture/prior changes between v1 and v2 mean that their longitudinal difference cannot uniquely establish the benefit of Pre-LN or class-permutation augmentation.''')
    emit('')
    emit('### 3.6 Deployment measurements')
    emit('**Table 7. Measured, differently configured CPU efficiency results; no cross-object footprint substitution.**')
    emit(md(ed,decimals=3))
    emit('')
    emit('At context 64, the three-expert PyTorch CPU system measured 17.760 ms median query time, 130.579 ms median context-CV fit time, and 4.348 MB combined checkpoint files. The standalone v2 Tiny Mixture ONNX graph measured 2.649 ms median query time and 1.522 MB graph size under a different implementation and repeat protocol. These properties belong to **different predictors** and must not be combined into one deployment point. The standalone ONNX process at context 64 reached 71.359 MB total process peak RSS, with 24.359 MB incremental peak over import baseline; it does not meet a 32 MB **total process** threshold. Adaptive multi-expert RSS was not measured in this archive.')
    emit('')
    emit('## 4. Discussion')
    emit('### 4.1 Interpretation')
    emit('The benchmark suggests that a compact PFN family pretrained on synthetic tables can offer competitive discrimination under small labeled contexts. The three-expert adaptive model leads the tested methods by the reported point estimate, but its superiority over TabPFN v2 is unproven in source-aware sensitivity analyses. In particular, TabPFN offers better aggregate Brier, ECE15 and accuracy, and the three-expert improvement is sensitive to which tasks are included. The standalone Tiny variant is a more directly resource-conscious deployment option, but its lower parameter/graph size is not evidence of the adaptive system’s same efficiency.')
    emit('')
    emit('### 4.2 Relation to existing work')
    emit('The general PFN approach builds on TabPFN [1], which evaluates pretrained inference on heterogeneous tabular datasets. Our contribution is the documented maternal-informed synthetic-prior research program, low-context transfer and accurate resource accounting. Because benchmark-specific model choices were informed by earlier results, our comparisons should be regarded as exploratory internal/external transfer benchmarks rather than fresh external confirmation. These results motivate independent clinical and deployment studies, but those are outside the current measured evidence.')
    emit('')
    emit('### 4.3 Baseline fairness')
    emit('Under the recorded configurations, XGBoost and LightGBM trailed the best PFN approaches on equal-task macro AUROC, whereas Random Forest and Logistic Regression remained important strong baselines on several tasks. In contexts below the LightGBM default minimum child-sample threshold, tree splitting can be constrained; no fully tuned LightGBM or nested-tuned boosted-tree comparison was performed. The comparison is therefore against **the recorded configurations**, not a universal statement about gradient-boosted tree families.')
    emit('')
    emit('### 4.4 Clinical and methodological limitations')
    emit('The frozen archive lacks patient-level predictions and outcomes, precluding honest calculation of DeLong covariance, patient-level calibration curves, calibration intercept/slope, subgroup fairness measures, patient-level uncertainty and decision-curve clinical net benefit. The seven source cohorts are heterogeneous convenience samples rather than a representative sample of maternal health settings; the Pima control is nonmaternal, UCI risk uses proxy labels, and VAT outcomes are shared. Some datasets contain repeated feature profiles and limited independent contexts. Retrospective public-table performance does not establish future patient outcomes, causal benefit, transportability or safety in clinical workflows.')
    emit('')
    emit('The three-expert idea and earlier v2 refinements were informed by previously inspected results, so formal tests are post hoc and vulnerable to benchmark-selection bias. The small number of source clusters makes bootstrap tails unstable and sign-flip tests coarse. Device timing does not include Android, cold-start downloads, energy, INT8 execution, quantized peak memory or clinical deployment conditions. Some timing routines used different repeat counts and interfaces. No claims about these unmeasured properties are made.')
    emit('')
    emit('### 4.5 Reporting standards')
    emit('This draft is aligned with the principles of TRIPOD+AI [2] and acknowledges the PROBAST+AI [3] domains. It is **not** a completed item-level external risk-of-bias adjudication. Missing data on eligibility decisions, patient-level class counts/prevalence, missingness rates, independent endpoint adjudication and subgroup results are flagged in `CLINICAL_REPORTING_AUDIT.md` rather than invented. No prospective risk-calibrated clinical decision threshold is proposed.')
    emit('')
    emit('## 5. Conclusion')
    emit('On eight previously inspected real-world prediction tasks from seven cohorts, three-expert nanoMaternalPFN Tiny produced the top observed macro AUROC among the evaluated methods, but source-aware statistical uncertainty did not establish superiority over official TabPFN v2 Default. Calibration was worse on the reported aggregate, and the three-expert system required more resources than the standalone Tiny ONNX graph. These findings support an exploratory accuracy–efficiency research contribution and motivate transparent independent clinical and hardware evaluations, not immediate patient-facing deployment.')
    emit('')
    emit('## Data, code and reproducibility')
    emit('The measurement source is the exact frozen `nanoMaternalPFN-paper_review_bundle.zip` whose SHA-256 is stated in §3.1. The full evidence archive contains no patient-level probability arrays. Audited CSVs, programmatic figures, source verification, source-cluster statistical routines and the full statistical report are organized under `docs/paper/` and `scripts/paper/`. Run `scripts/paper/run_paper_evidence.py --bundle PATH` to regenerate and verify; do not commit identifiable patient records or model credentials.')
    emit('')
    emit('## Declarations requiring author sign-off')
    emit('**Authors/contributions:** pending author confirmation.  **Funding:** not supplied in measured archive; author confirmation required.  **Ethics determination and data-use permissions:** verify source licenses and institutional requirements; no determination is asserted here.  **Conflicts of interest:** declarations pending.  **AI/model license constraints:** verify downstream licensing before distributing weights.  **Clinical regulatory status:** research only.')
    emit('')
    emit('## References')
    emit('[1] Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature*. 2025;637:319–326. https://doi.org/10.1038/s41586-024-08328-6')
    emit('')
    emit('[2] Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378')
    emit('')
    emit('[3] Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool. *BMJ*. 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505')
    emit('')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(m)+'\n',encoding='utf-8')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(write_manuscript(a.source,a.output))

if __name__=='__main__': main()
