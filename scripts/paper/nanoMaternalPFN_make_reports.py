#!/usr/bin/env python3
"""Generate full measured-only statistical report, figures and Excel audit workbook."""
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension

OUT=Path(os.environ.get('NANO_STATS_OUTPUT','/mnt/data/nanoMaternalPFN_statistics'))
M={fn.stem:pd.read_csv(fn) for fn in OUT.glob('*.csv')}
QA=json.loads((OUT/'data_audit.json').read_text())
LB=M['leaderboard_all']
LL=M['leaderboard_low_data']
T=M['task_auroc_all_models']
P=M['three_expert_contrasts_all_metrics']
C=M['primary_by_context']
R=M['robustness_exclusions']
W=M['three_expert_weight_diagnostics']
D=M['v1_vs_v2_development']
E=M['efficiency_measured']
Mem=M['onnx_memory_measured']
THREE='adaptive_tiny_3expert';TAB='tabpfn_v2_default';TWO='adaptive_tiny'


def row(comparator,metric='auroc'):
    return P[(P.comparator==comparator)&(P.metric==metric)].iloc[0]

def fr(v,nd=4):
    if pd.isna(v):return '—'
    if isinstance(v,str):return v
    return f'{v:.{nd}f}'

def pm(v,nd=4):return f'{v:+.{nd}f}'

def md_table(df,formats=None,max_rows=None):
    df=df.copy()
    if max_rows:df=df.head(max_rows)
    formats=formats or {}
    cols=list(df.columns)
    head='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    for c in cols:
        if c not in formats:
            if pd.api.types.is_numeric_dtype(df[c]):
                formats[c]=lambda x: '—' if pd.isna(x) else (f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x))
            else:formats[c]=str
    for values in df.itertuples(index=False,name=None):
        head+='| '+' | '.join(formats[c](x).replace('|','/') for c,x in zip(cols,values))+' |\n'
    return head

# key results (source-cluster bootstrap and exact 7-cluster sign-flip)
vtab=row(TAB);vtwo=row(TWO);vtiny=row('tiny_generic')
Bri=row(TAB,'brier');Ece=row(TAB,'ece15');Acc=row(TAB,'accuracy')
low=C[(C.comparator==TAB)&(C.metric=='auroc')&(C.contexts=='16,32,64')].iloc[0]

# Forest chart task-level effects, explicitly descriptive.
tasks=M['primary_task_effects']
tasks=tasks[(tasks.comparator==TAB)&(tasks.metric=='auroc')].sort_values('effect')
fig,ax=plt.subplots(figsize=(10,5.2))
y=np.arange(len(tasks));ax.barh(y,tasks.effect)
ax.set_yticks(y,tasks.dataset.str.replace('_',' ',regex=False))
ax.axvline(0,linestyle='--',linewidth=1)
ax.set_xlabel('AUROC difference: three-expert Tiny minus TabPFN v2 Default')
ax.set_title('Measured AUROC differences by prediction task')
for i,v in enumerate(tasks.effect):
    ax.text(v+(0.002 if v>=0 else -0.002),i,f'{v:+.3f}',
            va='center',ha='left' if v>=0 else 'right',fontsize=9)
fig.tight_layout();fig.savefig(OUT/'task_auroc_effects.png',dpi=175);plt.close(fig)

# Primary contrast forest chart with seven-source intervals.
comparators=[TAB,TWO,'tiny_generic','xgboost','lightgbm']
scores=P[(P.metric=='auroc')&P.comparator.isin(comparators)].set_index('comparator').loc[comparators]
fig,ax=plt.subplots(figsize=(10.5,4.6))
y=np.arange(len(scores))
ax.errorbar(scores.equal_task_effect,y,xerr=np.vstack([
    scores.equal_task_effect-scores.cohort_boot_ci_lower,
    scores.cohort_boot_ci_upper-scores.equal_task_effect]),fmt='o',capsize=4)
ax.axvline(0,linewidth=1,linestyle='--')
ax.set_yticks(y,[c.replace('_',' ') for c in comparators]);ax.invert_yaxis()
ax.set_xlabel('Equal-task macro AUROC difference (positive favors three-expert Tiny)')
ax.set_title('Exploratory 95% seven-source cluster bootstrap intervals')
fig.tight_layout();fig.savefig(OUT/'cohort_auroc_effects.png',dpi=175);plt.close(fig)

# AUROC macro bar plot from all 13 current configs.
fig,ax=plt.subplots(figsize=(10.2,6))
lead=LB.sort_values('auroc',ascending=True)
ax.barh(lead.method.str.replace('_',' ',regex=False),lead.auroc)
ax.set_xlim(0,0.82)
ax.set_xlabel('Equal-task macro AUROC (all eligible context sizes)')
ax.set_title('Measured nanoMaternalPFN evaluation leaderboard')
fig.tight_layout();fig.savefig(OUT/'full_auroc_leaderboard.png',dpi=175);plt.close(fig)

leaderboard=LB[['method','auroc','brier','ece15','accuracy']].copy()
lowboard=LL[['method','auroc','brier','ece15','accuracy']].copy()

auroc_comps=P[P.metric=='auroc'][[
  'comparator','equal_task_effect','equal_source_effect',
  'cohort_boot_ci_lower','cohort_boot_ci_upper',
  'exact_source_signflip_p_two_sided','holm_p_12_competitors',
  'task_wins','source_wins'
]].sort_values('equal_task_effect')

dataset=T[['dataset',THREE,TAB,TWO,'tiny_generic']].copy()
dataset['Δ_3expert_minus_TabPFN']=dataset[THREE]-dataset[TAB]
dataset['Δ_3expert_minus_2expert']=dataset[THREE]-dataset[TWO]

contexts=C[(C.comparator==TAB)&(C.metric=='auroc')][[
  'contexts','n_tasks','n_sources','paired_draws','three_macro','other_macro',
  'equal_task_effect','cohort_ci_lower','cohort_ci_upper','exact_signflip_p_two_sided'
]]

robust=R[(R.comparator==TAB)&(R.metric=='auroc')][[
  'scenario','n_tasks','n_sources','equal_task_effect','cohort_ci_lower',
  'cohort_ci_upper','exact_signflip_p_two_sided'
]]

calibration=P[(P.comparator==TAB)&P.metric.isin(['auroc','brier','ece15','accuracy'])][[
  'metric','macro_three','macro_competitor','equal_task_effect','cohort_boot_ci_lower',
  'cohort_boot_ci_upper','exact_source_signflip_p_two_sided'
]]

runtime=E[E.method.isin([THREE,TWO,'standalone_tiny_mixture_onnx',TAB])][[
   'method','context','runtime','fit_cv_p50_ms','query_p50_ms',
   'query_p99_ms','combined_checkpoint_mb','repeats'
]]

# Check detailed draw-level bootstrap divergence, important scientific point.
draw_low=vtab.paired_context_draw_ci_lower;draw_hi=vtab.paired_context_draw_ci_upper

report=f'''# Full Statistical Analysis — nanoMaternalPFN

**Evidence freeze:** uploaded 256-file `nanoMaternalPFN-paper_review_bundle.zip`  
**Analysis date:** 13 November 2026  
**Primary comparison:** three-expert Adaptive Tiny versus official TabPFN v2 Default  
**Classification:** retrospective exploratory model benchmark; no patient-level validation or prospective decision support testing.

## 1. Executive conclusion

The three-expert nanoMaternalPFN-Tiny model **ranks first descriptively** among 13 current evaluated models by equal-task macro AUROC: **{LB.loc[LB.method==THREE,'auroc'].iloc[0]:.6f}** versus **{LB.loc[LB.method==TAB,'auroc'].iloc[0]:.6f}** for TabPFN v2 Default. The observed difference is **{pm(vtab.equal_task_effect,6)} AUROC**, or **{vtab.equal_task_effect*100:.3f} percentage points of AUROC**. This is an *observed benchmark lead*, not an independently established superiority result.

Source-aware sensitivity analysis across seven source cohorts (eight prediction tasks; both VAT outcomes kept in one cluster) yields a **95% seven-source percentile-bootstrap interval [{vtab.cohort_boot_ci_lower:+.4f}, {vtab.cohort_boot_ci_upper:+.4f}]** and an **exact two-sided source sign-flip p-value of {vtab.exact_source_signflip_p_two_sided:.6f}**. The interval spans zero, and the p-value is not evidence of superiority. It beats TabPFN on **{int(vtab.task_wins)}/8 tasks** and **{int(vtab.source_wins)}/7 source-cohort averages**. The equal-source-weighted observed difference is **{pm(vtab.equal_source_effect,6)}**, lower than the task-weighted effect because VAT provides two outcomes.

At low labeled-context sizes (16/32/64), the three-expert model leads by **{pm(low.equal_task_effect,6)} AUROC**, but its seven-source interval **[{low.cohort_ci_lower:+.4f}, {low.cohort_ci_upper:+.4f}]** also crosses zero (exact sign-flip p={low.exact_signflip_p_two_sided:.6f}).

**Correct current claim:** “Three-expert nanoMaternalPFN-Tiny achieved the highest observed macro AUROC in an exploratory eight-task benchmark, with performance heterogeneity and no established statistical superiority over TabPFN v2 Default.” This should not be stated as prospective clinical superiority, validated equivalence, or a definitive superiority claim.

## 2. Integrity, completeness and estimand

- **Manifest audit:** {QA['manifest_file_count']} SHA-256 file hashes and byte sizes all match; ZIP CRC validation succeeds. Full input ZIP SHA-256: `{QA['input_zip_sha256']}`.
- **Raw evidence:** {QA['all_raw_draw_rows_including_v1']:,} measured draw records across 17 configurations (13 current model arms plus 4 historical v1 arms). All 13 current arms have **{QA['raw_draw_rows_per_current_model']} paired draws** over **{QA['matched_context_cells_per_model']} matched dataset/context cells** (total model/metric aggregate rows = {QA['master_summary_rows']+QA['three_expert_summary_rows']:,}).
- **Replicated summaries:** every numerical summary mean recomputed from its own per-draw CSV agrees within at most **{QA['raw_to_summary_recomputed_max_abs_error']:.2e}**; no missing or mismatched draw counts.
- **Study structure:** eight prediction tasks from seven source cohorts. VAT GDM and VAT Delivery are two outcomes from one source; Pima Diabetes is the nonmaternal transfer control and UCI maternal-risk category is a risk proxy.
- **Primary reported average:** average the eligible context-draw means within each task and average the eight task scores equally. Each of the {QA['matched_context_cells_per_model']} task/context cells contributes within its own task, rather than treating the number of draws or eligible contexts as a task's relative importance.
- **Matched comparison:** every model pair is joined by exact dataset, context size, and frozen draw seed; all contrasts use matching records. No synthetic clinical performance numbers are inserted.
- **Known data limitation:** the ZIP has raw *performance metrics per context draw*, but **does not have held-out patient-by-patient predictions or true test labels**. Therefore patient-level DeLong AUROC intervals, patient-block bootstrap, calibration curves by observation and subgroup/decision-curve statistics cannot be computed from this bundle.

## 3. Entire 13-method leaderboard (all eligible contexts)

All values below are directly recomputed from measured summary and draw-level CSVs. For AUROC and accuracy, higher is better. For Brier score and ECE15, lower is better. Brier is the recorded multiclass probability score.

{md_table(leaderboard)}

**Important:** the standalone *v2 Tiny Generic* model also has higher observed macro AUROC (**{LB.loc[LB.method=='tiny_generic','auroc'].iloc[0]:.6f}**) than the TabPFN v2 Default macro (**{LB.loc[LB.method==TAB,'auroc'].iloc[0]:.6f}**). That is an observed number, not a significance claim. The three-expert Adaptive model must not inherit Tiny Generic's single-model deployment size or latency.

## 4. Low-data leaderboard (labeled contexts 16, 32 and 64)

{md_table(lowboard)}

In this subset, the three-expert model is descriptively first, but it does not establish consistent superiority in source-level resampling. All eight tasks contribute to this low-data aggregate.

## 5. Primary inference: three-expert versus TabPFN v2 Default

### 5.1 Across-source effect and formal exploratory tests

- **Mean paired AUROC effect:** {pm(vtab.equal_task_effect,6)} (three-expert minus TabPFN, equal task weight).
- **95% source-cluster percentile-bootstrap interval:** [{vtab.cohort_boot_ci_lower:+.6f}, {vtab.cohort_boot_ci_upper:+.6f}] using 100,000 replicates of seven source units, keeping VAT outcomes together.
- **Exact two-sided source-cohort sign-flip test:** p={vtab.exact_source_signflip_p_two_sided:.6f}; exact one-sided sign-flip p={vtab.exact_source_signflip_p_one_sided:.6f}, under the sign-symmetry null. These are not confirmatory because three-expert design/model selection used earlier views of these same benchmarks.
- **Paired, source-averaged t-test:** p={vtab.source_mean_ttest_p_two_sided:.6f} on only seven cohort means; **source Wilcoxon exact signed rank:** p={vtab.source_wilcoxon_p_two_sided:.6f}; **source sign test:** p={vtab.source_sign_test_p_two_sided:.6f}. These are weak, small-sample sensitivity tests, not proof of equal performance.
- **Holm familywise p-value** over 12 exploratory AUROC comparator contrasts: {vtab.holm_p_12_competitors:.6f}.
- **Task wins:** {int(vtab.task_wins)} of 8; cohort-average wins: {int(vtab.source_wins)} of 7.

### 5.2 Why 671 draws do not make the p-value small

With the patients in each frozen test pool reused across context draws, the draw is **not an independent patient or hospital cohort**. Resampling the paired context draws while fixing the same patients produces a narrower **conditional** 95% interval of **[{draw_low:+.6f}, {draw_hi:+.6f}]**. It describes sensitivity to context sampling, *not* generalization to independent cohorts. The far wider seven-source interval is much more revealing of real between-task heterogeneity, although seven convenience-source cohorts still do not provide a definitive population generalization interval. **Do not present the draw-level interval as a clinical superiority confidence interval.**

### 5.3 Dataset-level AUROC differences

{md_table(dataset,{'Δ_3expert_minus_TabPFN':lambda x:pm(x), 'Δ_3expert_minus_2expert':lambda x:pm(x)})}

The three-expert model exceeds TabPFN on Low Birth Weight, VAT Delivery, VAT GDM, and CTU Cord Acidemia, while TabPFN leads on UCI Maternal Risk, Cardiotocography, TPEHG Preterm and Pima Diabetes. The largest positive effect is Low Birth Weight ({pm(dataset.loc[dataset.dataset=='low_birth_weight','Δ_3expert_minus_TabPFN'].iloc[0])}), whereas UCI Maternal Risk is a large negative contribution. Dataset selection strongly influences the aggregate.

## 6. Sensitivity to task and cohort inclusion

{md_table(robust,{'equal_task_effect':lambda x:pm(x), 'cohort_ci_lower':lambda x:pm(x), 'cohort_ci_upper':lambda x:pm(x)})}

Excluding the shared VAT pair changes the observed lead to **{pm(robust.loc[robust.scenario=='exclude_VAT_pair','equal_task_effect'].iloc[0],6)}**. Excluding Low Birth Weight changes it to **{pm(robust.loc[robust.scenario=='exclude_low_birth_weight','equal_task_effect'].iloc[0],6)}**. Neither is a preregistered primary analysis; both are transparent influence checks. Leave-one-source-out observed macro effects range from **{vtab.leave_one_source_out_min:+.6f}** to **{vtab.leave_one_source_out_max:+.6f}**.

## 7. Context-size analysis

Each row averages only tasks eligible for that context; larger budgets evaluate fewer tasks. Do **not** interpret differences across context budgets as controlled learning curves over an unchanged set of tasks.

{md_table(contexts,{'equal_task_effect':lambda x:pm(x), 'cohort_ci_lower':lambda x:pm(x), 'cohort_ci_upper':lambda x:pm(x)})}

Three-expert Tiny leads at context 16, 32 and 64 in point estimates. It trails TabPFN at context 128, 256 and 512, but these larger-context comparisons include fewer source cohorts. No individual context-budget test establishes source-level superiority.

## 8. Four-metric assessment: discrimination, accuracy and calibration

A positive “effect” always favors three-expert Tiny; for Brier and ECE15 the effect is **comparator minus three-expert** (because lower is better). This convention prevents accidentally interpreting worse calibration as improvement.

{md_table(calibration,{'equal_task_effect':lambda x:pm(x), 'cohort_boot_ci_lower':lambda x:pm(x), 'cohort_boot_ci_upper':lambda x:pm(x)})}

TabPFN's Brier score is lower by **{abs(Bri.equal_task_effect):.6f}**, ECE15 lower by **{abs(Ece.equal_task_effect):.6f}**, and accuracy higher by **{abs(Acc.equal_task_effect):.6f}** in the observed equal-task aggregates. This means the accuracy–efficiency narrative is **multiobjective**, not “three-expert wins all endpoints.” Because source-level p-tests and bootstrap intervals have different small-sample assumptions, treat conflicting significance indicators cautiously; none of these endpoints establish confirmatory model superiority.

Against the original *two-expert* Adaptive Tiny, adding the generic expert improved observed AUROC by **{pm(vtwo.equal_task_effect,6)}** and Brier by **{pm(row(TWO,'brier').equal_task_effect,6)}**, with ECE15 changing by **{pm(row(TWO,'ece15').equal_task_effect,6)}** (positive numbers favor three-expert). This suggests a useful exploratory incremental model improvement, but the cohort-bootstrap AUROC interval crosses zero, and the AUROC exact sign-flip p-value is **{vtwo.exact_source_signflip_p_two_sided:.6f}**.

## 9. Every AUROC comparator contrast, source-aware inference and multiplicity

The following table compares three-expert Tiny against *each* of the 12 other present model configurations. Holm adjustments were applied across these 12 exploratory AUROC comparisons. Additional metrics are fully listed in `three_expert_contrasts_all_metrics.csv`; they are not treated as preregistered coprimary tests.

{md_table(auroc_comps,{'equal_task_effect':lambda x:pm(x), 'equal_source_effect':lambda x:pm(x), 'cohort_boot_ci_lower':lambda x:pm(x), 'cohort_boot_ci_upper':lambda x:pm(x)})}

**None of the 12 AUROC comparisons meets p<0.05 after Holm correction.** For example, the exploratory unadjusted exact p-value for LightGBM is {row('lightgbm').exact_source_signflip_p_two_sided:.6f}, but adjusted Holm p={row('lightgbm').holm_p_12_competitors:.6f}. A lower LightGBM macro under its recorded default settings does not justify a broad claim against tuned gradient boosting.

## 10. What the three-expert gate actually learned

The expert gates were cross-fitted using labeled-context data only. Diagnostic JSONLs include three expert weights, CV log losses, temperature, status and context seed for all **671** matched draws. Their weights sum to one within numerical precision. Expert 3 is the pretrained **v2 Tiny Generic** checkpoint; the first two are v1 and v2 Tiny Maternal-Mixture checkpoints. Average weights are descriptive and do not by themselves prove causal benefit from a prior.

{md_table(W[['dataset','generic_mean_weight','fraction_generic_top','delta_auroc_vs_two','mean_temperature']],{'delta_auroc_vs_two':lambda x:pm(x)})}

Importantly, the generic expert was **not** universally the highest-weight model on VAT Delivery, despite VAT Delivery exhibiting the biggest three-expert performance increase over original Adaptive Tiny. This corrects the earlier unsupported claim that context CV must reliably pick the generic expert on every favored dataset. The selector improves ensemble behavior through the full weighting/calibration system, not a guaranteed per-task winner choice.

## 11. v1 versus v2 longitudinal development comparisons

These v1/v2 estimates compare historical arms on the same frozen test tasks. Changes between versions were not all isolated one factor at a time; therefore the comparisons describe **development progression**, not causal experiments attributing effects specifically to Pre-LN or permutation fixes.

{md_table(D[D.metric=='auroc'][['arm','v2_improvement_positive','v2_task_wins','cohort_ci_lower','cohort_ci_upper','source_signflip_p_two_sided']],{'v2_improvement_positive':lambda x:pm(x), 'cohort_ci_lower':lambda x:pm(x), 'cohort_ci_upper':lambda x:pm(x)})}

Full four-metric v1-versus-v2 effects appear in `v1_vs_v2_development.csv`.

## 12. Measured efficiency and memory — no cross-model mixing

The new adaptive CPU profile uses **warmed PyTorch inference**, single synthetic query, 6 features, and seven repeats for each context. The EdgeBench baseline profile uses a different routine (five repeats for classical/TabPFN, 100 for standalone ONNX). As a result, measured timings can be reported but should not be characterized as a controlled, equal-protocol system-speedup ratio. Fit/context CV time is separate from query time.

{md_table(runtime)}

The three-expert combined measured PyTorch checkpoint size is **4.347998 MB** (2-expert = **2.898644 MB**). At context 64, it measured **{E[(E.method==THREE)&(E.context==64)].query_p50_ms.iloc[0]:.3f} ms** warmed CPU query p50 and **{E[(E.method==THREE)&(E.context==64)].fit_cv_p50_ms.iloc[0]:.3f} ms** context fit/CV p50. This is slower than original Adaptive Tiny's **{E[(E.method==TWO)&(E.context==64)].query_p50_ms.iloc[0]:.3f} ms** query p50. The **1.522 MB ONNX** size at context 64 belongs to the **standalone v2 Tiny Mixture model**, not the ensemble that scored 0.707268. It cannot be used as the three-expert file footprint.

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

> Across eight previously inspected frozen real-world prediction tasks derived from seven source cohorts, a context-weighted combination of three pretrained nanoMaternalPFN Tiny checkpoints achieved a descriptive equal-task macro AUROC of **{LB.loc[LB.method==THREE,'auroc'].iloc[0]:.4f}**, compared with **{LB.loc[LB.method==TAB,'auroc'].iloc[0]:.4f}** for TabPFN v2 Default. The paired mean difference was **{vtab.equal_task_effect:+.4f}**. A seven-source cluster bootstrap yielded an exploratory 95% interval of **[{vtab.cohort_boot_ci_lower:+.4f}, {vtab.cohort_boot_ci_upper:+.4f}]**, and an exact two-sided source sign-flip sensitivity test produced **p={vtab.exact_source_signflip_p_two_sided:.3f}**. Three-expert Tiny led on four of eight tasks. Although it had the highest observed AUROC among the tested configurations, its superiority to TabPFN v2 was **not statistically established** under source-aware sensitivity analyses. TabPFN had better aggregate Brier score, ECE15 and accuracy. The analysis is exploratory because benchmark results had informed iterative model development.

## 15. Files and reproducibility

- `nanoMaternalPFN_full_statistics.py`: analysis implementation; same seed and 100,000-replicate source bootstrap for reproducibility.
- `leaderboard_all.csv`, `leaderboard_low_data.csv`, `task_means_all_metrics.csv`, `task_auroc_all_models.csv`.
- `three_expert_contrasts_all_metrics.csv`: 48 paired comparisons, each with source-cluster intervals, exact sign-flip p-values, Welch-free source t-test, Wilcoxon, task wins and familywise adjustments.
- `primary_task_effects.csv`, `primary_context_effects.csv`, `primary_by_context.csv`, `robustness_exclusions.csv`.
- `three_expert_weights_raw.csv`, `three_expert_weight_diagnostics.csv`, `v1_vs_v2_development.csv`.
- `efficiency_measured.csv`, `onnx_memory_measured.csv`, `raw_summary_consistency.csv`, `integrity_manifest_audit.csv`, `draw_counts.csv`, `data_audit.json`.
- `nanoMaternalPFN_Statistical_Results.xlsx` — curated, labeled, formatted analysis workbook.

**Citable origin:** user-supplied `nanoMaternalPFN-paper_review_bundle.zip`; exact SHA-256 above. Relevant development code: [nanoMaternalPFN research branch](https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/tree/research/gemini-g1-g20-verified-paper-20261113).
'''
report=report.replace(chr(16),'')
(OUT/'nanoMaternalPFN_Full_Statistical_Report.md').write_text(report)

# Excel workbook: imported empirical values green; formulas/derived differences black;
# metadata grey and warnings orange. Simple formulas are kept in Excel so readers
# can recompute directly; more complex bootstrap output is imported and sourced.
WB=Workbook();intro=WB.active;intro.title='Read Me'
fill_head=PatternFill('solid',fgColor='202B36')
fill_sub=PatternFill('solid',fgColor='EAF0F3')
fill_alt=PatternFill('solid',fgColor='F8FAFB')
fill_caution=PatternFill('solid',fgColor='FFF3DD')
font_head=Font(name='Aptos',size=10,bold=True,color='FFFFFF')
font_input=Font(name='Aptos',size=10,color='13734B')
font_formula=Font(name='Aptos',size=10,color='202124')
font_static=Font(name='Aptos',size=10,color='525960')
font_warn=Font(name='Aptos',size=10,color='924C12')

intro.sheet_view.showGridLines=False
intro.merge_cells('A1:H2');intro['A1']='nanoMaternalPFN | Statistical Audit';intro['A1'].font=Font(name='Aptos Display',size=20,bold=True,color='FFFFFF');intro['A1'].fill=fill_head;intro['A1'].alignment=Alignment(vertical='center')
for rr in (1,2):intro.row_dimensions[rr].height=23
intro.merge_cells('A4:H4');intro['A4']='Exploratory measured evidence • 13 current model arms • 8 tasks • 7 source cohorts';intro['A4'].font=Font(size=11,bold=True)
intro['A6']='Three-expert Tiny macro AUROC';intro['B6']='=\'Leaderboard All\'!B2'
intro['A7']='TabPFN v2 Default macro AUROC';intro['B7']='=\'Leaderboard All\'!B4'
intro['A8']='Difference (three-expert minus TabPFN)';intro['B8']='=B6-B7'
intro['A9']='Source-cohort sign-flip p-value';intro['B9']=float(vtab.exact_source_signflip_p_two_sided)
intro['A10']='Source-cluster bootstrap 95% lower';intro['B10']=float(vtab.cohort_boot_ci_lower)
intro['A11']='Source-cluster bootstrap 95% upper';intro['B11']=float(vtab.cohort_boot_ci_upper)
intro['A12']='Macro AUROC rank';intro['B12']=1
for i in range(6,13):
    intro.cell(i,1).font=font_static
    intro.cell(i,2).font=font_formula if i<=8 else font_input
    intro.cell(i,2).number_format='0.000000'
intro.merge_cells('A15:H15');intro['A15']='INTERPRETATION';intro['A15'].fill=fill_head;intro['A15'].font=font_head
intro.merge_cells('A16:H18');intro['A16']='Highest observed macro AUROC, but not statistically established as superior to TabPFN v2 Default. Exact two-sided p=0.578125; source-bootstrap interval crosses zero. Do not use patient-level interpretations of repeated context draws.';intro['A16'].alignment=Alignment(wrap_text=True,vertical='center');intro['A16'].fill=fill_caution;intro['A16'].font=font_warn
for rr in (16,17,18):intro.row_dimensions[rr].height=22
intro.merge_cells('A21:H21');intro['A21']='SOURCE AND REPRODUCIBILITY';intro['A21'].fill=fill_head;intro['A21'].font=font_head
intro['A22']='Original evidence ZIP SHA-256';intro['B22']=QA['input_zip_sha256']
intro['A23']='Manifests verified';intro['B23']=QA['manifest_file_count']
intro['A24']='Raw data records';intro['B24']=QA['all_raw_draw_rows_including_v1']
intro['A25']='Raw rows per current model';intro['B25']=671
intro['A26']='Independent source cohorts';intro['B26']=7
intro['A27']='Source / code';intro['B27']='https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/tree/research/gemini-g1-g20-verified-paper-20261113';intro['B27'].hyperlink=intro['B27'].value;intro['B27'].style='Hyperlink'
for i in range(22,28):intro.cell(i,1).font=font_static
intro.column_dimensions['A'].width=43
intro.column_dimensions['B'].width=31
for x in 'CDEFGH':intro.column_dimensions[x].width=14

book_tables=[
 ('Leaderboard All',LB),('Leaderboard Low',LL),
 ('Task AUROC',dataset),('Primary Inference',P[P.comparator.isin([TAB,TWO,'tiny_generic'])]),
 ('All Comparisons',P),('By Context',contexts),('Robustness',R),
 ('Expert Weights',W),('Development',D),('Efficiency',E),
 ('ONNX Memory',Mem),('Audit Counts',M['draw_counts']),
]

def fmt_cell(c,v):
    if isinstance(v,(int,np.integer)) and not isinstance(v,bool):c.number_format='#,##0'
    elif isinstance(v,(float,np.floating)):
        c.number_format='0.000000' if abs(v)<1 else '0.0000'
    elif isinstance(v,bool): c.number_format='General'
    else:c.number_format='General'

for title,df in book_tables:
    ws=WB.create_sheet(title)
    ws.sheet_view.showGridLines=False
    ws.freeze_panes='B2'
    for col_idx,col in enumerate(df.columns,1):
        c=ws.cell(1,col_idx,str(col))
        c.fill=fill_head;c.font=font_head
        c.alignment=Alignment(wrap_text=True,vertical='center')
        width=min(max(len(str(col))+5,14),41)
        ws.column_dimensions[get_column_letter(col_idx)].width=width
    ws.row_dimensions[1].height=35
    for row_idx,values in enumerate(df.itertuples(index=False,name=None),2):
        for col_idx,val in enumerate(values,1):
            if pd.isna(val):v=None
            elif isinstance(val,np.generic):v=val.item()
            else:v=val
            c=ws.cell(row_idx,col_idx,v)
            c.font=font_input if isinstance(v,(int,float,np.integer,np.floating)) else font_static
            if row_idx%2==0:c.fill=fill_alt
            c.alignment=Alignment(vertical='center',wrap_text=False)
            if v is not None:fmt_cell(c,v)
    ws.auto_filter.ref=f'A1:{get_column_letter(len(df.columns))}{len(df)+1}'

# Formulas in the task sheet explicitly recompute differences from measured source values.
ws=WB['Task AUROC']; header={c.value:c.column for c in ws[1]}
for i in range(2,ws.max_row+1):
    for dest,first,second in [
        ('Δ_3expert_minus_TabPFN',THREE,TAB),('Δ_3expert_minus_2expert',THREE,TWO)]:
        dest_cell=ws.cell(i,header[dest]);dest_cell.value=f'={get_column_letter(header[first])}{i}-{get_column_letter(header[second])}{i}'
        dest_cell.font=font_formula;dest_cell.number_format='+0.000000;-0.000000;0.000000'

# Formula recalculation policy: Excel will compute on load. Source stats precomputed
# via audited NumPy/scipy script; bootstrap cannot be represented as a simple
# spreadsheet formula without obscuring the statistical procedure.
try:
    WB.calculation.fullCalcOnLoad=True
    WB.calculation.forceFullCalc=True
    WB.calculation.calcMode='auto'
except Exception:
    pass
WB.properties.title='nanoMaternalPFN Complete Statistics — Observed Evidence'
WB.properties.description='Derived exclusively from immutable uploaded ZIP; 100k source-cohort resamples'
BOOK=OUT/'nanoMaternalPFN_Statistical_Results.xlsx'
WB.save(BOOK)
# Workbook integrity and formula/linked-input visual check.
book=load_workbook(BOOK,read_only=False,data_only=False)
assert len(book.sheetnames)==13,book.sheetnames
assert book['Read Me']['B8'].value=='=B6-B7'
assert book['Task AUROC']['F2'].value.startswith('=')
assert book['Leaderboard All'].max_row==14
assert len(report)>14000
print('Generated report characters',len(report),'spreadsheet worksheets',book.sheetnames)
print('files',len(list(OUT.iterdir())))
