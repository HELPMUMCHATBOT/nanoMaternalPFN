# Clinical reporting and risk-of-bias evidence matrix

**Scope:** TRIPOD+AI and PROBAST+AI *self-audit* against the actual frozen benchmark. This is not a completed item-by-item clinical review, validated clinical AI product, institutional ethics sign-off, or outside assessor endorsement. Status labels distinguish **Verified in archive**, **Documented with limitations**, and **Not measurable**.

| Domain | Status | Frozen evidence or limitation |
| --- | --- | --- |
| Research objective | Verified in archive | Retrospective synthetic-to-real comparison on eight tasks; model development partly informed by previous benchmark inspection |
| Intended use | Documented with limitations | Methodological benchmarking; not a patient-facing deployment recommendation |
| Study setting and population | Documented with limitations | Seven publicly sourced cohorts, heterogeneous endpoints; VAT GDM/delivery participants overlap; nonmaternal Pima control |
| Predictors and outcome definitions | Documented with limitations | Described in `docs/DATASET_PORTFOLIO.md`; UCI risk proxy and CTU cord pH classification not equivalent to adjudicated clinical events |
| Participant eligibility, exclusions | Not measurable | Frozen summary ZIP lacks the complete original patient/eligibility audit; manifests and scripts document sampling procedure, not an independent clinical source review |
| Handling of missing values | Documented with limitations | Model and baseline preprocessing recorded; no patient-level per-predictor missingness distribution in this analysis ZIP |
| Sample size and event counts | Not measurable | Source dataset counts in portfolio, but outcome-level independent event counts and denominator auditing not independently derivable from this ZIP |
| Data leakage safeguards | Documented with limitations | Frozen duplicate-safe context sampling and synthetic-only pretraining documented; previously viewed real-test results informed model selection |
| Validation design | Verified in archive | Frozen context draws, same real holdouts; not independent confirmatory external validation of later model choices |
| Performance discrimination | Verified in archive | AUROC, matched draws and seven-source exploratory sensitivity intervals |
| Accuracy | Verified in archive | Full raw and summary accuracy |
| Calibration scores | Verified in archive | Multiclass Brier and confidence ECE15 only |
| Calibration intercept/slope, curves | Not measurable | Patient-level probabilities absent; must not be invented from Brier/ECE |
| Subgroup fairness and missingness strata | Not measurable | Patient-level protected/clinical subgroup descriptors and predictions absent |
| Clinically useful thresholds, decision curve | Not measurable | No threshold policy, patient-level predictions or clinical action/outcome data |
| Statistical uncertainty | Documented with limitations | Paired context-draw matching and seven-source bootstrap/exact sign flips; only seven convenience source clusters, non-confirmatory |
| Comparator optimization | Documented with limitations | Classical defaults; no nested hyperparameter optimization; LightGBM small-sample behavior may reflect defaults |
| External applicability | Not established | Convenience public cohorts, many proxies, no independent new prospective or untouched confirmation |
| Deployment resource evidence | Verified with restrictions | Standalone ONNX and adaptive multi-expert CPU timings under different procedures; no Android/INT8/energy |
| Model licensing | Requires review | Official TabPFN model-weight license and source dataset permissions must be checked by authors before release |
| Ethics and consent | Requires sign-off | Access to public datasets does not itself establish local ethics/consent determination |
| Authorship, conflicts, funding | Requires sign-off | Not in quantitative archive; leave blank until provided |

**Assessment:** Risk of bias from adaptive benchmark reuse and incomplete patient-level quality evidence is material. Broad clinical applicability cannot be assumed. The manuscript should call its analysis *exploratory and retrospective* and quote the real source-cluster interval, not promote "significant clinical superiority". TRIPOD+AI: Collins et al., *BMJ* 2024, e078378; PROBAST+AI: Moons et al., *BMJ* 2025, e082505.
