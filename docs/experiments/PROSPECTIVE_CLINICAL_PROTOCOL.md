# E12: prospective clinical utility study — draft preregistration and governance protocol

**Status: study design ONLY. No participants recruited or patient outcomes measured.**

**Objective:** Assess whether a *fixed and previously validated* nanoMaternalPFN model can safely improve a defined maternal-health clinical workflow compared with existing care. This is not a blanket diagnostic device claim.

**Population and setting:** Define participating hospitals, recruitment dates, inclusion/exclusion criteria, target clinical population, gestational age at prediction, language/access needs, and referral availability before recruitment. Use maternal endpoints with clinically established definitions; the UCI risk proxy is not itself a clinical endpoint. Define what information is known at the prediction time and exclude future-derived features.

**Design:** First run a silent prospective observational validation with unchanged care and blinded model outputs to determine discrimination, calibration, error rate, referral burden and missingness. If acceptable and independently approved, consider a separate controlled implementation study with a formally specified human-in-the-loop decision pathway and appropriate randomization or phased implementation. Do not silently transition from silent testing to patient-facing action.

**Primary outcomes:** Predefine the clinically meaningful maternal/perinatal outcome, forecast horizon and threshold. Report event denominators, discrimination uncertainty, calibration-in-the-large, calibration slope, sensitivity/specificity, positive and negative predictive value, net benefit, and patient-level bootstrap intervals. Define subgroup strata and minimum event counts before analysis. Do not promise positive clinical effects.

**Safety:** Obtain clinical ethics/IRB determination and any necessary approvals, data protection impact assessment, consent or waiver, data access agreements, device/regulatory review where applicable, clinician oversight, adverse-event escalation and stopping rules. Maintain a model freeze and protected independent monitoring. Models must not independently decide treatment, discharge or emergency referral.

**Data governance:** Encrypt identifiable clinical records; limit access on least-privilege basis; do not publish patient-level features, identifiers or raw prediction arrays in a public GitHub repository. Retain site/participant IDs for cluster-aware testing only in approved secure storage.

**Analysis:** Prespecify sample size using expected outcome event prevalence and confidence-width/clinical decision requirements, not an invented participant count. Lock the algorithm, threshold, model weights and analysis plan before unblinding outcomes. Account for hospital clustering, missingness and temporal drift. Distinguish algorithmic discrimination from actual clinical benefit.

**Required sign-offs before study initiation:** clinical PI, institutional ethics body, hospital data controllers, applicable regulators, trained clinicians and participating-site technical leads. No sign-off is implied by this document.
