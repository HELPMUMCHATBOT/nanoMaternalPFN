# Model Card — nanoMaternalPFN

**Research-only prototype, not clinically validated.** Do not use for diagnosis, triage, treatment, or individual patient advice.

## Purpose
Study lightweight in-context tabular classification in low-data maternal-health settings using synthetic priors.

## Limitations
Synthetic data-generating assumptions are not observed epidemiology. No quantitative clinical predictive performance is established, and the reference memory chart is not a hardware measurement from this codebase. Real datasets must be licensed, verified, and split with duplicate-profile separation. No patient data or governed datasets are included.

## Planned evaluation
AUROC, Brier score, ECE, subgroup metrics, independent holdouts, CPU and mobile latency, memory, energy and calibration under shift.
