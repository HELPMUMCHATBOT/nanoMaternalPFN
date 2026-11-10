# nanoMaternalPFN — Priority 1: evidence and acquisition register

**Revision:** 2026-11-10. **Policy:** No governed/government datasets, no IRB/DUA waiting, no pilot stage, and no invented "real" data. Synthetic tables are used **only for PFN pretraining**. This page records externally established dataset *sources*, not claimed completed acquisitions.

## Core open-access candidate tasks

| Task | Exact source | Cohort | Use | Caveat |
| --- | --- | --- | --- | --- |
| UCI maternal-risk categories | [UCI 863](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) | Rural Bangladesh IoT risk table (~1,013) | Proxy-label sanity check | Extensive duplicate feature profiles; categories are not independently adjudicated prospective outcomes |
| GDM | [PhysioNet maternal VAT 1.0.0](https://physionet.org/content/maternal-visceral-adipose/1.0.0/) | Brazil, 133 pregnancies | Prospective gestational diabetes prediction | Small event count, potential diagnostic ascertainment bias |
| Cesarean delivery | [Same PhysioNet VAT cohort](https://physionet.org/content/maternal-visceral-adipose/1.0.0/) | **Same 133 pregnancies** | Separate delivery-type endpoint | **Not an independent cohort**; group correctly for cross-task inference |
| Low infant birth weight | [MASS birthwt](https://rdrr.io/cran/MASS/man/birthwt.html) | 189 records, Baystate Medical Center | Birthweight outcome | Never include birthweight-in-grams (`bwt`) as a predictor of its derived low-birthweight label |
| CTG fetal status | [UCI 193](https://archive.ics.uci.edu/dataset/193/cardiotocography) | 2,126 records | Obstetric/fetal secondary outcome | Remove `CLASS` morphological label and `NSP` outcome from input features |
| Pima female diabetes | [OpenML 37](https://www.openml.org/d/37) | 768 records | Nonmaternal metabolic transfer control | Not gestational diabetes or a pregnancy dataset; zero-coded measurements require missingness treatment |

**Important:** Six tasks above correspond to **five cohorts**, not six independent maternal cohorts. Pima is a nonmaternal control, UCI risk labels are proxy categories, and CTG assesses fetal state. Model results must be stratified by target type and cohort. Count independent cohorts rather than all task endpoints for statistical tests.

## Additional public candidates (NOT yet an approved seventh dataset)

- [Term-Preterm EHG Database, PhysioNet](https://physionet.org/content/tpehgdb/1.0.1/): 300 pregnancies. Published record-level clinical metadata and signal-derived features are available, but the extractor must exclude delivery gestation, `Group`, `Premature`, or any target-derived column from predictors. Non-tabular signal extraction requires a separate validated workflow.
- [UCI Caesarian Section Classification](https://archive.ics.uci.edu/dataset/472/caesarian+section+classification+dataset): 80 pregnancy records. Too small for several planned 128–512 context budgets; methodological and endpoint review required.
- [Maternal ultrasound and nutritional assessment, PhysioNet](https://physionet.org/content/maternal-ultrasound-nutrition/1.0.0/): **credentialed access**; excluded under the no-wait rule.
- Proposed 3,525-record GDM, 850-record obstetric triage, and 1,205-record 11-feature maternal-morbidity datasets mentioned in the original blueprint: exact downloadable records, licensing, and labeling are **unverified**. Do **not** generate synthetic lookalikes and call them clinical cohorts.
- NICHD nuMoM2b, NDHS and governed HelpMum records: excluded from this stage by user preference and governance/access requirements.

## Acquisition and data-quality controls

Run `python -m pip install -e ".[data,test]"`, then:

```bash
python scripts/01_acquire_open_sources.py
cat artifacts/manifests/open_portfolio_report.json
```

The report marks each dataset as `frozen` only after the actual external source is fetched, its schema and class count pass checks, and `freeze_splits.py` writes source-separated, group-safe splits. Any failure is recorded as `unavailable`, **never silently synthesized**. Unavailable records must be resolved before the benchmark can be called complete.

Files in `artifacts/manifests/` contain source metadata, context draws and SHA256 of processed NPZ. Store raw patient-level data outside public Git. Validate any source/license restriction before sharing raw or transformed records.

## Pretraining separation

No real dataset is read by `hdf5_replay.py` or `train_replay.py`. Full training uses **generic prior versus 70% maternal/30% generic mixture**, Tiny and Base, and variable rows/features; the maximum total row budget must exceed 512 to allow **512 context rows plus query rows**. Freeze real holdouts before inspecting predictive results.

## Independently sourced cohort expansion — 10 November 2026

Real acquisition succeeded for two additional open PhysioNet cohorts in [GitHub Actions run 42752532407](https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/actions/runs/42752532407):

| Cohort | Records acquired | Distinct feature profiles | Label conflicts among repeated profiles | Outcome |
| --- | ---: | ---: | ---: | --- |
| TPEHG, Ljubljana | 300 | 299 | 0 | Preterm birth from independently recorded delivery gestation |
| CTU-UHB, Brno | 552 | 176 | 36 | Exploratory cord arterial pH below 7.15 |

Both cohorts are independently collected from those in the original six-task set. Original acquisition yields **8 tasks over 7 distinct cohorts**, but Pima is a *non-maternal control*, VAT GDM/delivery are *two outcomes on the same cohort*, and CTU-UHB outcome is neonatal rather than maternal. These are not eight independent maternal predictive cohorts.

**Methodological correction:** after examining CTU-UHB, context sampling was changed to require one distinct feature profile per context example, not merely a distinct row index. The new group-safe unique-context-v2 split artifacts must be regenerated and their digests recorded before transfer benchmarking. This can remove 512-context eligibility from UCI 863 and 256-context eligibility from CTU-UHB. The previous 8-task acquisition artifacts prove source accessibility but are superseded for final context sampling.

See [the new source adapters](../src/nanomaternalpfn/data/physionet_headers.py), [AWS guide](SAGEMAKER_G5_AND_OPEN_COHORTS.md), and [split implementation](../src/nanomaternalpfn/data/freeze_splits.py).
