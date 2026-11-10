# Priority 1 and 2 — Recorded execution evidence (10 November 2026)

This status file is a **historical snapshot of verified work**. It is not a clinical validation report or claim that large-scale GPU pretraining completed.

## Priority 1 — public, non-government datasets

Real source acquisition succeeded in GitHub Actions [Run #42749190357](https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/actions/runs/42749190357), with a downloadable `open-sources-audit` artifact containing the original prepared-source metadata and frozen split manifests. All six requested adapters returned `frozen`. No fake clinical datasets were substituted.

| Task | Fetched rows | Distinct feature profiles | Duplicate rows relative to profiles | Profiles with contradictory outcomes | Eligible 20-seed context budgets |
| --- | ---: | ---: | ---: | ---: | --- |
| UCI Maternal Health Risk | 1,014 | 416 | 598 | 35 | 16, 32, 64, 128, 256, 512 |
| PhysioNet VAT gestational diabetes | 133 | 133 | 0 | 0 | 16, 32, 64 |
| PhysioNet VAT delivery type | 133 | 133 | 0 | 0 | 16, 32, 64 |
| MASS low birthweight | 189 | 183 | 6 | 0 | 16, 32, 64, 128 |
| UCI cardiotocography | 2,126 | 2,111 | 15 | 2 | 16, 32, 64, 128, 256, 512 |
| OpenML Pima diabetes | 768 | 768 | 0 | 0 | 16, 32, 64, 128, 256, 512 |

**Interpretation:** 6 tasks are derived from 5 distinct source cohorts. VAT GDM and delivery share the same 133 women, Pima diabetes is nonmaternal, CTG is a fetal endpoint, and UCI maternal-risk is a proxy risk categorization. These must **not** be claimed as six independent maternal clinical validations. The original blueprint's 3,525-record GDM, 850-record obstetric triage, and 1,205-record maternal risk cohorts are not included without verifiable provenance.

For UCI maternal-risk, 416 unique *feature profiles* differs from counts of distinct feature+label record combinations; keeping all conflicting outcomes in the same split does not silently resolve them. Frozen manifests and the audit workflow are the source of truth; the source download job itself keeps raw/processed arrays out of Git.

**Independent repeatability:** an earlier acquisition workflow [Run #42749169288](https://github.com/HELPMUMCHATBOT/nanoMaternalPFN/actions/runs/42749169288) fetched four tasks. On the later run, all four of their frozen NPZ SHA256 digests, holdout indices, and context draws matched exactly. The earlier run lacked the optional UCI dependency, since fixed in the later run.

## Priority 2 — synthetic pretraining engineering

The code now supports deterministic **variable-shape** 2/3-class HDF5 prior generation, partial-dump detection, completion markers, 16–512 context tasks (maximum 576 rows), online minibatch sampling from a completed HDF5 prior, resumable optimizer/RNG state, and separate safe inference checkpoints.

**Executed CPU-only software checks (NOT paper results):**

| Demo run | Synthetic prior tasks in file | Executed training steps | Description |
| --- | ---: | ---: | --- |
| Tiny maternal mixture | 256 | 30 | CPU smoke checkpoint |
| Tiny generic | 256 | 30 | CPU smoke checkpoint |
| Base maternal mixture | 256 | 5 | CPU smoke checkpoint |
| Base generic | 256 | 5 | CPU smoke checkpoint |

The planned scientific experiment remains **200,000 synthetic tasks per prior, Tiny and Base × generic and 70/30 maternal mixture, and 2,500 optimizer steps per checkpoint**, followed by independent cohort-level evaluation. **No 200,000-task generation, full-scale training, AUROC improvement or clinical validation has been claimed.** A GPU machine with adequate disk and runtime is required for this full study; the current local environment was CPU-only. CPU smoke model weights and source-split metadata are available separately as a non-public evidence archive for reproduction.

## Next factual gates

1. Inspect and approve the endpoint/provenance choices; obtain more **independent** verified maternal cohorts if a seven-source portfolio is essential.
2. Execute the complete GPU training regimen with real logs/checkpoint hashes, not illustrative estimates. Export all four completed weights and source SHA256s.
3. Run the locked independent test comparisons; report model discrimination/calibration with uncertainty and group-aware statistical analysis.

See [dataset portfolio](DATASET_PORTFOLIO.md), [study config](../configs/pretraining/study_2x2.yaml), [HDF5 replay](../src/nanomaternalpfn/priors/hdf5_replay.py), and [replay trainer](../src/nanomaternalpfn/train_replay.py).
