# Reproducible Experiment Protocol

## Study design
Compare Tiny and Base architectures trained on generic versus maternal synthetic prior: a 2 × 2 factorial ablation. Assess low-data transfer using context sizes 16, 32, 64, 128, 256, 512 where supported by independent holdouts, with 20 seeds. Prespecify excluded budgets for small datasets.

## Data
Use open, source-verified real maternal/obstetric datasets only. External provenance, permission, schema, outcome definition, patient-level duplication and contradictory labels must be recorded. Group identical feature profiles before holdout creation. Never insert synthetic fallback rows into real-data benchmarks.

## Training
Record random seeds, configurations, hardware, framework versions, number of prior tables, optimizer, steps, and artifact hashes. All synthetic pretraining is disjoint from real-world test sets.

## Evaluation
Report AUROC, Brier, ECE, calibration slopes, confidence intervals, baselines and subgroup performance. No clinical deployment claim from these experiments. Keep the UCI rule-derived maternal risk label clearly distinguished from clinical outcomes.

## Edge
Benchmark real hardware, separate FP32 and INT8 memory and runtime, and distinguish warmed context caches from full recomputation. The reference charts are hypotheses until independently measured. Document any missing ONNX, ExecuTorch, hardware-energy or KV-caching implementation as pending.

## November 2026 preregistered synthetic pretraining schedule

The four arms (Tiny/Base × generic/maternal mixture) use one NVIDIA A10G, a
maximum of **5,000 optimizer steps each**, and **effective batch size 32**,
implemented as four gradient-accumulated microbatches of 8.
At 5,000 optimizer steps this is 160,000 sampled synthetic table presentations
per model, not 160,000 unique tables. There are two 200,000-table training
prior archives; the mixture archive is 70% maternal-shaped in expectation,
not 200,000 maternal-only tasks.

**Convergence monitoring:** Generate a separate 512-table synthetic validation
archive per prior, using seed 20271008 (training seed 20261107).
Evaluate every 250 optimizer steps and at the final step, with the same
fixed seeded draws across evaluations: 2 validation batches of 8 tables per
eligible context budget (16, 32, 64, 128, 256, 512). Record the mean held-out
synthetic cross-entropy for each budget and the equally weighted mean across
budgets. Keep the best-validation-loss model and the final model, plus
intermediate inference snapshots. The full 5,000-step budget is run unless
interrupted; selecting the best checkpoint is not early stopping and does not
guarantee convergence. All four arms receive the same maximum budget.
Validation metrics must never be described as clinical transfer results.
Clinical evaluation on independently sourced held-out real-world cohorts
must follow pretraining.

The A10G benchmark previously conducted measured **native batch 8 only** and
excluded I/O and synthetic archive generation. Gradient-accumulated batch 32,
held-out evaluation costs, checkpoint synchronization and the full 5,000-step
study have not yet been benchmarked end-to-end. SageMaker may terminate the job
at its configured maximum runtime; partial arms then require a resumed job.
