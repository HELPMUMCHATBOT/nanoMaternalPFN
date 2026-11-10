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
