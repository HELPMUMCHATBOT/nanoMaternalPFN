# nanoMaternalPFN C1–C6 research implementation

**Branch:** `research/c1-c6-improvements` (kept separate from `main`).

This branch contains six **implemented experimental directions**, not six proven
clinical improvements. Existing eight-task paper findings must not be
retroactively relabeled as results of these new implementations.

## C1: Context-only expert ensembles

`src/nanomaternalpfn/adaptive_classifier.py` supports 2–8 frozen checkpoints,
with softmax log-loss weighting, winner selection, and uniform controls.
The selector cross-fits **only labeled context samples**; if a small context
cannot be stratified, it uses uniform weights. Query labels are never read.

For frozen real-data measurements (with checkpoints and data staged):

```bash
python scripts/18_run_complete_benchmark.py \
  --input-dir artifacts/results/paper_full_study \
  --output-dir artifacts/results/c1_expert_comparison \
  --methods adaptive_tiny adaptive_tiny_three adaptive_tiny_four \
  --device cpu
```

Check `scripts/18_run_complete_benchmark.py --help` for the required
input structure. The output must be a **new directory**; do not overwrite
published paper experiment results. These newly selected variants are exploratory.

## C2: Robust synthetic prior

`v3_robust` introduces stochastic clinical-shape hypotheses:
rare binary outcomes, nonlinear feature interactions, synthetic hospital shifts,
covariate-dependent missingness, heavy-tailed measurement corruptions, and label
noise. These are **simulation assumptions**, not clinically established causal
effects. Original `v1` and `v2` generation are unchanged.

```bash
python -m nanomaternalpfn.priors.hdf5_replay \
  --path artifacts/priors_h5/robust_mix70.h5 \
  --variant v3_robust --prior mixture \
  --maternal-probability 0.7 --num-tables 200000
```

## C3: Tiny feature representations

`ModelConfig(feature_encoder_type="gated", normalization="robust")`
selects a shared learnable feature gate and context-only median/MAD scaling.
The defaults (`linear`, `standard`) keep old checkpoint behavior. Changes
are encoded into checkpoint configs and can be tested as separate ablations.

## C4: Teacher-to-Tiny distillation

`src/nanomaternalpfn/distillation.py` and
`scripts/24_distill_tiny.py` train a single Tiny student using frozen PFN
teacher soft targets plus synthetic ground-truth labels. Report the
student's *own* AUROC, file size, latency, and memory; never borrow teacher
performance or edge measurements.

```bash
python scripts/24_distill_tiny.py \
  --replay artifacts/priors_h5/robust_mix70.h5 \
  --validation-replay artifacts/priors_h5/robust_val.h5 \
  --teachers /path/to/v1_tiny_mixture.best.inference.pt /path/to/v2_tiny_mixture.best.inference.pt \
  --output artifacts/results/c4_distilled_tiny.inference.pt \
  --steps 2500 --batch-size 32 --microbatch-size 8 --device cuda
```

For any experiment, validation replay must come from a **different seed**.
The distillation report includes actual completed steps, replay hashes,
teacher hashes and held-out synthetic-validation history.

## C5: Training-budget study and C6: maternal/generic mixture sweep

The single runner `scripts/23_run_c1_c6_research.py` combines controlled
step budgets with synthetic mixture proportions and C2/C3 configurations.
It writes a dry-run `research_plan.json` by default; **`--execute`
really trains the models** and writes `measured_results.json`.

Preview the full matrix:

```bash
python scripts/23_run_c1_c6_research.py --device cuda
```

Run the prespecified comparison on an adequately provisioned GPU instance:

```bash
python scripts/23_run_c1_c6_research.py \
  --execute --device cuda \
  --variants v2 v3_robust \
  --maternal-fractions 0 .3 .5 .7 1 \
  --step-budgets 2500 5000 10000 25000 \
  --encoders linear gated \
  --normalizations standard robust \
  --tiers tiny \
  --batch-size 32 --microbatch-size 8
```

The full default factorial matrix contains 80 independent model trainings
(2 variants × 5 mixtures × 4 budgets × 2 encoders × 2 normalizers).
It is computationally substantial. Each run has an independent checkpoint,
fresh optimizer trajectory, measured time and held-out synthetic losses.
The script refuses to overwrite an existing checkpoint. A budget comparison
does **not** imply a given improvement in clinical performance.

For only a functional smoke test (not a scientific result), choose one
combination, 1 optimizer step, 20 tables, and a separate output directory:

```bash
python scripts/23_run_c1_c6_research.py \
  --execute --smoke --device cpu \
  --output artifacts/results/c1_c6_smoke \
  --variants v3_robust --maternal-fractions .7 \
  --step-budgets 1 --encoders gated --normalizations robust \
  --num-tables 20 --validation-tables 20 \
  --batch-size 2 --microbatch-size 1 --validate-every 1
```

## Validation, attribution and reporting rules

1. Store every run's config, replay SHA-256, random seed, optimizer budget,
   measured time, loss trajectory, and inference checkpoint.
2. Use the original paper's frozen context draws for **exploratory comparison**
   only, because v1/v2 results previously influenced model development.
3. Distinguish eight prediction tasks from seven distinct source cohorts,
   including shared VAT GDM/Delivery participants.
4. Complete frozen-run paired comparisons with uncertainty intervals; never
   fabricate score improvements. Small-context CV chooses expert weights by
   log-loss, not test AUROC.
5. Explicitly distinguish single Tiny, multi-checkpoint adaptive and distilled
   Tiny compute, memory, disk size, and latency.
6. For a new confirmatory superiority claim, assess a subsequently locked
   independent evaluation source or separate prespecified confirmation design.
7. Running tests/CI establishes **software correctness and synthetic smoke
   metrics**, not real-world clinical superiority or statistical equivalence.

## Implementation map

| Direction | Implementation | Clinical-result status |
|---|---|---|
| C1 | Adaptive multi-expert classifier; frozen benchmark methods | Not yet measured for new variants |
| C2 | Versioned `v3_robust` HDF5 generator | Synthetic-only |
| C3 | Gated Tiny feature encoder + robust context normalization | Not yet clinically measured |
| C4 | Frozen-teacher distillation and standalone student checkpoint | Not yet clinically measured |
| C5 | Independent budgeted training with synthetic validation | Not yet clinically measured |
| C6 | Prior-mixture factorial comparisons (0–100% maternal) | Not yet clinically measured |
