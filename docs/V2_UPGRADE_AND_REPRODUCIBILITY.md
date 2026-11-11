# nanoMaternalPFN v2: Unified research implementation

**Status:** Experimental v2 training / inference research implementation.
**Scientific status:** UNVALIDATED. Prior v1 test sets were already examined, so
any v2 improvement measured there is exploratory and not an independent
confirmation or a guarantee of superiority. This is not for clinical use.

## Changes

- Both `generic` and 70%-maternal `mixture` synthetic prior archives now
  support v2 class-index randomization and task-level sign-flip augmentation.
  `create_replay(..., variant='v2')` produces a new schema-v3 HDF5 archive
  with deterministic task seed streams and explicit augmentation provenance.
  v1 archive format and outputs remain readable.
- The `pre_ln_v2` architecture uses Pre-LN in feature attention, patient
  attention and MLP, plus **affine-free** final LayerNorm before the head.
  Affine-free normalization preserves the *number of trainable parameters*;
  it does not preserve the function of preexisting v1 weights. Old v1 models
  keep `post_ln_v1` as their default config and load with strict state dict.
  Feature columns have no positional embeddings, so feature permutation is
  already a one-pass invariant in exact arithmetic, apart from FP roundoff.
- Tiny v2 uses constant LR 5e-4; Base v2 uses 2e-4 peak LR, 250-step linear
  warmup then cosine to 5% peak at the final 5,000-step update. Resume rejects
  a changed schedule length. Training and synthetic validation archives must
  have matching variants.
- `NanoMaternalPFNClassifier` offers explicit
  `symmetrize_classes=True` with **batched** `C!` class permutations, not
  one-pass inference. Class-augmented training does **not** mathematically
  guarantee single-pass equivariance or any latency target.
- `calibrate_context=True` fits 2 global parameters (temperature and a
  smoothed context-prior log-odds strength) by stratified 4-fold predictions
  within the labeled context only. It needs no query distribution and can
  predict one patient at a time. Calibration is skipped when folds cannot
  cover classes. It is not proven to improve Brier/ECE, and extra CV forward
  passes at fit time increase startup cost. It should be considered
  exploratory, especially for 16-row contexts.
- **v1 original evaluations remain unchanged.** The new `--checkpoint-dir`
  and `--output-dir` arguments in the existing study runner let users
  evaluate v2 under new names. Summary tables use the original multiclass
  Brier-score convention (sum of squared class errors).

## SageMaker

Verify all tests before submitting. The default command is a DRY RUN.

```bash
cd ~/nanoMaternalPFN
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[test,data]'
pytest -q

python scripts/08_submit_sagemaker.py --mode train --variant v2 \
  --steps 5000 --batch-size 32 --microbatch-size 8 \
  --validate-every 250 --validation-tables 512
```

When satisfied with the dry run and the AWS IAM identity, add `--submit`
to launch a **billable** SageMaker `ml.g5.2xlarge` job. Its job name starts
`nanomaternalpfn-v2-`; S3 prefix defaults to
`s3://sagemaker-us-east-1-355119164071/nanomaternalpfn/v2`.
It is intentionally separate from the v1 model archive. This repo change
does not automatically launch cloud compute. The prior archives generated
on SageMaker scratch storage are not automatically uploaded to S3.

After training, download and extract `model.tar.gz` from the new job into
`artifacts/results/aws_v2_pretrain/`, preserving the four
`*.best.inference.pt` files. Verify hashes and synthetic provenance.

## Real-data evaluation

Pilot on all eight real tasks (one frozen draw of 16 rows per eligible cohort):
```bash
python scripts/09_run_downstream_study.py \
  --checkpoint-dir artifacts/results/aws_v2_pretrain \
  --output-dir artifacts/results/downstream_v2_pilot
```

Full 20-draw 2x2 comparison (same previously frozen v1 holdouts;
**post-hoc comparison**, not independent clinical validation):
```bash
python scripts/09_run_downstream_study.py --full \
  --checkpoint-dir artifacts/results/aws_v2_pretrain \
  --output-dir artifacts/results/downstream_v2_full
```

Optional methods should each have distinct result directories, e.g.:

```bash
python scripts/09_run_downstream_study.py --full \
  --checkpoint-dir artifacts/results/aws_v2_pretrain \
  --symmetrize-classes \
  --output-dir artifacts/results/downstream_v2_symmetrized

python scripts/09_run_downstream_study.py --full \
  --checkpoint-dir artifacts/results/aws_v2_pretrain \
  --calibrate-context \
  --output-dir artifacts/results/downstream_v2_calibrated
```

**Avoid promising 24x near-zero overhead:** batched class symmetrization
can be faster than sequential calls on GPU, but still processes C! times
as many contexts and may require more CPU/GPU RAM. Report actual latency.

## Edge

```bash
python -m pip install -e '.[edge]'
python scripts/12_profile_v2_edge.py \
  --checkpoint artifacts/results/aws_v2_pretrain/tiny_mixture.best.inference.pt \
  --contexts 16 64 --features 6 --repeats 100 --onnx
```

Use this for observed **macOS/Linux CPU** PyTorch/ONNX latency and graph
size. Do not report Android latency, energy, INT8 peak memory or
clinical efficacy from this profiler. It benchmarks a single patient and
does not include a context CV calibration fit nor class-symmetrized ensemble.

## Study limitations

- Known labels of the v1 datasets informed the proposed changes. A fresh,
  externally sourced locked cohort is needed for a confirmatory v2 claim.
- VAT delivery and VAT GDM share one maternal cohort. UCI maternal risk
  is a risk-label proxy. Pima diabetes is a nonmaternal control.
- Class symmetrization is exact only when all C! permutations are used;
  training-time permutation augmentation only encourages equivariance.
- Calibration can make AUROC/Brier/ECE worse and may have high variance
  on low-prevalence outcomes. Always compare identity, calibrated, and
  symmetrized options, with no test-set method selection.
- 5,000 optimization steps and a runtime expectation do not guarantee
  convergence, improved performance, or a particular AWS cost.
