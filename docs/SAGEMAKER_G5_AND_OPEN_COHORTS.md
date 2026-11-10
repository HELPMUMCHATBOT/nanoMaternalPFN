# Independent cohort expansion and SageMaker A10G guide

## New accessible cohort candidates

- TPEHG (University Medical Centre Ljubljana): 300 pregnancies, 38 preterm deliveries. Preterm-vs-term outcome is computed from delivery gestation, which is never given to the predictor. Only recording-time clinical header fields are used. Source: https://physionet.org/content/tpehgdb/1.0.1/
- CTU-UHB CTG (University Hospital Brno): 552 deliveries. Exploratory cord pH below 7.15 classification using *pre-outcome maternal characteristics only*. This threshold is a modeling convention, not a validated clinical diagnosis. Delivery type, neonatal variables, pH, and outcome-proxy fields are excluded. Source: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

These are distinct new cohorts, not copies of existing datasets. Overall: 8 candidate tasks from 7 distinct cohorts, but Pima is a nonmaternal control, VAT contributes two related tasks, and CTU-UHB predicts a neonatal biochemical outcome. Do not call these seven independent maternal clinical validations. Both new adapters require completed and passing live acquisition manifests before inclusion in reported results.

## Acquire

    python -m pip install -e '.[data,test]'
    python scripts/01_acquire_open_sources.py
    cat artifacts/manifests/open_portfolio_report.json

Only public unsigned PhysioNet S3 clinical header metadata is retrieved, not waveforms. Missing sources must be reported unavailable, never synthesized.

## SageMaker ml.g5.2xlarge

Configuration: single NVIDIA A10G with 24 GB GPU VRAM, 8 vCPUs, and 32 GiB RAM. Intended full study: 200,000 synthetic tasks per prior (generic and 70/30 maternal/generic mixture), four model trainings (Tiny/Base x two priors), and 2,500 steps *per model* (10,000 total). Effective batch size 32 (four accumulated microbatches of 8); 16–512 labeled context rows plus 8 queries. Current replay trainer is FP32 by default, without mixed-precision optimization.

Planning estimate, not measured on A10G: allow roughly 6–18 hours for the four sequential GPU runs and generation, with a 24-hour first-run allocation for inefficiency, context-dependent memory, and restarts. In local CPU-only testing, 1,000 prior tables were generated in about 1.5–1.6 seconds, and extrapolation suggests around 10–30 minutes for 400,000 tasks (subject to AWS throughput and I/O). Approximate HDF5 storage: 6 GB for two full priors, but verify actual usage.

For AWS US East pricing, roughly USD 1.69 per ml.g5.2xlarge hour is an *indicative* SageMaker training rate, not a quote for the user's region or plan. Under this rate a 6–18-hour run is about USD 10–31 compute; 24 hours is about USD 41, plus storage and additional charges.

## Benchmark the actual GPU before full study

    python scripts/07_benchmark_sagemaker_a10g.py --timed 10 --warmup 3 --contexts 16 128 512 --tiers tiny base --batch-size 8

The benchmark times an actual forward, backward, optimizer step and reports CUDA peak memory; it excludes HDF5 I/O and full-run overhead. If Base at 512 contexts is out-of-memory, lower the batch and record that deviation before training.

    python scripts/06_pretrain_study.py --num-tables 200000 --steps 2500 --batch-size 32 --device cuda

Estimate training hours as [2500*(2*tiny_weighted_seconds+2*base_weighted_seconds)]/3600 plus prior generation, data loading, initialization, and checkpointing. Save all final checkpoints, prior archives and their hashes to durable storage; a SageMaker instance shutdown must not erase the results.

Reference: official AWS SageMaker AI instance/pricing pages and original PhysioNet cohort documentation, accessed 2026-11-10. No clinical efficacy or completed full-training claim is made.
## November 2026 training batch-size decision

The **full 2×2 pretraining** now uses an **effective batch of 32 synthetic tables per optimizer step**, implemented as **four microbatches of 8** with gradient accumulation and a single optimizer update. At 2,500 steps, this is 80,000 task presentations per model (with replacement). Both priors contain 200,000 generated tables; the 2,500-step protocol does not guarantee exposure to every unique table. Class and context budgets are sampled per microbatch. Gradient clipping happens once after accumulation and prior to the optimizer step. The initial A10G benchmark job previously submitted with native batch 8 is unchanged, and is not a complete benchmark for accumulated batch 32 throughput. The final experiment needs a measured end-to-end training throughput check.

After pulling the latest main branch on the Mac:

```bash
export AWS_PROFILE=helpmum-training
python scripts/08_submit_sagemaker.py --mode train
# Add --submit only when ready to start the billable full job.
```

The SageMaker entrypoint recreates synthetic HDF5 archives on instance scratch using the recorded seed and produces provenance, checkpoints, and output artifacts in S3. The original complete locally generated HDF5 files are not automatically uploaded. If reproducibility requires byte-identical archives, compare the regenerated task content and library versions rather than assuming cross-environment HDF5 hashes will match.
