# Complete AWS GPU paper benchmark

**Title:** nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction

**One SageMaker A10G job. Eighteen methods. Eight frozen real datasets.**
No local GPU compute is required. The MacBook stages files and submits the job.
This uses the existing AWS account, \`us-east-1\`, \`ml.g5.2xlarge\` (A10G
24 GB), and IAM SageMaker execution role.

## Methods (13, all required)

- Logistic Regression, Random Forest, XGBoost, LightGBM
- **nanoTabPFN (generic and maternal-mixture prior):** pinned upstream architecture at automl/nanoTabPFN
  commit \`6071eec58a1a3225854480125001d5a62e279f6d\`
  (Apache-2.0; preserved verbatim and attributed), trained **from scratch**
  on our independently generated **generic and maternal-mixture v2 synthetic priors**.
  200,000 synthetic tasks, 5,000 optimization steps, effective batch 32.
  It uses the original alternating feature/patient attention architecture,
  with context-only missing-value imputation necessary for NaNs.
  This is a comparable retrained model, NOT an official off-the-shelf
  pretrained nanoTabPFN.
- Official Prior Labs TabPFN V2, one estimator and version default ensemble
  (explicit \`ModelVersion.V2\`; tabpfn 9.2.0 pinned).
- nanoMaternalPFN v1 and v2 Tiny/Base x Generic/Maternal Mixture (eight arms).
- Context-only v1/v2 maternal expert ensemble at Tiny and Base tiers.

## Primary evaluation

Same 8 source-verified frozen manifests: UCI maternal risk, VAT GDM,
VAT delivery, low birth weight, cardiotocography, Pima diabetes,
TPEHG preterm, CTU cord acidemia.

Same frozen context budgets \`16,32,64,128,256,512\` where eligible, with
all 20 prespecified draws per eligible budget. Skip only contexts with
one observed class, identically for every method. Compute AUROC, accuracy,
multiclass Brier and ECE15. Report mean and SD across valid draws for
each dataset/context/method; preserve individual predictions' metrics in
\`all_draws.csv\`. Log fit times, full test-pool predict times, GPU memory
reserved and allocated peaks on the A10G. No fabricated successes.

The v1/v2 comparison is exploratory because existing downstream holdouts
informed changes. VAT GDM and delivery share a cohort. Pima is a
nonmaternal control and UCI maternal risk is a proxy. Do not drop CTU
or TPEHG based on results.

## Run

~~~bash
cd ~/nanoMaternalPFN
source .venv/bin/activate
git pull --ff-only
python -m pip install 'sagemaker>=2.230,<3' boto3
export AWS_PROFILE=helpmum-training
export AWS_DEFAULT_REGION=us-east-1
python scripts/19_submit_full_aws_benchmark.py --submit
~~~

The command verifies and bundles **8 × JSON/NPZ frozen datasets**,
**4 v1** and **4 v2** pretrained weights. It rejects missing files
and SHA256 mismatches. It uploads only those 24 inputs to S3 and only the
minimal source bundle (not \`.venv\`, not raw data folders, not other artifacts).

Job uses \`ml.g5.2xlarge\`, 80 GB EBS, and 48-hour requested max duration.
The SageMaker job incurs billed GPU and storage costs as long as it runs.
An official TabPFN v2 checkpoint downloads during the job; AWS network
access must be available. Ensure Prior Labs v2 model-weight license permits
your research use. The vendored nanoTabPFN code is Apache-2.0; its original
model training depends on upstream separately distributed prior files, but
this comparison trains on our own documented synthetic generic prior.

The pretraining model and evaluation raw output are written to
\`/opt/ml/checkpoints\` and synced to the provided
\`checkpoint_s3_uri\`. Subsequent jobs with the same prefix can resume
completed nanoTabPFN optimizer steps and method/context draws without
changing frozen inputs. Completed reports are also packaged in the
SageMaker \`model.tar.gz\` output archive.

### Benchmark outputs

- \`paper_benchmark/all_draws.csv\`: one row per valid frozen draw/method/context,
  with AUROC, accuracy, Brier, ECE15, fit/predict times and GPU memory.
- \`paper_benchmark/summary.csv\`: mean and between-draw SD by dataset/method/context.
- \`paper_benchmark/runtime_versions.json\`: GPU/runtime/library versions.
- \`paper_benchmark/hardware_profile.json\`: matched synthetic one-query,
  fit versus predict p50/p99 latency on the A10G.
- \`nanotab_training_protocol.json\` and \`nanotab_training_results.json\`
  for full reproducibility.
- \`COMPLETE.json\` exists **only if** all three stages finish.

### Limits

This is **one GPU system benchmark**; it does not establish on-device Android
RAM or latency. Fit time is separately measured from query time: PFNs adapt
to labeled context without gradient descent. The adaptive model has two
expert checkpoints and cannot inherit Tiny's single-model size or latency.
Do not interpret runtime on an A10G as mobile latency. Missingness handling
in the nanoTabPFN adaptation is a documented input compatibility choice.

For confirmatory conference claims, add an independent locked external
validation and isolate pretraining architecture/optimizer/augmentation changes
in controlled ablation experiments.
