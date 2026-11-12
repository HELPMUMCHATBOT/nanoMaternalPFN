# nanoMaternalPFN: paper-completion benchmark

Working title: **nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction**

This upgrade adds controlled comparators and profiling. It does **not** revise
any existing v1/v2 experiment results. Preserve all eight datasets, including
CTU cord acidemia and TPEHG preterm; do not drop datasets based on performance.

## Install and validation

Run on the user's Mac (local frozen manifests and model weights are not on GitHub).

~~~bash
cd ~/nanoMaternalPFN
source .venv/bin/activate
python -m pip install -e '.[paper,test]'
pytest -q
~~~

TabPFN v2 weights may download on first use. Review official license terms,
model pretraining overlap risk, and model/cache provenance. The official package
defaults to newer models, so the code explicitly invokes
TabPFNClassifier.create_default_for_version(ModelVersion.V2).
The single model uses n_estimators=1; the alternative uses the v2 checkpoint
default ("auto"). Report which implementation package and checkpoint ran.

## New full predictive comparison (separate output)

~~~bash
python scripts/09_run_downstream_study.py --full --resume \
  --methods xgboost lightgbm tabpfn_v2_single tabpfn_v2_default \
    adaptive_tiny adaptive_base \
  --checkpoint-dir artifacts/results/aws_v2_pretrain \
  --v1-checkpoint-dir artifacts/results/aws_full_pretrain \
  --output-dir artifacts/results/paper_completion_full

python scripts/15_merge_paper_results.py
~~~

All methods use the same frozen manifests and draw indices, with AUROC,
accuracy, multiclass Brier, and ECE15. The runner logs each draw, writes a
resumable per-method CSV, and logs adaptive weights and context NLL to JSONL.
The two VAT outcomes derive from the same underlying cohort.

The adaptive comparator uses two separately pretrained maternal-mixture experts
(v1 and v2 at matched tier). Stratified labeled-context CV determines mixture
weights, and an optionally fitted single regularized temperature is learned
using only labeled-context held-out predictions. When the smallest context
class cannot support CV folds, it falls back to 50:50 and T=1. This
**does not guarantee** any improvement and adds startup, memory, and inference
cost. It cannot inherit the single Tiny model's 1.52 MB file footprint.

The original v1 results were inspected before v2 design; comparisons on
previously seen datasets remain exploratory rather than independent evidence.

## ONNX parity and memory

~~~bash
python scripts/16_verify_onnx_parity.py
python scripts/14_measure_onnx_memory.py \
  --output artifacts/results/paper_onnx_memory.json

python scripts/12_profile_v2_edge.py \
  --checkpoint artifacts/results/aws_v2_pretrain/tiny_mixture.best.inference.pt \
  --output-dir artifacts/results/edge_paper_comparison \
  --contexts 16 64 128 --features 6 --repeats 100 --onnx \
  --baseline-methods logistic random_forest xgboost lightgbm \
    tabpfn_v2_single tabpfn_v2_default --baseline-repeats 5
~~~

The memory script uses an isolated Python process per graph with only
onnxruntime/NumPy, **not PyTorch**. It separately reports total process peak
RSS and incremental high-water above import baseline. Incremental RAM under
32 MB is NOT proof of under-32 MB total RAM or of Android deployment RAM.
The ONNX exporter is fixed-shape, and parity checks must pass before quoting
benchmarks for intended inputs.

The hardware comparison measures fit and query time separately, using one
synthetic target query and exactly the matching context/feature dimensions.
It excludes cold TabPFN checkpoint downloads; this timing benchmark is not
predictive accuracy benchmarking.

No Android, INT8, or energy results are claimed. For top venues, add a fresh
external locked validation cohort and controlled pretraining ablations.
