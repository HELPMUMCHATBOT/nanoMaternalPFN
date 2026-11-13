#!/usr/bin/env python3
"""Create factual research tables and a draft manuscript ONLY from measured files.

Never fills absent results with estimates. A 3-expert result is included only
when its separate real-data evaluation output is explicitly supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TITLE = "nanoMaternalPFN: A Lightweight Tabular Foundation Model for Maternal Health Prediction"
DATASETS = (
    "uci_maternal_risk", "vat_gdm", "vat_delivery", "low_birth_weight",
    "cardiotocography", "pima_diabetes", "tpehg_preterm", "ctu_cord_acidemia",
)
METHODS = (
    "logistic", "random_forest", "tiny_generic", "tiny_mixture",
    "base_generic", "base_mixture", "xgboost", "lightgbm",
    "tabpfn_v2_single", "tabpfn_v2_default", "adaptive_tiny", "adaptive_base",
)
METRICS = ("auroc", "brier", "ece15", "accuracy")
CONTEXTS_LOW = (16, 32, 64)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path, *, require_methods=METHODS):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Measured summary unavailable: {path}")
    df = pd.read_csv(path)
    needed = {"dataset", "method", "context", "metric", "mean", "n_valid_draws"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing summary columns: {sorted(needed - set(df.columns))}")
    if df.duplicated(["dataset", "method", "context", "metric"]).any():
        raise ValueError("Duplicate dataset/method/context/metric summary rows")
    for key in ("context", "n_valid_draws", "mean"):
        df[key] = pd.to_numeric(df[key], errors="raise")
    if df.empty or (~np.isfinite(df["mean"])).any():
        raise ValueError("Summary has absent or nonfinite aggregate measurements")
    if (df["n_valid_draws"] < 1).any():
        raise ValueError("Summary contains aggregates without valid draws")
    missing_methods = set(require_methods) - set(df["method"])
    if missing_methods:
        raise ValueError("Required measured methods missing: " +
                         ", ".join(sorted(missing_methods)))
    if set(df["dataset"]) != set(DATASETS):
        raise ValueError("Expected exactly 8 frozen evaluation tasks; found " +
                         str(sorted(set(df["dataset"]))))
    for method in require_methods:
        observed = set(df.loc[(df.method == method) & (df.metric == "auroc"),
                              ["dataset", "context"]].itertuples(index=False, name=None))
        expected = set(df.loc[(df.method == "tabpfn_v2_default") & (df.metric == "auroc"),
                              ["dataset", "context"]].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"Unequal AUROC dataset/context coverage for {method}")
    return df


def macro_table(df):
    # First average context means within each dataset, then give each task
    # equal weight. This is descriptive: two VAT outcomes share a cohort.
    per_task = df.groupby(["method", "metric", "dataset"],
                          as_index=False)["mean"].mean()
    values = per_task.groupby(["method", "metric"])["mean"].mean().unstack()
    return values.sort_values("auroc", ascending=False).reindex(columns=METRICS)


def markdown(frame, decimals=4):
    data = frame.reset_index()
    def fmt(x):
        if pd.isna(x):
            return "not measured"
        if isinstance(x, (int, float, np.integer, np.floating)):
            return f"{x:.{decimals}f}"
        return str(x).replace("|", "/").replace("\n", " ")
    cols = [str(c) for c in data.columns]
    rows = ["| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |"]
    for record in data.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(fmt(v) for v in record) + " |")
    return "\n".join(rows)


def load_raw_draws(root):
    """Read original per-draw CSVs; never reconstruct draws from summary means."""
    sources = (
        ("downstream_v2_full", ""),
        ("paper_completion_full", ""),
        ("paper_three_expert_full", ""),
        ("downstream_full", "v1_"),
    )
    frames = []
    inputs = []
    for directory, prefix in sources:
        folder = root / directory
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.csv")):
            bits = file.stem.rsplit(".", 1)
            if len(bits) != 2 or bits[0] not in DATASETS:
                continue
            dataset, method = bits
            if directory == "downstream_full" and method not in (
                "tiny_generic", "tiny_mixture", "base_generic", "base_mixture"
            ):
                continue
            raw = pd.read_csv(file)
            needed = {"context", "seed", *METRICS}
            if not needed.issubset(raw.columns):
                raise ValueError(f"Raw CSV missing needed metrics: {file}")
            raw = raw.loc[:, list(needed)].copy()
            raw["dataset"] = dataset
            raw["method"] = prefix + method
            frames.append(raw)
            inputs.append(str(file))
    if not frames:
        return pd.DataFrame(), inputs
    joined = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "method", "context", "seed"]
    if joined.duplicated(keys).any():
        raise ValueError("Duplicated raw draws across evidence directories")
    return joined, inputs


def descriptive_pairs(raw, resamples=2000):
    """Paired differences over context draws; NOT patient/cohort-level CIs."""
    fields = ["dataset", "context", "method_a", "method_b", "metric",
              "n_shared_draws", "mean_improvement_a", "draw_ci2_5",
              "draw_ci97_5"]
    if raw.empty:
        return pd.DataFrame(columns=fields)
    rows = []
    pairs = [
        ("adaptive_tiny", "tabpfn_v2_default"),
        ("adaptive_base", "tabpfn_v2_default"),
        ("adaptive_tiny", "xgboost"),
        ("adaptive_tiny", "lightgbm"),
        ("adaptive_tiny_3expert", "tabpfn_v2_default"),
        ("adaptive_tiny_3expert", "adaptive_tiny"),
    ]
    for a, b in pairs:
        lhs = raw.loc[raw.method == a]
        rhs = raw.loc[raw.method == b]
        if lhs.empty or rhs.empty:
            continue
        match = lhs.merge(rhs, on=["dataset", "context", "seed"],
                          suffixes=("_a", "_b"), validate="one_to_one")
        for (dataset, context), group in match.groupby(["dataset", "context"]):
            for metric in METRICS:
                av = group[f"{metric}_a"].to_numpy(dtype=float)
                bv = group[f"{metric}_b"].to_numpy(dtype=float)
                valid = np.isfinite(av) & np.isfinite(bv)
                diffs = (bv[valid] - av[valid]) if metric in ("brier", "ece15") else (
                    av[valid] - bv[valid])
                if not len(diffs):
                    continue
                lo = hi = float("nan")
                if len(diffs) > 1:
                    rng = np.random.default_rng(20261113 + int(context))
                    samples = rng.integers(0, len(diffs), size=(resamples,len(diffs)))
                    lo,hi = np.quantile(diffs[samples].mean(axis=1), [0.025,0.975])
                rows.append([dataset,context,a,b,metric,len(diffs),
                             float(diffs.mean()),float(lo),float(hi)])
    return pd.DataFrame(rows, columns=fields)


def make_manuscript(full, low, low_by_dataset, ctu, edge, provenance,
                    pairs, adaptive_cpu=None):
    tiny = full.loc["adaptive_tiny", "auroc"]
    reference = full.loc["tabpfn_v2_default", "auroc"]
    xgb = full.loc["xgboost", "auroc"]
    lgb = full.loc["lightgbm", "auroc"]
    edge_rows = []
    for item in edge.get("results", []):
        ort = item.get("onnxruntime")
        if ort:
            edge_rows.append({
                "context": item.get("context"),
                "model": edge.get("model_variant", "standalone model"),
                "CPU ONNX median ms": ort.get("p50_ms"),
                "CPU ONNX p99 ms": ort.get("p99_ms"),
                "ONNX file MB": ort.get("file_size_mb"),
            })
    edge_table = markdown(pd.DataFrame(edge_rows)) if edge_rows else (
        "No validated ONNX timing entries in supplied EdgeBench profile.")
    baseline_comparisons = edge.get("baseline_comparisons", [])
    baseline_table = (
        markdown(pd.DataFrame(baseline_comparisons))
        if baseline_comparisons else
        "No matched baseline CPU timing entries supplied."
    )
    extra = ""
    if adaptive_cpu:
        if adaptive_cpu.get("results"):
            extra = "\n### Measured adaptive CPU profile\n\n" + markdown(
                pd.DataFrame(adaptive_cpu["results"])) + "\n"
    pair_text = ("No paired raw draw files were supplied; uncertainty not estimated."
                 if pairs.empty else
                 "Matching-context, descriptive bootstrap contrasts were exported "
                 "to paired_effects.csv. Draw-based intervals are not independent "
                 "patient- or cohort-level uncertainty estimates.")
    ctu_text = markdown(ctu) if not ctu.empty else "No eligible CTU AUROC rows."
    return f"""# {TITLE}

**Research manuscript draft — automatically populated from measured local files.**
**Clinical status:** Research-only; not evaluated for patient-facing decision support.
**Authors/affiliations:** To be confirmed by the research team.
**Not an independent confirmatory validation.**

## Abstract

**Background.** Very small maternal and obstetric datasets motivate compact,
task-adaptive prediction systems. **Methods.** We evaluated pretrained
nanoMaternalPFN models on eight frozen public prediction tasks from seven
source cohorts, comparing synthetic-prior variants, classical estimators and
explicit TabPFN v2 configurations. Training of neural PFN weights used only
synthetic tasks; downstream predictors condition on labeled real-data contexts.

**Results.** Across the eight tasks and eligible contexts, the observed
equal-task macro AUROC was {tiny:.4f} for Adaptive Tiny versus
{reference:.4f} for TabPFN v2 Default (difference {tiny-reference:+.4f}).
XGBoost achieved {xgb:.4f} and LightGBM {lgb:.4f} in the same
descriptive aggregation. These values are not an equivalence test.
See the measured EdgeBench outputs below for the standalone model.
**Conclusion.** Results support competitive synthetic-to-real low-data
transfer with a compact standalone architecture option. This is retrospective
exploratory evaluation, not clinical utility or prospective validation.

## 1. Introduction

Maternal health risk modelling often involves few labelled records,
heterogeneous features and potentially missing clinical observations.
Prior-data fitted networks learn inference algorithms from synthetic tasks,
enabling conditioning on small labelled contexts without downstream neural
weight updates. Our question is whether a compact maternal-informed PFN
can provide useful discrimination and calibration at modest inference cost.

## 2. Methods

### 2.1 Architecture and synthetic pretraining

The nanoMaternalPFN encoder models feature values and missingness, alternates
feature and context-restricted patient attention, and predicts labels for query
records. The v1 and v2 checkpoints were trained on synthetic prior-generated
tasks rather than on the downstream real rows. Generic and maternal-mixture
priors are separate arms. The current adaptive system combines v1 and v2
maternal-mixture checkpoints using only cross-validated labelled-context
log-loss and optional context-fitted temperature scaling.

### 2.2 Evaluation tasks and protocol

Eight tasks: UCI maternal risk, VAT gestational diabetes, VAT delivery,
low birth weight, cardiotocography, Pima diabetes, term/preterm EHG and
CTU cord acidemia. VAT outcomes share a cohort; Pima is a non-maternal
transfer control; UCI maternal risk uses a categorized risk proxy. Frozen
manifest split indices and context draws are reused across methods.
Context sizes 16, 32, 64 and larger eligible budgets are summarized
separately. Query/test labels do not determine adaptive expert weights.

### 2.3 Comparison methods and estimands

Classical comparators are Logistic Regression, Random Forest, XGBoost and
LightGBM with the recorded implementations. Official TabPFN v2 is evaluated
as single-pass and version-default configurations. Report AUROC, accuracy,
Brier score and ECE15. LightGBM's default minimum child samples can prevent
useful splits in very small contexts, so do not interpret the configured
comparison as a fully tuned boosted-tree contest.

### 2.4 Hardware measures

The ONNX standalone model's file size, single-query CPU median and p99
latencies are separate from multi-checkpoint Adaptive results.
Incremental process RSS and total process peak RSS must not be interchanged.
No Android, INT8 device energy or mobile peak RAM is inferred.

### 2.5 Analysis status

The macro average first averages eligible contexts within each task,
then weights each task equally. It treats VAT GDM and VAT delivery as
separate prediction tasks, not independent patient cohorts. Draw-level
uncertainty is descriptive only, because the same test patients recur
across context draws. Earlier inspection of the benchmark informed
model development; later comparisons are exploratory.

## 3. Measured results

### 3.1 Full leaderboard (all eligible contexts)

{markdown(full)}

### 3.2 Low-data leaderboard (contexts 16, 32, 64)

{markdown(low)}

### 3.3 Low-data AUROC by dataset

{markdown(low_by_dataset)}

### 3.4 CTU cord acidemia, AUROC by context

{ctu_text}

### 3.5 Inference efficiency: standalone ONNX model

{edge_table}

### 3.6 Measured baseline EdgeBench timings

{baseline_table}
{extra}
### 3.7 Statistical comparisons

{pair_text}

## 4. Discussion

The measured macro AUROC suggests strong competitiveness with TabPFN v2
under the evaluated settings. A difference as small as the observed macro
gap does not prove statistical equivalence. Adaptive Tiny combines at least
two checkpoints and cannot inherit the runtime, memory or size of a single
Tiny ONNX export. A formal accuracy-efficiency Pareto frontier requires
performance and hardware measurements of the same predictor configurations.

## 5. Limitations and clinical interpretation

This is not prospective external validation or evidence of patient benefit.
Seven source cohorts underpin eight prediction tasks. Labels, prevalence
and clinical timing differ across datasets. Performance in the low-data
regime may depend on comparator hyperparameters and repeated small contexts.
Earlier benchmark inspection informed v2 and adaptive development, so
the results are exploratory. Do not use the model for diagnosis or triage.

## 6. Reproducibility

The source CSV and EdgeBench JSON SHA-256 hashes, discovered checkpoint
fingerprints and raw draw audit are recorded in provenance.json.
No missing experiment is filled with a fabricated result.

## 7. Conclusion

Synthetic-only PFN pretraining produced a competitive compact research
model under the evaluated tasks. Further clinical claims would require
external validation, calibration assessment and deployment studies.

## References

Bibliographic citations, author list, ethical/data licenses and final
submission formatting must be verified by the research team before submission.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/results"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/results/paper_verified"))
    p.add_argument("--include-three-expert", action="store_true",
                   help="Include only after the separate 3-expert run exists")
    p.add_argument("--resamples", type=int, default=2000)
    args = p.parse_args()
    if args.resamples < 100:
        p.error("at least 100 bootstrap draws required")
    root = args.root
    result = args.output_dir
    result.mkdir(parents=True, exist_ok=True)
    primary_path = root / "paper_all_methods_summary.csv"
    edge_path = root / "edge_paper_comparison/profile.json"
    memory_path = root / "paper_onnx_memory.json"
    for file in (primary_path, edge_path, memory_path):
        if not file.is_file():
            raise FileNotFoundError(f"Required measured artifact missing: {file}")
    primary = load_summary(primary_path)
    full = macro_table(primary)
    low = macro_table(primary[primary.context.isin(CONTEXTS_LOW)])
    by_dataset = (primary.loc[(primary.context.isin(CONTEXTS_LOW)) &
                              (primary.metric == "auroc")]
                  .groupby(["dataset","method"])["mean"].mean().unstack())
    ctu = (primary.loc[(primary.dataset=="ctu_cord_acidemia") &
                       (primary.metric=="auroc") &
                       (primary.context.isin(CONTEXTS_LOW))]
           .pivot(index="context",columns="method",values="mean"))
    three = root / "paper_three_expert_full/summary.csv"
    if args.include_three_expert:
        if not three.is_file():
            raise FileNotFoundError(f"3-expert measurements unavailable: {three}")
        measured_three = pd.read_csv(three)
        if set(measured_three.method) != {"adaptive_tiny_3expert"}:
            raise ValueError("3-expert summary method mismatch")
        if measured_three.duplicated(["dataset","method","context","metric"]).any():
            raise ValueError("Duplicate three-expert summary rows")
        extended = pd.concat([primary,measured_three],ignore_index=True)
        if extended.duplicated(["dataset","method","context","metric"]).any():
            raise ValueError("Original results overlap 3-expert results")
        # Accept ONLY a complete, same-coverage comparator, not a partial run.
        expected = primary.loc[
            (primary.method=="adaptive_tiny") & (primary.metric=="auroc"),
            ["dataset","context","n_valid_draws"]
        ].set_index(["dataset","context"]).sort_index()
        observed = measured_three.loc[
            measured_three.metric=="auroc",
            ["dataset","context","n_valid_draws"]
        ].set_index(["dataset","context"]).sort_index()
        if not expected.index.equals(observed.index):
            raise ValueError("Three-expert run has incomplete context/dataset coverage")
        if not expected["n_valid_draws"].equals(observed["n_valid_draws"]):
            raise ValueError("Three-expert run has missing or extra valid context draws")
        if set(measured_three.metric) != set(METRICS):
            raise ValueError("Missing metric in three-expert evaluation")
        if not np.isfinite(pd.to_numeric(measured_three["mean"])).all():
            raise ValueError("Three-expert metric has nonfinite aggregate")
        # Additional exploratory results live in separate outputs.
        extended_macro = macro_table(extended)
        extended_macro.to_csv(result/"exploratory_extended_leaderboard.csv")
    edge = json.loads(edge_path.read_text())
    memory_measurements = json.loads(memory_path.read_text())
    (result/"measured_memory_original.json").write_text(
        json.dumps(memory_measurements,indent=2)+"\n")
    adaptive_path = root / "adaptive_cpu_profile.json"
    adaptive = (json.loads(adaptive_path.read_text())
                if adaptive_path.is_file() else None)
    raw, evidence_files = load_raw_draws(root)
    pairs = descriptive_pairs(raw, args.resamples)
    # Original v1 and v2 raw scores are preserved as observational development
    # history, not a clean single-factor causal architecture ablation.
    development_table = None
    if not raw.empty:
        ablation = raw.loc[raw.method.isin([
            "v1_tiny_generic","v1_tiny_mixture","v1_base_generic","v1_base_mixture",
            "tiny_generic","tiny_mixture","base_generic","base_mixture",
            "adaptive_tiny","adaptive_base",
        ])]
        if not ablation.empty:
            development_table = (
                ablation.groupby(["method","context"],as_index=False)[list(METRICS)]
                .mean()
            )
            development_table.to_csv(
                result/"v1_v2_adaptive_development_table.csv",index=False
            )
    full.to_csv(result/"leaderboard_all.csv")
    low.to_csv(result/"leaderboard_low_data.csv")
    by_dataset.to_csv(result/"low_data_auroc_by_dataset.csv")
    ctu.to_csv(result/"ctu_auroc_by_context.csv")
    pairs.to_csv(result/"paired_effects.csv",index=False)
    observed_files = [primary_path,edge_path,memory_path] + [
        Path(x) for x in evidence_files
    ]
    if args.include_three_expert:
        observed_files.append(three)
    if adaptive_path.exists():
        observed_files.append(adaptive_path)
    provenance = {
        "title":TITLE,
        "status":"measured_source_draft_not_independent_confirmation",
        "datasets":list(DATASETS),
        "macro_aggregation":"equal task weight, equal available context weight per task",
        "shared_cohorts":[["vat_gdm","vat_delivery"]],
        "nonmaternal_control":"pima_diabetes",
        "observed_summary_rows":int(len(primary)),
        "number_of_raw_draws":int(len(raw)),
        "three_expert_included":bool(args.include_three_expert),
        "draw_bootstrap_not_patient_ci":True,
        "files":[{"path":str(f),"sha256":sha(f),"bytes":f.stat().st_size}
                 for f in sorted(set(observed_files))],
        "not_measured_or_not_verified":[],
    }
    if raw.empty:
        provenance["not_measured_or_not_verified"].append(
            "No raw paired draw metrics found; no paired effect estimates")
    if not adaptive:
        provenance["not_measured_or_not_verified"].append(
            "Adaptive CPU timing and memory not supplied")
    for variant in ("aws_full_pretrain","aws_v2_pretrain"):
        folder = root / variant
        if folder.is_dir():
            for ckpt in sorted(folder.glob("*.best.inference.pt")):
                provenance.setdefault("checkpoint_fingerprints",[]).append({
                    "path":str(ckpt),"sha256":sha(ckpt),"bytes":ckpt.stat().st_size
                })
        else:
            provenance["not_measured_or_not_verified"].append(
                f"Checkpoint folder unavailable for hashing: {folder}")
    (result/"provenance.json").write_text(json.dumps(
        provenance,indent=2,allow_nan=False)+"\n")
    manuscript = make_manuscript(full,low,by_dataset,ctu,edge,provenance,pairs,
                                 adaptive_cpu=adaptive)
    if development_table is not None:
        manuscript += (
            "\n## Supplement: v1, v2 and adaptive development results\n\n"
            "Descriptive chronological model comparison, not a single-factor "
            "causal ablation, because multiple changes differed across versions.\n\n"
            + markdown(development_table) + "\n"
        )
    if args.include_three_expert:
        manuscript += (
            "\n## Supplement: exploratory three-expert Tiny result\n\n"
            "This selector was added after multiple inspections of the same test "
            "benchmarks and cannot be described as independent confirmation. "
            "Weights use context-only CV; measured clinical scores are from "
            "the separate, completed frozen-context evaluation.\n\n"
            + markdown(extended_macro) + "\n"
        )
    (result/"manuscript_draft.md").write_text(manuscript)
    print(json.dumps({
        "source_summary_rows":len(primary),
        "raw_draws":len(raw),
        "paired_effect_rows":len(pairs),
        "three_expert_included":args.include_three_expert,
        "output":str(result),
        "status":"DRAFT_FROM_MEASURED_FILES_ONLY",
    },indent=2))


if __name__ == "__main__":
    main()
