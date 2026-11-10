"""Audited source adapters for open, uncredentialed research datasets.

No synthetic or third-party fallback is allowed in this module. Dataset
documentation describes source-level claims; a dataset is 'acquired' only
after retrieval, schema validation and a recorded content checksum.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Source:
    name: str
    cohort: str
    tier: str
    outcome_type: str
    url: str
    license: str
    expected_n: tuple[int, int]
    source_note: str

SOURCES = {
    "uci_maternal_risk": Source("uci_maternal_risk","uci_863","proxy_sanity","risk_category",
        "https://archive.ics.uci.edu/dataset/863/maternal+health+risk","CC BY 4.0",(900,1100),
        "RiskLevel is not an adjudicated prospective clinical outcome; duplicate profiles are common."),
    "cardiotocography": Source("cardiotocography","uci_193","fetal_secondary","fetal_state",
        "https://archive.ics.uci.edu/dataset/193/cardiotocography","CC BY 4.0",(2000,2200),
        "Fetal CTG classification; exclude CLASS (a second target), use only 21 documented predictors."),
    "vat_gdm": Source("vat_gdm","physionet_vat_2020","maternal_outcome","gestational_diabetes",
        "https://physionet.org/content/maternal-visceral-adipose/1.0.0/","ODC-By 1.0",(120,140),
        "133 women followed to delivery; small prospective cohort; all post-outcome columns excluded."),
    "vat_delivery": Source("vat_delivery","physionet_vat_2020","maternal_outcome","cesarean_delivery",
        "https://physionet.org/content/maternal-visceral-adipose/1.0.0/","ODC-By 1.0",(120,140),
        "Same 133 women as vat_gdm. Correlated tasks; do not count as independent datasets."),
    "low_birth_weight": Source("low_birth_weight","mass_birthwt_1986","obstetric_outcome","low_birthweight",
        "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/birthwt.html","verify redistribution terms",(185,193),
        "189-record hospital-derived MASS example. Exclude birth weight in grams: it defines the label."),
    "pima_diabetes": Source("pima_diabetes","openml_37","nonmaternal_control","diabetes",
        "https://www.openml.org/d/37","verify OpenML source terms",(760,780),
        "Not a gestational diabetes study: female diabetes negative-control/transfer task."),
}

# Other candidate sources are documented but intentionally excluded until files
# and endpoint definitions can be verified. In particular, 3525-case GDM and
# Mendeley 1205-case maternal-risk tables must not silently become fake data.

VAT_CSV = "https://physionet.org/files/maternal-visceral-adipose/1.0.0/visceral_fat.csv"
BIRTHWT_CSV = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/birthwt.csv"

def _gestation_weeks(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return np.nan
    value = str(cell).strip()
    if "," in value:
        weeks, days = value.split(",", 1)
        if not (weeks.strip().isdigit() and days.strip().isdigit()):
            raise ValueError(f"Invalid gestational age: {value}")
        return int(weeks) + int(days) / 7
    return float(value)

def _vat_dataframe(path=None):
    df = pd.read_csv(path or VAT_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]
    needed = [
        "age (years)", "ethnicity", "diabetes mellitus",
        "mean diastolic bp (mmhg)", "mean systolic bp (mmhg)",
        "central armellini fat (mm)", "current gestational age",
        "pregnancies (number)", "first fasting glucose (mg/dl)",
        "bmi pregestational (kg/m)", "type of delivery", "gestational dm",
    ]
    missing = set(needed) - set(df)
    if missing:
        raise ValueError(f"PhysioNet schema changed, missing columns: {sorted(missing)}")
    df["current gestational age"] = df["current gestational age"].map(_gestation_weeks)
    return df

def _from_uci(uid, outcome):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as ex:
        raise ImportError("Install the data extra: pip install -e '.[data]'") from ex
    ds = fetch_ucirepo(id=uid)
    x = ds.data.features.copy()
    labels = ds.data.targets.copy()
    if isinstance(labels, pd.Series):
        labels = labels.to_frame()
    if outcome not in labels:
        raise ValueError(f"UCI {uid} did not expose expected target {outcome!r}")
    if uid == 193:
        # Prevent leakage from secondary morphology-class label.
        x = x.drop(columns=[c for c in ("CLASS", "NSP") if c in x], errors="ignore")
        if len(x.columns) != 21:
            raise ValueError(f"CTG expected 21 predictors, got {x.shape[1]}")
    elif uid == 863:
        if len(x.columns) != 6:
            raise ValueError(f"UCI maternal expected six predictors, got {x.shape[1]}")
    return x, labels[outcome]

def _pima():
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=37, as_frame=True)
    x = ds.data.copy()
    if x.shape[1] != 8:
        raise ValueError("Pima expected eight predictors")
    for c in ("plas", "pres", "skin", "insu", "mass"):
        if c in x:
            x[c] = x[c].replace(0, np.nan)
    return x, ds.target

def _birthwt(path=None):
    df = pd.read_csv(path or BIRTHWT_CSV)
    drop = [c for c in ("rownames", "Unnamed: 0", "bwt") if c in df]
    x = df.drop(columns=drop+["low"])
    return x, df["low"]

def get_open_task(name, local_file=None):
    """Return (numeric features, integer outcome, auditable metadata)."""
    if name not in SOURCES:
        raise KeyError(f"Unavailable/unverified source: {name}. Nothing was fabricated.")
    src = SOURCES[name]
    if name == "uci_maternal_risk":
        x,y = _from_uci(863, "RiskLevel")
    elif name == "cardiotocography":
        x,y = _from_uci(193, "NSP")
    elif name in ("vat_gdm", "vat_delivery"):
        frame = _vat_dataframe(local_file)
        outcome = "gestational dm" if name == "vat_gdm" else "type of delivery"
        forbidden = {
            "number", "type of delivery", "gestational dm",
            "gestational age at birth", "child birth weight (g)",
        }
        x = frame.drop(columns=list(forbidden & set(frame.columns)))
        y = frame[outcome]
    elif name == "low_birth_weight":
        x,y = _birthwt(local_file)
    elif name == "pima_diabetes":
        x,y = _pima()
    if not (src.expected_n[0] <= len(x) <= src.expected_n[1]):
        raise ValueError(f"{name}: unexpected row count {len(x)}")
    if y.isna().any():
        raise ValueError(f"{name}: missing outcome labels")
    if pd.DataFrame(x).isna().all(axis=0).any():
        raise ValueError(f"{name}: entirely absent predictor column")
    if any(str(c).lower().strip() in ("id","risklevel","class","nsp","bwt","low","outcome") for c in x):
        raise ValueError(f"{name}: leaked outcome or identifier")
    labels = pd.Categorical(y)
    if len(labels.categories) not in (2,3):
        raise ValueError(f"{name}: unexpected class count {len(labels.categories)}")
    encoded = pd.get_dummies(x, dummy_na=False, dtype=float).apply(pd.to_numeric, errors="raise")
    encoded = encoded.astype(np.float32)
    # Stable hash of the exact prepared input and target, not a claim of raw-file hash.
    stable = encoded.to_csv(index=False, float_format="%.8g").encode() + (
        "\nTARGET\n" + ",".join(str(v) for v in labels.codes)
    ).encode()
    meta = asdict(src) | {
        "rows":len(x),"features":list(encoded.columns),"classes":[str(c) for c in labels.categories],
        "prepared_sha256":sha256(stable).hexdigest(),
        "provenance_verified_after_fetch":True,"local_override":str(local_file) if local_file else None,
    }
    return encoded, labels.codes.astype(np.int64), meta

def acquire_and_freeze(names, out_dir="artifacts/manifests", seed=20261107):
    """Attempt real dataset acquisition, record any failed source separately."""
    from .freeze_splits import freeze
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report={"seed":seed,"dataset_results":{},"cohort_nonindependence":["vat_gdm","vat_delivery"]}
    for name in names:
        try:
            x,y,meta=get_open_task(name)
            manifest=freeze(x,y,out_dir / f"{name}.json",seed=seed)
            report["dataset_results"][name] = {
                "status":"frozen","source":meta,"split_sha256":manifest["sha256"],"audit":manifest["audit"],
                "eligible_context_budgets":sorted(int(s) for s in manifest["context_draws"]),
            }
        except Exception as exc:
            report["dataset_results"][name]={"status":"unavailable","reason":str(exc)}
    (out_dir / "open_portfolio_report.json").write_text(json.dumps(report,indent=2))
    return report
