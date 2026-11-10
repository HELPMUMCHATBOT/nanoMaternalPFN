"""Leakage-guarded public PhysioNet WFDB header adapters.

Data are obtained from PhysioNet's public (unsigned) AWS S3 mirror.
Only metadata .hea files are used: no waveform downloads or post-delivery
outcome values are included as predictors.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re

import numpy as np
import pandas as pd


PHYSIONET_BUCKET = "physionet-open"

# All features must be available at or before the assessment time.
TPEHG_CLINICAL_KEYS = (
    "Rectime", "Age", "Parity", "Abortions", "Weight", "Hypertension",
    "Diabetes", "Placental_position", "Bleeding_first_trimester",
    "Bleeding_second_trimester", "Funneling", "Smoker",
)
CTU_MATERNAL_KEYS = (
    "Age", "Gravidity", "Parity", "Diabetes", "Hypertension", "Preeclampsia",
)


def _public_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError("Install public S3 dependency with pip install -e '.[data]'") from exc
    return boto3.client(
        "s3", region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, max_pool_connections=24,
                      connect_timeout=15, read_timeout=30,
                      retries={"max_attempts":5, "mode":"standard"}),
    )


def _parse_comments(text):
    result={}
    for line in text.splitlines():
        s=line.strip()
        if not s.startswith("#"):
            continue
        s=s.lstrip("#").strip()
        if not s or s.lower().startswith(("comments:", "--", "-----")):
            continue
        # Keep all text after the *first* whitespace as a single value.
        match=re.match(r"^([A-Za-z][A-Za-z_.]*)\\s+(.+)$",s)
        if match:
            result[match.group(1).strip()]=match.group(2).strip()
    return result


def _numeric(value):
    if value is None:
        return np.nan
    text=str(value).strip().lower()
    if text in ("none", "n/a", "na", "null", "?", "", "-"):
        return np.nan
    if text in ("yes","true","t","positive","pos"):
        return 1.0
    if text in ("no","false","f","negative","neg"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def _fetch_headers(prefix, expected_min, expected_max):
    client=_public_client()
    paths=[]
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=PHYSIONET_BUCKET, Prefix=prefix
    ):
        paths.extend(item["Key"] for item in page.get("Contents",[])
                     if item["Key"].lower().endswith(".hea"))
    paths=sorted(set(paths))
    if not expected_min <= len(paths) <= expected_max:
        raise ValueError(
            f"Unexpected PhysioNet header count for {prefix}: {len(paths)}"
        )
    def get(path):
        raw=client.get_object(Bucket=PHYSIONET_BUCKET, Key=path)["Body"].read()
        if len(raw)>16_384:
            raise ValueError(f"Unexpectedly large WFDB header {path}")
        return path,_parse_comments(raw.decode("utf-8",errors="replace"))
    with ThreadPoolExecutor(max_workers=16) as pool:
        pairs=list(pool.map(get,paths))
    if len(pairs)!=len(paths):
        raise ValueError("Missing PhysioNet WFDB headers")
    return pairs


def _tpehg_dataframe(pairs):
    rows=[]
    for path,header in pairs:
        if not all(k in header for k in ("Gestation","Rectime")):
            raise ValueError(f"TPEHG missing critical header keys: {path}")
        gest=_numeric(header.get("Gestation"))
        if not np.isfinite(gest) or not 22<=gest<=46:
            raise ValueError(f"Implausible duration of pregnancy {gest} in {path}")
        row={k:header.get(k) for k in TPEHG_CLINICAL_KEYS}
        row["preterm"] = int(gest <= 37)
        rows.append(row)
    df=pd.DataFrame(rows)
    # Distinct metadata fields, not recorded delivery gestation or the
    # outcome-derived 'Group/Premature' indicators.
    binary = ("Hypertension","Diabetes","Bleeding_first_trimester",
              "Bleeding_second_trimester","Smoker","Funneling")
    for k in binary:
        df[k]=df[k].map(_numeric)
    for k in ("Rectime","Age","Parity","Abortions","Weight"):
        df[k]=pd.to_numeric(df[k],errors="coerce")
    # Placental position is categorical, not an ordinal biological quantity.
    placenta=df.pop("Placental_position")
    dummy=pd.get_dummies(placenta,dummy_na=False,prefix="placenta",dtype=float)
    for k in dummy:
        df[k]=dummy[k]
    if df["preterm"].nunique()!=2:
        raise ValueError("Preterm outcome collapsed to a single class")
    x=df.drop(columns=["preterm"])
    # Data quality check: 5+ nonconstant clinical predictors required.
    if x.nunique(dropna=True).gt(1).sum()<5:
        raise ValueError("Insufficient nonconstant TPEHG clinical predictors")
    return x, df["preterm"]


def _ctu_dataframe(pairs, ph_threshold=7.15):
    """Exploratory binary cord-acidemia task, NOT a validated clinical cutoff.

    The assessment is at entry into labor monitoring. To avoid post-outcome
    leakage, features are ONLY maternal age/obstetric history/risk factors.
    Delivery type, fetal weight, Apgar, neonatal outcomes, CTG signal statistics,
    cord pH and all labor durations are excluded.
    """
    rows=[]
    for path,header in pairs:
        pH=_numeric(header.get("pH"))
        if not np.isfinite(pH) or not 6.5<=pH<=8.0:
            # A few records can use a missing-pH placeholder; drop explicitly.
            continue
        row={k:_numeric(header.get(k)) for k in CTU_MATERNAL_KEYS}
        row["cord_ph_below_7_15"]=int(pH<ph_threshold)
        rows.append(row)
    if not rows:
        raise ValueError("No CTU-UHB records with plausible arterial cord pH")
    df=pd.DataFrame(rows)
    outcome=df.pop("cord_ph_below_7_15")
    if outcome.nunique()!=2 or min(outcome.value_counts())<10:
        raise ValueError("CTU-UHB outcome has too few positive/negative observations")
    if df.nunique(dropna=True).gt(1).sum()<4:
        raise ValueError("Insufficient informative CTU-UHB maternal predictors")
    return df, outcome


def load_tpehg_preterm():
    pairs=_fetch_headers("tpehgdb/1.0.1/tpehgdb/",290,310)
    return _tpehg_dataframe(pairs)


def load_ctu_uhb_acidemia():
    pairs=_fetch_headers("ctu-uhb-ctgdb/1.0.0/",540,565)
    return _ctu_dataframe(pairs)
