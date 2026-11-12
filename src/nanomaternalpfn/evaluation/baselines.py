"""Class-safe classical comparators and explicitly pinned official TabPFN v2."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class EncodedXGBoost:
    """XGBoost requires contiguous target indices even if a context omits class 1."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.classes_, local_y = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("at least two observed classes required")
        self.model_ = XGBClassifier(
            n_estimators=200, tree_method="hist", n_jobs=1, verbosity=0,
            random_state=self.seed,
            objective="binary:logistic" if k == 2 else "multi:softprob",
            eval_metric="logloss" if k == 2 else "mlogloss",
        )
        self.model_.fit(X, local_y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def get_baseline(name, seed=0, tabpfn_device="cpu"):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(add_indicator=True),
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        )
    if name == "xgboost":
        return EncodedXGBoost(seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, verbosity=-1,
                              random_state=seed, n_jobs=1)
    if name in ("tabpfn_v2_single", "tabpfn_v2_default"):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise ImportError(
                "Official TabPFN required: pip install -e '.[paper]'"
            ) from exc
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=1 if name == "tabpfn_v2_single" else "auto",
            device=tabpfn_device,
            random_state=int(seed),
        )
    raise ValueError(f"unsupported baseline: {name}")
