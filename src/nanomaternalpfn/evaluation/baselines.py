"""Fit baselines solely on a context subset; no cross-split preprocessing."""
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_baseline(name, seed=0):
    if name=='logistic':return make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
    if name=='random_forest':return make_pipeline(SimpleImputer(add_indicator=True),RandomForestClassifier(n_estimators=200,random_state=seed))
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200,random_state=seed)
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200,verbosity=-1,random_state=seed)
    raise ValueError(f'unsupported baseline: {name}')
