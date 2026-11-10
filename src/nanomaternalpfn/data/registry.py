"""Explicit open-data registry; never synthesize missing 'real' datasets."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    target: str
    file: str
    status: str

REGISTRY = {
    'uci_maternal_risk': DatasetSpec('uci_maternal_risk', 'UCI ML Repository ID 863', 'RiskLevel', 'uci_maternal_risk.csv', 'open; verify download and license'),
    'pima_diabetes': DatasetSpec('pima_diabetes', 'OpenML 37; non-pregnancy-specific', 'class', 'pima_diabetes.csv', 'open; transfer-only'),
    'cardiotocography': DatasetSpec('cardiotocography', 'UCI ML Repository ID 193', 'NSP', 'cardiotocography.csv', 'open; fetal rather than maternal task'),
    'gestational_diabetes': DatasetSpec('gestational_diabetes', 'External GDM cohort; source/permission needs verification', 'GDM', 'gestational_diabetes.csv', 'unverified'),
    'low_birth_weight': DatasetSpec('low_birth_weight', 'Hosmer-Lemeshow low birthweight cohort', 'LOW', 'low_birth_weight.csv', 'source/version verification required'),
    'cesarean': DatasetSpec('cesarean', 'C-section dataset; metadata confirmation required', 'Caesarian', 'cesarean.csv', 'source/version verification required'),
    'maternal_morbidity': DatasetSpec('maternal_morbidity', '11-feature external maternal cohort; unverified', 'Outcome', 'maternal_morbidity.csv', 'unverified'),
}

def load_local(name, directory='data/raw', target=None):
    if name not in REGISTRY:
        raise KeyError(f'unknown dataset: {name}')
    spec = REGISTRY[name]
    path = Path(directory) / spec.file
    if not path.is_file():
        raise FileNotFoundError(f'{path} is unavailable; no synthetic data will be substituted')
    df = pd.read_csv(path)
    label = target or spec.target
    if label not in df:
        raise ValueError(f'missing verified target column {label!r} in {path}')
    y = df.pop(label)
    if y.isna().any():
        raise ValueError('missing target labels must be resolved before split freezing')
    x = pd.get_dummies(df, dummy_na=True, dtype=float)
    x = x.apply(pd.to_numeric, errors='raise')
    return x, pd.Categorical(y).codes, {'name':name,'source':spec.source,'file':str(path),'target':label}
