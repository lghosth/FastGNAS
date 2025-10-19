from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
try:
    import shap
except Exception as exc:
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

@dataclass
class ReferenceShapleyConfig:
    csv_path: str
    target_column: str
    shap_type: str = 'tree'
    nsamples: Optional[int] = None
    random_seed: Optional[int] = None
    kernel_background_size: Optional[int] = None

def _ensure_shap_available() -> None:
    if shap is None:
        raise ImportError("The 'shap' package is required for reference Shapley computation" + (f': {_SHAP_IMPORT_ERROR}' if _SHAP_IMPORT_ERROR else ''))

def _load_reference_frame(cfg: ReferenceShapleyConfig) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f'Reference Shapley CSV not found: {cfg.csv_path}')
    df = pd.read_csv(cfg.csv_path)
    if cfg.target_column not in df.columns:
        raise KeyError(f"Target column '{cfg.target_column}' not present in CSV. Available columns: {list(df.columns)}")
    if cfg.nsamples is not None and cfg.nsamples > 0:
        df = df.iloc[:cfg.nsamples]
    df = df.sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)
    return df

def _compute_tree_explainer_importance(X: pd.DataFrame, y: pd.Series, random_seed: Optional[int]) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=1000, max_depth=4, random_state=random_seed)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    values = explanation.values
    return np.asarray(values, dtype=float)

def _compute_kernel_explainer_importance(X: pd.DataFrame, y: pd.Series, background_size: Optional[int], random_seed: Optional[int]) -> np.ndarray:
    base_model = StackingRegressor(estimators=[('rfr', RandomForestRegressor())], final_estimator=RandomForestRegressor(), cv=3)
    base_model.fit(X, y)
    background = X
    if background_size is not None and 0 < background_size < len(X):
        background = X.sample(background_size, random_state=random_seed)
    explainer = shap.KernelExplainer(base_model.predict, background, link='identity')
    shap_values = explainer.shap_values(X)
    return np.asarray(shap_values, dtype=float)

def compute_reference_shapley(cfg: ReferenceShapleyConfig) -> Dict[str, float]:
    _ensure_shap_available()
    df = _load_reference_frame(cfg)
    X = df.drop(columns=[cfg.target_column])
    y = df[cfg.target_column]
    _X, X_val, _y, y_val = train_test_split(X, y, test_size=0.2, random_state=cfg.random_seed)
    X_train = _X
    y_train = _y
    shap_type = cfg.shap_type.lower()
    if shap_type == 'tree':
        contributions = _compute_tree_explainer_importance(X_train, y_train, cfg.random_seed)
    elif shap_type == 'kernel':
        contributions = _compute_kernel_explainer_importance(X_train, y_train, cfg.kernel_background_size, cfg.random_seed)
    else:
        raise ValueError(f'Unsupported shap_type: {cfg.shap_type}')
    if contributions.ndim == 3:
        contributions = np.mean(np.asarray(contributions), axis=0)
    if contributions.ndim != 2:
        raise ValueError(f'Unexpected SHAP contribution shape {contributions.shape}; expected (n_samples, n_features)')
    mean_abs = np.mean(np.abs(contributions), axis=0)
    if mean_abs.shape[0] != X_train.shape[1]:
        raise ValueError(f'Mismatch between computed importance and feature count: {mean_abs.shape[0]} vs {X_train.shape[1]}')
    return dict(zip(X_train.columns.tolist(), mean_abs.tolist()))

def remap_feature_names(importance: Dict[str, float], mapping: Optional[Dict[str, str]]=None, allowed_features: Optional[Iterable[str]]=None) -> Dict[str, float]:
    if mapping:
        remapped = {}
        for src, value in importance.items():
            dest = mapping.get(src)
            if dest is None:
                continue
            remapped[dest] = remapped.get(dest, 0.0) + float(value)
    else:
        remapped = dict(importance)
    if allowed_features is not None:
        allowed = set(allowed_features)
        remapped = {name: value for name, value in remapped.items() if name in allowed}
    return remapped