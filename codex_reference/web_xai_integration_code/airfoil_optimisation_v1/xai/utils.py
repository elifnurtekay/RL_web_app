from __future__ import annotations
import math
from typing import Any, Iterable
import numpy as np
import pandas as pd

def safe_float(value: Any, default=None):
    try:
        if value is None: return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except Exception:
        return default

def l2_norm(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))

def top_shap_features(feature_names: list[str], shap_values: np.ndarray, top_k: int = 8) -> dict:
    vals = np.asarray(shap_values).reshape(-1)
    rows = [{'feature': n, 'impact': float(v), 'abs_impact': float(abs(v))} for n, v in zip(feature_names, vals)]
    pos = sorted([r for r in rows if r['impact'] > 0], key=lambda r: r['impact'], reverse=True)[:top_k]
    neg = sorted([r for r in rows if r['impact'] < 0], key=lambda r: r['impact'])[:top_k]
    all_ = sorted(rows, key=lambda r: r['abs_impact'], reverse=True)[:top_k]
    for item in pos + neg + all_: item.pop('abs_impact', None)
    return {'top_positive': pos, 'top_negative': neg, 'top_overall': all_}

def row_from_feature_dict(feature_dict: dict, feature_order: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{name: safe_float(feature_dict.get(name), 0.0) for name in feature_order}])
