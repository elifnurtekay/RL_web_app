from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json, joblib
@lru_cache(maxsize=64)
def load_joblib(path: str): return joblib.load(path)
def load_proxy_model(artifact_dir: Path, target: str):
    path = artifact_dir / f'{target}.joblib'
    return None if not path.exists() else load_joblib(str(path))
def load_feature_order(artifact_dir: Path, target: str):
    path = artifact_dir / f'{target}_features.json'
    return None if not path.exists() else json.loads(path.read_text(encoding='utf-8'))
