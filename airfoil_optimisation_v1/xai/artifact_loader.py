from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_LOAD_ERRORS: dict[str, str] = {}


@lru_cache(maxsize=64)
def load_joblib(path: str):
    import joblib

    return joblib.load(path)


def artifact_error(path: Path) -> str | None:
    return _LOAD_ERRORS.get(str(path))


def load_proxy_model(artifact_dir: Path, target: str):
    path = artifact_dir / f"{target}.joblib"
    _LOAD_ERRORS.pop(str(path), None)
    if not path.exists():
        return None
    try:
        return load_joblib(str(path))
    except Exception as exc:
        _LOAD_ERRORS[str(path)] = str(exc)
        return None


def load_feature_order(artifact_dir: Path, target: str):
    path = artifact_dir / f"{target}_features.json"
    _LOAD_ERRORS.pop(str(path), None)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOAD_ERRORS[str(path)] = str(exc)
        return None
