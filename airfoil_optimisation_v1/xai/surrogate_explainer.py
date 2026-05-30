from __future__ import annotations

from .artifact_loader import artifact_error, load_feature_order, load_proxy_model
from .config import SURROGATE_TARGETS, XAIArtifactConfig
from .shap_local import explain_tree_model_local


class SurrogateExplainer:
    def __init__(self, config: XAIArtifactConfig):
        self.config = config

    def explain(self, feature_row, top_k=8):
        results = {}
        artifact_dir = self.config.surrogate_dir()
        for target in SURROGATE_TARGETS:
            model_path = artifact_dir / f"{target}.joblib"
            feature_path = artifact_dir / f"{target}_features.json"
            model = load_proxy_model(artifact_dir, target)
            order = load_feature_order(artifact_dir, target)
            if model is None or order is None:
                results[target] = {
                    "available": False,
                    "reason": artifact_error(model_path) or artifact_error(feature_path) or f"artifact not found: {model_path if model is None else feature_path}",
                }
            else:
                results[target] = explain_tree_model_local(model, feature_row, order, top_k)
        return {"available": any(r.get("available") for r in results.values()), "layer": "surrogate_local_explanation", "targets": results}
