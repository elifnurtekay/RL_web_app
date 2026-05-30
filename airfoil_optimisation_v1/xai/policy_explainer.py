from __future__ import annotations

from .artifact_loader import artifact_error, load_feature_order, load_proxy_model
from .config import PPO_POLICY_TARGETS, SAC_POLICY_TARGETS, TD3_POLICY_TARGETS, XAIArtifactConfig
from .shap_local import explain_tree_model_local


class PolicyExplainer:
    def __init__(self, config: XAIArtifactConfig):
        self.config = config

    def _targets_for_algorithm(self, alg):
        alg = alg.lower()
        if alg == "td3":
            return TD3_POLICY_TARGETS
        if alg == "sac":
            return SAC_POLICY_TARGETS
        if alg == "ppo":
            return PPO_POLICY_TARGETS
        raise ValueError(f"Desteklenmeyen algorithm: {alg}")

    def explain(self, algorithm, feature_row, top_k=8):
        alg = algorithm.lower()
        artifact_dir = self.config.algorithm_dir(alg)
        results = {}
        for target in self._targets_for_algorithm(alg):
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
        return {
            "available": any(r.get("available") for r in results.values()),
            "layer": "policy_local_explanation",
            "algorithm": alg,
            "targets": results,
        }
