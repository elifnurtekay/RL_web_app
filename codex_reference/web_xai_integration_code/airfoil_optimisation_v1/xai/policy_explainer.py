from __future__ import annotations
from .artifact_loader import load_feature_order, load_proxy_model
from .config import PPO_POLICY_TARGETS, SAC_POLICY_TARGETS, TD3_POLICY_TARGETS, XAIArtifactConfig
from .shap_local import explain_tree_model_local
class PolicyExplainer:
    def __init__(self, config: XAIArtifactConfig): self.config = config
    def _targets_for_algorithm(self, alg):
        alg=alg.lower()
        if alg=='td3': return TD3_POLICY_TARGETS
        if alg=='sac': return SAC_POLICY_TARGETS
        if alg=='ppo': return PPO_POLICY_TARGETS
        raise ValueError(f'Desteklenmeyen algorithm: {alg}')
    def explain(self, algorithm, feature_row, top_k=8):
        alg=algorithm.lower(); artifact_dir=self.config.algorithm_dir(alg); results={}
        for target in self._targets_for_algorithm(alg):
            model, order = load_proxy_model(artifact_dir, target), load_feature_order(artifact_dir, target)
            results[target] = {'available': False, 'reason': f'{alg.upper()} policy proxy artifact bulunamadı: {artifact_dir / (target + ".joblib")}' } if model is None or order is None else explain_tree_model_local(model, feature_row, order, top_k)
        return {'layer':'policy_local_explanation','algorithm':alg,'targets':results}
