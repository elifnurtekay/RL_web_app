from __future__ import annotations
from .artifact_loader import load_feature_order, load_proxy_model
from .config import SURROGATE_TARGETS, XAIArtifactConfig
from .shap_local import explain_tree_model_local
class SurrogateExplainer:
    def __init__(self, config: XAIArtifactConfig): self.config = config
    def explain(self, feature_row, top_k=8):
        results, artifact_dir = {}, self.config.surrogate_dir()
        for target in SURROGATE_TARGETS:
            model, order = load_proxy_model(artifact_dir, target), load_feature_order(artifact_dir, target)
            results[target] = {'available': False, 'reason': f'Surrogate proxy artifact bulunamadı: {artifact_dir / (target + ".joblib")}' } if model is None or order is None else explain_tree_model_local(model, feature_row, order, top_k)
        return {'layer': 'surrogate_local_explanation', 'targets': results}
