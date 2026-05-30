from __future__ import annotations
from pathlib import Path
from .config import XAIArtifactConfig
from .counterfactual_explainer import CounterfactualExplainer
from .feature_builder import build_feature_row
from .llm_payload_builder import LLMPayloadBuilder
from .policy_explainer import PolicyExplainer
from .surrogate_explainer import SurrogateExplainer
from .target_selector import select_xai_target_step
from .trajectory_explainer import TrajectoryExplainer
class XAIService:
    def __init__(self, artifact_root: str | Path):
        self.config=XAIArtifactConfig(artifact_root=Path(artifact_root)); self.surrogate_explainer=SurrogateExplainer(self.config); self.policy_explainer=PolicyExplainer(self.config); self.trajectory_explainer=TrajectoryExplainer(); self.counterfactual_explainer=CounterfactualExplainer(); self.llm_payload_builder=LLMPayloadBuilder()
    def explain_optimized_airfoil(self, algorithm, user_input, trajectory, optimized_result=None, solver_fn=None, top_k=8):
        target_step=select_xai_target_step(trajectory, optimized_result); feature_row=build_feature_row(user_input, target_step)
        surrogate_xai=self.surrogate_explainer.explain(feature_row, top_k); policy_xai=self.policy_explainer.explain(algorithm, feature_row, top_k); trajectory_xai=self.trajectory_explainer.explain(trajectory, target_step); counterfactual_xai=self.counterfactual_explainer.explain(target_step, user_input, solver_fn)
        llm_payload=self.llm_payload_builder.build(algorithm, target_step, surrogate_xai, policy_xai, trajectory_xai, counterfactual_xai)
        return {'available': True, 'algorithm': algorithm.lower(), 'target_step': target_step, 'feature_row': feature_row, 'surrogate_xai': surrogate_xai, 'policy_xai': policy_xai, 'trajectory_xai': trajectory_xai, 'counterfactual_xai': counterfactual_xai, 'llm_payload': llm_payload}
