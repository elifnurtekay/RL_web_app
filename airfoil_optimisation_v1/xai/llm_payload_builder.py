from __future__ import annotations


class LLMPayloadBuilder:
    def build(self, algorithm, target_step, surrogate_xai, policy_xai, trajectory_xai, counterfactual_xai):
        constraint_status = {
            "is_feasible": target_step.get("is_feasible"),
            "is_CM_feasible": target_step.get("is_CM_feasible"),
            "is_tc_feasible": target_step.get("is_tc_feasible"),
            "is_geometry_valid": target_step.get("is_geometry_valid"),
            "solver_status": target_step.get("solver_status"),
            "risk_flag": target_step.get("risk_flag"),
        }
        return {
            "algorithm": str(algorithm).upper(),
            "selected_step": {
                "step_id": target_step.get("step_id", target_step.get("step")),
                "selection_reason": target_step.get("selection_reason"),
                "CL": target_step.get("CL", target_step.get("cl")),
                "CD": target_step.get("CD", target_step.get("cd")),
                "CM": target_step.get("CM", target_step.get("cm")),
                "CL_CD": target_step.get("CL_CD", target_step.get("cl_cd")),
                "t_c": target_step.get("t_c", target_step.get("tc")),
            },
            "constraint_status": constraint_status,
            "surrogate_xai_top_features": self._compact_xai(surrogate_xai),
            "policy_xai_top_features": self._compact_xai(policy_xai),
            "trajectory_summary": trajectory_xai,
            "counterfactual_sensitivity_summary": self._compact_counterfactual(counterfactual_xai),
            "instruction": (
                "Bu XAI JSON verilerine dayanarak optimize edilmiş airfoil tasarımını teknik fakat anlaşılır şekilde açıkla. "
                "Kesin olmayan iddia kurma. CM ve t/c kısıt durumunu belirt. SHAP değerlerini kesin fiziksel sebep değil, "
                "model açıklaması olarak ifade et."
            ),
        }

    def _compact_xai(self, xai_block):
        compact = {}
        for target, result in xai_block.get("targets", {}).items():
            if not result.get("available"):
                compact[target] = {"available": False, "reason": result.get("reason")}
                continue
            shap = result.get("shap", {})
            compact[target] = {
                "prediction": result.get("prediction"),
                "top_positive": shap.get("top_positive", [])[:5],
                "top_negative": shap.get("top_negative", [])[:5],
                "top_overall": shap.get("top_overall", [])[:5],
            }
        return compact

    def _compact_counterfactual(self, xai_block):
        if not xai_block.get("available"):
            return {"available": False, "reason": xai_block.get("reason")}
        return {
            "available": True,
            "base": xai_block.get("base"),
            "all_evaluated_count": xai_block.get("all_evaluated_count"),
            "failed_count": xai_block.get("failed_count"),
            "safe_count": xai_block.get("safe_count"),
            "risky_count": xai_block.get("risky_count"),
            "top_sensitive_changes": xai_block.get("top_sensitive_changes", [])[:8],
        }
