from __future__ import annotations

from copy import deepcopy

from .feature_builder import extract_cst
from .target_selector import is_step_feasible
from .utils import safe_float


class CounterfactualExplainer:
    def __init__(self, perturbation_rates=(-0.03, -0.01, 0.01, 0.03)):
        self.perturbation_rates = perturbation_rates

    def explain(self, target_step: dict, user_input: dict, solver_fn, max_results: int = 12) -> dict:
        if solver_fn is None:
            return {"available": False, "reason": "counterfactual için solver_fn verilmedi"}

        base_cst = extract_cst(target_step)
        aoa = safe_float(target_step.get("AoA", user_input.get("aoa", user_input.get("AoA", 0.0))), 0.0)
        re = safe_float(target_step.get("Re", user_input.get("reynolds", user_input.get("Re", user_input.get("re", 1e6)))), 1e6)
        base = {
            "CL": safe_float(target_step.get("CL", target_step.get("cl")), None),
            "CD": safe_float(target_step.get("CD", target_step.get("cd")), None),
            "CM": safe_float(target_step.get("CM", target_step.get("cm")), None),
            "CL_CD": safe_float(target_step.get("CL_CD", target_step.get("cl_cd")), None),
            "t_c": safe_float(target_step.get("t_c", target_step.get("tc")), None),
        }

        names = ["CST_u1", "CST_u2", "CST_u3", "CST_u4", "CST_l1", "CST_l2", "CST_l3", "CST_l4"]
        rows = []
        for i, name in enumerate(names):
            for rate in self.perturbation_rates:
                new_cst = deepcopy(base_cst)
                delta = new_cst[i] * rate if abs(new_cst[i]) > 1e-8 else rate
                new_cst[i] += delta
                try:
                    pred = solver_fn(new_cst, aoa, re)
                except Exception as exc:
                    rows.append({
                        "change": f"{name} {rate:+.0%}",
                        "feature": name,
                        "rate": float(rate),
                        "available": False,
                        "reason": str(exc),
                    })
                    continue

                values = {
                    "CL": safe_float(pred.get("CL", pred.get("cl")), None),
                    "CD": safe_float(pred.get("CD", pred.get("cd")), None),
                    "CM": safe_float(pred.get("CM", pred.get("cm")), None),
                    "CL_CD": safe_float(pred.get("CL_CD", pred.get("cl_cd")), None),
                    "t_c": safe_float(pred.get("t_c", pred.get("tc")), None),
                }
                row = {
                    "change": f"{name} {rate:+.0%}",
                    "feature": name,
                    "rate": float(rate),
                    "available": True,
                    "constraint_status": "safe" if is_step_feasible({**values, "is_geometry_valid": pred.get("is_geometry_valid", True), "solver_status": pred.get("solver_status", "ok")}) else "risky",
                    **values,
                }
                for metric, value in values.items():
                    row[f"delta_{metric}"] = value - base[metric] if value is not None and base[metric] is not None else None
                rows.append(row)

        valid = [r for r in rows if r.get("available")]
        clcd_valid = [r for r in valid if r.get("delta_CL_CD") is not None]
        return {
            "available": True,
            "base": base,
            "all_evaluated_count": len(rows),
            "failed_count": len([r for r in rows if not r.get("available")]),
            "safe_count": len([r for r in valid if r.get("constraint_status") == "safe"]),
            "risky_count": len([r for r in valid if r.get("constraint_status") == "risky"]),
            "top_sensitive_changes": sorted(clcd_valid, key=lambda r: abs(r["delta_CL_CD"]), reverse=True)[:max_results],
            "evaluations": rows,
        }
