from __future__ import annotations

from .target_selector import is_step_feasible
from .utils import safe_float


def _series(trajectory, *keys):
    vals = []
    for step in trajectory:
        value = None
        for key in keys:
            if key in step:
                value = step.get(key)
                break
        f = safe_float(value, None)
        if f is not None:
            vals.append(f)
    return vals


def _summary(vals):
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


class TrajectoryExplainer:
    def explain(self, trajectory, selected_step):
        if not trajectory:
            return {"available": False, "reason": "trajectory boş"}

        initial, final = trajectory[0], trajectory[-1]
        clcd = _series(trajectory, "CL_CD", "cl_cd")
        cm_values = _series(trajectory, "CM", "cm")
        tc_values = _series(trajectory, "t_c", "tc")
        rewards = _series(trajectory, "reward_total", "reward")
        penalties = _series(trajectory, "penalty_total")
        actions = _series(trajectory, "action_norm")
        feasible_count = sum(1 for s in trajectory if is_step_feasible(s))
        violation_count = len(trajectory) - feasible_count
        init_clcd = safe_float(initial.get("CL_CD", initial.get("cl_cd")), None)
        final_clcd = safe_float(final.get("CL_CD", final.get("cl_cd")), None)
        selected_clcd = safe_float(selected_step.get("CL_CD", selected_step.get("cl_cd")), None)
        return {
            "available": True,
            "num_steps": len(trajectory),
            "selected_step_id": selected_step.get("step_id", selected_step.get("step")),
            "selection_reason": selected_step.get("selection_reason"),
            "initial_CL_CD": init_clcd,
            "final_CL_CD": final_clcd,
            "selected_CL_CD": selected_clcd,
            "best_CL_CD_in_trajectory": max(clcd) if clcd else None,
            "delta_selected_vs_initial_CL_CD": selected_clcd - init_clcd if selected_clcd is not None and init_clcd is not None else None,
            "CL_CD_summary": _summary(clcd),
            "CM_summary": _summary(cm_values),
            "t_c_summary": _summary(tc_values),
            "reward_summary": _summary(rewards),
            "penalty_summary": _summary(penalties),
            "action_norm_summary": _summary(actions),
            "feasible_step_count": feasible_count,
            "constraint_violation_step_count": violation_count,
            "final_is_feasible": is_step_feasible(final),
            "selected_is_feasible": is_step_feasible(selected_step),
        }
