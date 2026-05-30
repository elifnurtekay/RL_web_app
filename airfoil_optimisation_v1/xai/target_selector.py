from __future__ import annotations

from .config import CM_MAX, CM_MIN, TC_MAX, TC_MIN
from .utils import safe_bool, safe_float


def _solver_success(step: dict) -> bool:
    status = str(step.get("solver_status", "")).strip().lower()
    return status in {"", "ok", "valid", "non_positive_cd_clipped"}


def is_step_feasible(step: dict) -> bool:
    if step.get("is_feasible") is not None:
        return bool(step.get("is_feasible"))
    cm = safe_float(step.get("CM", step.get("cm")), None)
    tc = safe_float(step.get("t_c", step.get("tc")), None)
    valid = safe_bool(step.get("is_geometry_valid"), True)
    return bool(valid and _solver_success(step) and cm is not None and tc is not None and CM_MIN <= cm <= CM_MAX and TC_MIN <= tc <= TC_MAX)


def _is_valid_geometry(step: dict) -> bool:
    return safe_bool(step.get("is_geometry_valid"), False) and _solver_success(step)


def _score_step(step: dict) -> float:
    cl_cd = safe_float(step.get("CL_CD", step.get("cl_cd")), 0.0) or 0.0
    reward = safe_float(step.get("reward_total", step.get("reward")), 0.0) or 0.0
    penalty = safe_float(step.get("penalty_total"), 0.0) or 0.0
    return float(cl_cd + 0.01 * reward - penalty)


def select_xai_target_step(trajectory: list[dict], optimized_result: dict | None = None) -> dict:
    if not trajectory:
        if optimized_result:
            return {**optimized_result, "selection_reason": "optimized_result_only", "risk_flag": True}
        raise ValueError("XAI için trajectory veya optimized_result gerekli.")

    final = trajectory[-1]
    if is_step_feasible(final):
        return {**final, "selection_reason": "final_feasible_step", "risk_flag": False}

    feasible = [s for s in trajectory if is_step_feasible(s)]
    if feasible:
        return {
            **max(feasible, key=_score_step),
            "selection_reason": "best_feasible_step_because_final_not_feasible",
            "risk_flag": False,
        }

    valid = [s for s in trajectory if _is_valid_geometry(s)]
    if valid:
        return {
            **max(valid, key=_score_step),
            "selection_reason": "best_valid_step_no_feasible_step",
            "risk_flag": True,
        }

    return {**final, "selection_reason": "no_valid_step_available_final_step_risky", "risk_flag": True}
