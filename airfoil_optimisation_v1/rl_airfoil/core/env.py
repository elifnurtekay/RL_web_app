from __future__ import annotations

from typing import Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_airfoil.config.schema import ExperimentConfig
from rl_airfoil.evaluators.base import Evaluator


class AirfoilEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: ExperimentConfig, evaluator: Evaluator):
        super().__init__()

        self.cfg = cfg
        self.evaluator = evaluator
        self.max_steps = cfg.episode_max_steps

        self.action_space = spaces.Box(
            low=np.float32(cfg.action_range[0]),
            high=np.float32(cfg.action_range[1]),
            shape=(8,),
            dtype=np.float32,
        )

        # Observation:
        # 8 CST + AoA + Re_norm + log10_Re + CL + CD + CM + CL/CD + t/c = 16
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.cst = np.asarray(cfg.initial_cst, dtype=np.float32)
        self.last_aero = None

    def _safe_cl_cd(self, cl: float, cd: float) -> float:
        return float(cl / max(float(cd), self.cfg.cd_lower_bound))

    def _obs(self) -> np.ndarray:
        if self.last_aero is None:
            cl, cd, cm, tc = 0.0, 1.0, 0.0, 0.0
        else:
            cl = float(self.last_aero.cl)
            cd = float(self.last_aero.cd)
            cm = float(self.last_aero.cm)
            tc = float(self.last_aero.tc)

        cl_cd = self._safe_cl_cd(cl, cd)

        # RL actor/critic observation için normalize edilmiş Reynolds sayısı.
        # Surrogate evaluator yine ham Re kullanır; yalnızca RL observation ölçeği düzeltilir.
        re_norm = float(self.cfg.re / 1e6)
        log10_re = float(np.log10(self.cfg.re))

        return np.array(
            [
                *self.cst.tolist(),
                float(self.cfg.aoa),
                re_norm,
                log10_re,
                cl,
                cd,
                cm,
                cl_cd,
                tc,
            ],
            dtype=np.float32,
        )

    def _sample_initial_cst(self) -> np.ndarray:
        base = np.asarray(self.cfg.initial_cst, dtype=np.float32)
        noise_std = float(self.cfg.initial_cst_noise_std)

        if noise_std <= 0.0:
            return base.copy()

        for _ in range(100):
            candidate = base + self.np_random.normal(
                loc=0.0,
                scale=noise_std,
                size=8,
            ).astype(np.float32)

            candidate = np.clip(
                candidate,
                self.cfg.cst_bounds[0],
                self.cfg.cst_bounds[1],
            ).astype(np.float32)

            out = self.evaluator.evaluate(candidate, self.cfg.aoa, self.cfg.re)

            if out.is_geometry_valid:
                return candidate

        return base.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.cst = self._sample_initial_cst()
        self.last_aero = self.evaluator.evaluate(self.cst, self.cfg.aoa, self.cfg.re)

        return self._obs(), {"done_reason": "reset"}

    def _constraint_violations(self, cm: float, tc: float) -> Dict[str, float]:
        cm_lo, cm_hi = self.cfg.constraints.cm_bounds
        tc_lo, tc_hi = self.cfg.constraints.tc_bounds

        return {
            "CM_lower_violation": float(max(0.0, cm_lo - cm)),
            "CM_upper_violation": float(max(0.0, cm - cm_hi)),
            "tc_lower_violation": float(max(0.0, tc_lo - tc)),
            "tc_upper_violation": float(max(0.0, tc - tc_hi)),
        }

    def _reward_components(self, aero, cl_cd: float, action: np.ndarray) -> Dict[str, float]:
        v = self._constraint_violations(float(aero.cm), float(aero.tc))
        w = self.cfg.reward_weights

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_l2_raw = float(np.sum(np.square(action)))
        reward_action_penalty = float(w.w_action * action_l2_raw)

        # CL/CD oranı çok büyüyebildiği için sign-preserving log scaling.
        reward_cl_cd_term = float(w.w1 * np.sign(cl_cd) * np.log1p(abs(cl_cd)))

        cm_penalty_raw = float(
            v["CM_lower_violation"] ** 2
            + v["CM_upper_violation"] ** 2
        )

        tc_penalty_raw = float(
            v["tc_lower_violation"] ** 2
            + v["tc_upper_violation"] ** 2
        )

        geom = aero.geometry_features or {}

        min_local_thickness = float(geom.get("min_local_thickness", 0.0))
        min_required = float(self.cfg.geometry.min_local_thickness_required)

        local_violation = float(
            max(
                0.0,
                min_required - min_local_thickness,
            )
        )

        # Önemli düzeltme:
        # local_violation chord ölçeğinde çok küçük olduğu için karesini doğrudan almak
        # penalty'yi neredeyse etkisiz yapar.
        # Bu yüzden violation, minimum gerekli lokal kalınlığa göre normalize edilir.
        local_violation_normalized = float(
            local_violation / max(min_required, 1e-8)
        )

        # TD3 critic stabilitesi için local thickness penalty sınırlanır.
        # Aksi halde küçük geometrik ihlaller çok büyük Q target sıçramaları oluşturabilir.
        local_violation_normalized_clipped = float(
            np.clip(local_violation_normalized, 0.0, 1.0)
        )

        reward_cm_penalty = float(w.w2 * cm_penalty_raw)
        reward_tc_penalty = float(w.w3 * tc_penalty_raw)

        reward_local_penalty = float(
            w.w_local_thickness * (local_violation_normalized_clipped ** 2)
        )

        reward_invalid_geometry_penalty = (
            float(self.cfg.invalid_geometry_penalty)
            if not bool(aero.is_geometry_valid)
            else 0.0
        )

        reward_solver_error_penalty = (
            float(self.cfg.solver_error_penalty)
            if aero.solver_status not in {"ok", "invalid_geometry", "non_positive_cd_clipped"}
            else 0.0
        )

        penalty_total = float(
            reward_cm_penalty
            + reward_tc_penalty
            + reward_local_penalty
            + reward_invalid_geometry_penalty
            + reward_solver_error_penalty
            + reward_action_penalty
        )

        reward_total = float(reward_cl_cd_term - penalty_total)

        return {
            "reward_total": float(reward_total),
            "reward_objective_term": float(reward_cl_cd_term),
            "reward_CL_CD_term": float(reward_cl_cd_term),

            "reward_CM_penalty": float(reward_cm_penalty),
            "reward_tc_penalty": float(reward_tc_penalty),
            "reward_local_thickness_penalty": float(reward_local_penalty),
            "reward_invalid_geometry_penalty": float(reward_invalid_geometry_penalty),
            "reward_solver_error_penalty": float(reward_solver_error_penalty),
            "reward_action_penalty": float(reward_action_penalty),

            "penalty_total": float(penalty_total),

            "action_l2_penalty_raw": float(action_l2_raw),

            "local_thickness_violation": float(local_violation),
            "local_thickness_violation_normalized": float(local_violation_normalized),
            "local_thickness_violation_normalized_clipped": float(local_violation_normalized_clipped),
            "min_local_thickness": float(min_local_thickness),
            "min_local_thickness_required": float(min_required),

            **v,
        }

    def _invalid_geometry_reason(self, aero) -> str:
        geom = aero.geometry_features or {}
        min_local_thickness = float(geom.get("min_local_thickness", 0.0))

        if min_local_thickness < 0.0:
            return "invalid_geometry_negative_local_thickness"

        return "invalid_geometry"

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(8)
        action = np.clip(
            action,
            self.cfg.action_range[0],
            self.cfg.action_range[1],
        )

        prev_cst = self.cst.copy()

        next_cst = prev_cst + action * float(self.cfg.action_scale)
        next_cst = np.clip(
            next_cst,
            self.cfg.cst_bounds[0],
            self.cfg.cst_bounds[1],
        ).astype(np.float32)

        self.cst = next_cst

        aero = self.evaluator.evaluate(self.cst, self.cfg.aoa, self.cfg.re)
        self.last_aero = aero

        cl_cd = self._safe_cl_cd(float(aero.cl), float(aero.cd))
        reward_parts = self._reward_components(aero, cl_cd, action)

        is_cm_feasible = bool(
            reward_parts["CM_lower_violation"] == 0.0
            and reward_parts["CM_upper_violation"] == 0.0
        )

        is_tc_feasible = bool(
            reward_parts["tc_lower_violation"] == 0.0
            and reward_parts["tc_upper_violation"] == 0.0
        )

        terminated = False
        truncated = False
        done_reason = "running"

        if aero.solver_status == "nan_output":
            terminated = True
            done_reason = "nan_output"

        elif aero.solver_status == "solver_error":
            terminated = True
            done_reason = "solver_error"

        elif not bool(aero.is_geometry_valid):
            terminated = True
            done_reason = self._invalid_geometry_reason(aero)

        elif self.step_count >= self.max_steps:
            truncated = True

            geom = aero.geometry_features or {}
            min_local_thickness = float(geom.get("min_local_thickness", 0.0))
            local_thickness_violation = float(
                reward_parts.get("local_thickness_violation", 0.0)
            )

            is_solver_ok = str(aero.solver_status) in {
                "ok",
                "valid",
                "",
                "non_positive_cd_clipped",
            }

            is_final_feasible_at_max_step = (
                is_cm_feasible
                and is_tc_feasible
                and bool(aero.is_geometry_valid)
                and is_solver_ok
                and local_thickness_violation <= 0.0
                and min_local_thickness >= float(
                    self.cfg.geometry.min_local_thickness_required
                )
            )

            done_reason = (
                "max_episode_steps_feasible"
                if is_final_feasible_at_max_step
                else "max_episode_steps"
            )

        geom = aero.geometry_features or {}

        info: Dict = {
            "prev_cst": prev_cst,
            "next_cst": self.cst.copy(),
            "delta_cst": self.cst - prev_cst,

            "CL": float(aero.cl),
            "CD": float(aero.cd),
            "CM": float(aero.cm),
            "t_c": float(aero.tc),
            "CL_CD": float(cl_cd),

            "CL_pred": float(aero.cl),
            "CD_pred": float(aero.cd),
            "CM_pred": float(aero.cm),
            "CL_CD_pred": float(cl_cd),

            "is_CM_feasible": bool(is_cm_feasible),
            "is_tc_feasible": bool(is_tc_feasible),
            "is_geometry_valid": bool(aero.is_geometry_valid),

            "action_norm": float(np.linalg.norm(action)),
            "upper_action_norm": float(np.linalg.norm(action[:4])),
            "lower_action_norm": float(np.linalg.norm(action[4:])),
            "action_max_abs": float(np.max(np.abs(action))),
            "action_saturation_count": int(
                np.sum(np.isclose(np.abs(action), 1.0, atol=1e-4))
            ),

            "solver_status": str(aero.solver_status),
            "solver_error_message": str(aero.solver_error_message),
            "aero_wall_time_step_sec": float(aero.runtime_ms / 1000.0),
            "xfoil_converged": bool(geom.get("xfoil_converged", False))
            if "xfoil_converged" in geom
            else "",
            "xfoil_error_message": str(geom.get("xfoil_error_message", "")),
            "xfoil_runtime_ms": float(aero.runtime_ms),

            "done_reason": done_reason,

            # Reward decomposition
            **reward_parts,

            # Geometry features
            "max_thickness": float(geom.get("max_thickness", aero.tc)),
            "x_max_thickness": float(geom.get("x_max_thickness", np.nan)),
            "max_camber": float(geom.get("max_camber", np.nan)),
            "x_max_camber": float(geom.get("x_max_camber", np.nan)),
            "leading_edge_radius_proxy": float(geom.get("leading_edge_radius_proxy", np.nan)),
            "trailing_edge_thickness": float(geom.get("trailing_edge_thickness", np.nan)),
            "upper_surface_curvature_mean": float(geom.get("upper_surface_curvature_mean", np.nan)),
            "lower_surface_curvature_mean": float(geom.get("lower_surface_curvature_mean", np.nan)),
            "surface_smoothness": float(geom.get("surface_smoothness", np.nan)),
            "min_local_thickness": float(geom.get("min_local_thickness", reward_parts["min_local_thickness"])),
            "area_proxy": float(geom.get("area_proxy", np.nan)),
        }

        return (
            self._obs(),
            float(reward_parts["reward_total"]),
            terminated,
            truncated,
            info,
        )