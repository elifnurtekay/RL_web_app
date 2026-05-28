from __future__ import annotations
import csv
import torch.nn.functional as F
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import Optional
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from rl_airfoil.config.schema import ExperimentConfig, create_run_dir, write_experiment_metadata
from rl_airfoil.core.env import AirfoilEnv
from rl_airfoil.evaluators.surrogate import SurrogateEvaluator
from rl_airfoil.evaluators.xfoil import XFOILEvaluator
from rl_airfoil.logging.xai_logger import CSVLogger


def _rl_model_filename(cfg) -> str:
    algorithm = str(cfg.algorithm).lower()
    evaluator = str(cfg.evaluator).lower()

    if evaluator == "xfoil":
        return f"{algorithm}_xfoil.zip"

    surrogate_name = str(cfg.surrogate_model_name).lower()
    return f"{algorithm}_{evaluator}_{surrogate_name}.zip"

def _linear_schedule(start: float, end: float):
    """
    SB3 schedule fonksiyonu.
    progress_remaining eğitim başında 1.0, sonunda 0.0 olur.
    Bu nedenle başlangıçta start, sonda end döndürür.
    """
    def schedule(progress_remaining: float) -> float:
        return float(end + (start - end) * progress_remaining)

    return schedule

def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {"true", "1", "yes"}


def _standardize_done_reason(info: dict, done: bool = False) -> str:
    """
    Step-level ve episode-level done_reason standardizasyonu.

    Eğer episode max step ile bittiyse ve final state gerçekten feasible ise:
        max_episode_steps -> max_episode_steps_feasible

    Solver error, invalid geometry vb. durumlara dokunmaz.
    """
    if not isinstance(info, dict):
        return ""

    done_reason = str(info.get("done_reason", ""))

    if done and done_reason == "max_episode_steps" and _is_feasible_info(info):
        return "max_episode_steps_feasible"

    return done_reason


def _safe_get(row, key: str, default=np.nan):
    if not isinstance(row, dict):
        return default
    return row.get(key, default)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].map(_as_bool)

def _safe_float_value(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _is_constraint_feasible_info(info: dict) -> bool:
    """
    Paper / ana feasibility tanımı:
    CM constraint + t/c constraint + geometry valid.

    Local thickness safety margin bu tanıma dahil değildir.
    Bu alan ana PPO/TD3/SAC karşılaştırması için kullanılmalıdır.
    """
    return bool(
        _as_bool(info.get("is_CM_feasible", False))
        and _as_bool(info.get("is_tc_feasible", False))
        and _as_bool(info.get("is_geometry_valid", False))
    )


def _is_solver_success_info(info: dict) -> bool:
    """
    Solver başarısı kontrolü.

    Surrogate için:
        solver_status boş / ok / valid ise başarılı kabul edilir.

    XFOIL için:
        solver_status ok/valid olmalı
        ve xfoil_converged alanı boş değilse True olmalı.

    Böylece solver_error veya xfoil_converged=False olan step/episode
    feasible sayılmaz.
    """
    if not isinstance(info, dict):
        return False

    solver_status = str(info.get("solver_status", "")).strip().lower()

    if solver_status not in {"", "ok", "valid"}:
        return False

    xfoil_converged_raw = info.get("xfoil_converged", "")

    # Surrogate evaluator'da bu alan genelde boş olur.
    # Boşsa solver_status yeterli kabul edilir.
    if str(xfoil_converged_raw).strip() == "":
        return True

    return _as_bool(xfoil_converged_raw)


def _is_feasible_info(info: dict) -> bool:
    """
    Ana raporlama feasibility tanımı:
    CM + t/c + geometry valid + solver success.

    XFOIL solver_error durumları burada feasible sayılmaz.
    """
    return bool(
        _is_constraint_feasible_info(info)
        and _is_solver_success_info(info)
    )


def _is_local_margin_safe_info(
    info: dict,
    cfg: ExperimentConfig,
    eps: float = 1e-8,
) -> bool:
    """
    Ek geometri güvenlik metriği:
    min_local_thickness >= cfg.geometry.min_local_thickness_required.

    Bu ana paper feasibility değildir.
    XAI ve geometri güvenliği analizi için ayrıca raporlanmalıdır.
    """
    local_violation = _safe_float_value(
        info.get("local_thickness_violation", 0.0),
        default=0.0,
    )

    min_local_thickness = _safe_float_value(
        info.get("min_local_thickness", np.nan),
        default=np.nan,
    )

    local_ok = local_violation <= eps

    if np.isfinite(min_local_thickness):
        local_ok = local_ok and (
            min_local_thickness
            >= float(cfg.geometry.min_local_thickness_required) - eps
        )

    return bool(local_ok)


def _is_strict_safe_info(
    info: dict,
    cfg: ExperimentConfig,
    eps: float = 1e-8,
) -> bool:
    """
    Strict safety tanımı:
    Feasible + local thickness safety margin.
    """
    return bool(
        _is_feasible_info(info)
        and _is_local_margin_safe_info(info, cfg, eps=eps)
    )


def _is_strict_feasible_info(
    info: dict,
    cfg: ExperimentConfig,
    eps: float = 1e-8,
) -> bool:
    """
    Geriye dönük uyumluluk alias'ı.

    Eski kodda _is_strict_feasible_info kullanıldığı için bu fonksiyonu
    tamamen silmiyoruz. Yeni isimlendirmede bunun karşılığı
    _is_strict_safe_info'dur.
    """
    return _is_strict_safe_info(info, cfg, eps=eps)


def _constraint_feasible_mask(df: pd.DataFrame) -> pd.Series:
    """
    Paper/ana feasible step maskesi.
    """
    if len(df) == 0:
        return pd.Series(False, index=df.index)

    return (
        df.get("is_CM_feasible", False).map(_as_bool)
        & df.get("is_tc_feasible", False).map(_as_bool)
        & df.get("is_geometry_valid", False).map(_as_bool)
    )


def _solver_success_mask(df: pd.DataFrame) -> pd.Series:
    """
    DataFrame için solver success maskesi.
    XFOIL'de xfoil_converged=False olan satırları dışlar.
    Surrogate'de xfoil_converged boş olduğu için solver_status ok/boş yeterlidir.
    """
    if len(df) == 0:
        return pd.Series(False, index=df.index)

    if "solver_status" in df.columns:
        solver_status = df["solver_status"].fillna("").astype(str).str.strip().str.lower()
        solver_ok = solver_status.isin(["", "ok", "valid"])
    else:
        solver_ok = pd.Series(True, index=df.index)

    if "xfoil_converged" in df.columns:
        xfoil_raw = df["xfoil_converged"].fillna("").astype(str).str.strip()
        xfoil_empty = xfoil_raw == ""
        xfoil_ok = xfoil_empty | xfoil_raw.map(_as_bool)
    else:
        xfoil_ok = pd.Series(True, index=df.index)

    return solver_ok & xfoil_ok


def _numeric_aero_mask(df: pd.DataFrame) -> pd.Series:
    """
    CL/CD ve CD sayısal ve fiziksel olarak kullanılabilir mi?
    """
    if len(df) == 0:
        return pd.Series(False, index=df.index)

    cl_cd = pd.to_numeric(df.get("CL_CD", np.nan), errors="coerce")
    cd = pd.to_numeric(df.get("CD", np.nan), errors="coerce")

    return (
        cl_cd.notna()
        & np.isfinite(cl_cd)
        & cd.notna()
        & np.isfinite(cd)
        & (cd > 0.0)
    )


def _feasible_mask(df: pd.DataFrame) -> pd.Series:
    """
    Ana raporlama feasible step maskesi:
    CM + t/c + geometry valid + solver success + numeric aero output.
    """
    return (
        _constraint_feasible_mask(df)
        & _solver_success_mask(df)
        & _numeric_aero_mask(df)
    )


def _strict_safe_mask(df: pd.DataFrame, cfg: ExperimentConfig, eps: float = 1e-8) -> pd.Series:
    """
    Local thickness safety dahil strict safe step maskesi.
    """
    if len(df) == 0:
        return pd.Series(False, index=df.index)

    local_violation = pd.to_numeric(
        df.get("local_thickness_violation", 0.0),
        errors="coerce",
    ).fillna(0.0)

    min_local_thickness = pd.to_numeric(
        df.get("min_local_thickness", np.nan),
        errors="coerce",
    )

    mask = (
        _feasible_mask(df)
        & (local_violation <= eps)
    )

    if min_local_thickness.notna().any():
        mask = mask & (
            min_local_thickness
            >= float(cfg.geometry.min_local_thickness_required) - eps
        )

    return mask


def _best_row_by_cl_cd(df: pd.DataFrame):
    """
    Verilen DataFrame içinde CL_CD maksimum olan satırı dict olarak döndürür.
    """
    if len(df) == 0 or "CL_CD" not in df.columns:
        return None

    tmp = df.copy()
    tmp["CL_CD_numeric"] = pd.to_numeric(tmp["CL_CD"], errors="coerce")
    tmp = tmp[tmp["CL_CD_numeric"].notna()]

    if len(tmp) == 0:
        return None

    return tmp.loc[tmp["CL_CD_numeric"].idxmax()].to_dict()


class TD3TrainingDiagnosticsCallback(BaseCallback):
    """
    TD3 için XAI uyumlu training callback.

    Bu callback iki işi birlikte yapar:
    1. Her environment step'i için train_rollout_step_logs.csv verisini RAM'de tutar.
    2. Belirli aralıklarla replay buffer'dan batch örnekleyerek
       training_update_logs.csv içine TD3 critic/actor diagnostic metriklerini yazar.
    """

    def __init__(
        self,
        log_path: Path,
        cfg=None,
        log_every: int = 100,
        batch_size: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.log_path = Path(log_path)
        self.cfg = cfg
        self.log_every = int(log_every)
        self.batch_size = int(batch_size)
        self.update_id = 0

        self.columns = [
            "update_id",
            "critic_loss_Q1",
            "critic_loss_Q2",
            "actor_loss",
            "target_Q_mean",
            "target_Q_std",
            "Q1_mean",
            "Q2_mean",
            "policy_delay_step",
            "learning_rate_actor",
            "learning_rate_critic",
            "gradient_norm_actor",
            "gradient_norm_critic",
        ]

        # Eski runner.py sonunda kullanılan attribute.
        self.train_steps = []

        # Olası eski isimlerle uyumlu olsun diye birkaç alias bırakıyoruz.
        self.episode_rows = []

        # Eski runner.py attribute isimleriyle uyumluluk için alias'lar
        self.ep_rows = self.episode_rows
        self.episodes = self.episode_rows
        self.episode_summaries = self.episode_rows
        self.train_episodes = self.episode_rows

        self._episode_id = 0
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_penalty = 0.0
        self._episode_violation_count = 0
        self._episode_best_cl_cd = -float("inf")
        self._episode_initial_cl_cd = None
        # Son environment step'inden gelen info bilgisini train_metrics için saklar.
        self.last_info = {}
        self.last_step_row = {}

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    @staticmethod
    def _first(value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else default
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value[0]
        return value

    @staticmethod
    def _float(value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _bool(value) -> bool:
        try:
            if isinstance(value, np.ndarray):
                return bool(value.item()) if value.shape == () else bool(value[0])
            return bool(value)
        except Exception:
            return False

    def _current_lr(self, optimizer) -> float:
        try:
            return float(optimizer.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def _grad_norm(self, module: torch.nn.Module) -> float:
        total_sq = 0.0
        has_grad = False

        for param in module.parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = float(param.grad.detach().data.norm(2).cpu())
                total_sq += grad_norm ** 2

        if not has_grad:
            return float("nan")

        return float(total_sq ** 0.5)

    def _log_train_rollout_step(self) -> None:
        infos = self.locals.get("infos", [{}])
        info = self._first(infos, default={}) or {}

        # train_td3 sonunda train_metrics üretmek için son step bilgisini sakla.
        if isinstance(info, dict):
            self.last_info = dict(info)
        else:
            self.last_info = {}

        rewards = self.locals.get("rewards", [info.get("reward_total", 0.0)])
        dones = self.locals.get("dones", [False])
        actions = self.locals.get("actions", None)

        reward_value = self._float(self._first(rewards, info.get("reward_total", 0.0)))
        done = self._bool(self._first(dones, False))

        action = self._first(actions, default=np.zeros(8, dtype=np.float32))
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        prev_cst = np.asarray(info.get("prev_cst", np.full(8, np.nan)), dtype=np.float32).reshape(-1)
        next_cst = np.asarray(info.get("next_cst", np.full(8, np.nan)), dtype=np.float32).reshape(-1)
        delta_cst = np.asarray(info.get("delta_cst", next_cst - prev_cst), dtype=np.float32).reshape(-1)

        if prev_cst.size < 8:
            prev_cst = np.pad(prev_cst, (0, 8 - prev_cst.size), constant_values=np.nan)
        if next_cst.size < 8:
            next_cst = np.pad(next_cst, (0, 8 - next_cst.size), constant_values=np.nan)
        if delta_cst.size < 8:
            delta_cst = np.pad(delta_cst, (0, 8 - delta_cst.size), constant_values=np.nan)
        if action.size < 8:
            action = np.pad(action, (0, 8 - action.size), constant_values=np.nan)

        cl_cd = self._float(info.get("CL_CD", 0.0))
        penalty_total = self._float(info.get("penalty_total", 0.0))

        if self._episode_step == 0:
            self._episode_initial_cl_cd = cl_cd

        self._episode_best_cl_cd = max(self._episode_best_cl_cd, cl_cd)

        action_norm = float(np.linalg.norm(action[:8])) if np.all(np.isfinite(action[:8])) else float("nan")
        row_done_reason = _standardize_done_reason(info, done=done)
        row = {
            "episode_id": self._episode_id,
            "step_id": self._episode_step,
            "global_step": self.num_timesteps,
            "algorithm": "TD3",
            "evaluator": getattr(self.cfg, "evaluator", "surrogate") if self.cfg is not None else "surrogate",

            "state_CST_u1": float(prev_cst[0]),
            "state_CST_u2": float(prev_cst[1]),
            "state_CST_u3": float(prev_cst[2]),
            "state_CST_u4": float(prev_cst[3]),
            "state_CST_l1": float(prev_cst[4]),
            "state_CST_l2": float(prev_cst[5]),
            "state_CST_l3": float(prev_cst[6]),
            "state_CST_l4": float(prev_cst[7]),

            "AoA": getattr(self.cfg, "aoa", ""),
            "Re": getattr(self.cfg, "re", ""),
            "log10_Re": np.log10(getattr(self.cfg, "re", 1.0)) if self.cfg is not None else "",

            "CL": info.get("CL", ""),
            "CD": info.get("CD", ""),
            "CM": info.get("CM", ""),
            "CL_CD": info.get("CL_CD", ""),
            "t_c": info.get("t_c", ""),

            "max_thickness": info.get("max_thickness", ""),
            "x_max_thickness": info.get("x_max_thickness", ""),
            "max_camber": info.get("max_camber", ""),
            "x_max_camber": info.get("x_max_camber", ""),
            "leading_edge_radius_proxy": info.get("leading_edge_radius_proxy", ""),
            "trailing_edge_thickness": info.get("trailing_edge_thickness", ""),
            "upper_surface_curvature_mean": info.get("upper_surface_curvature_mean", ""),
            "lower_surface_curvature_mean": info.get("lower_surface_curvature_mean", ""),
            "surface_smoothness": info.get("surface_smoothness", ""),
            "min_local_thickness": info.get("min_local_thickness", ""),
            "area_proxy": info.get("area_proxy", ""),

            "action_u1": float(action[0]),
            "action_u2": float(action[1]),
            "action_u3": float(action[2]),
            "action_u4": float(action[3]),
            "action_l1": float(action[4]),
            "action_l2": float(action[5]),
            "action_l3": float(action[6]),
            "action_l4": float(action[7]),
            "action_norm": action_norm,
            "action_max_abs": float(np.nanmax(np.abs(action[:8]))),
            "action_saturation_count": int(np.sum(np.isclose(np.abs(action[:8]), 1.0, atol=1e-4))),

            "next_CST_u1": float(next_cst[0]),
            "next_CST_u2": float(next_cst[1]),
            "next_CST_u3": float(next_cst[2]),
            "next_CST_u4": float(next_cst[3]),
            "next_CST_l1": float(next_cst[4]),
            "next_CST_l2": float(next_cst[5]),
            "next_CST_l3": float(next_cst[6]),
            "next_CST_l4": float(next_cst[7]),

            "delta_CST_u1": float(delta_cst[0]),
            "delta_CST_u2": float(delta_cst[1]),
            "delta_CST_u3": float(delta_cst[2]),
            "delta_CST_u4": float(delta_cst[3]),
            "delta_CST_l1": float(delta_cst[4]),
            "delta_CST_l2": float(delta_cst[5]),
            "delta_CST_l3": float(delta_cst[6]),
            "delta_CST_l4": float(delta_cst[7]),

            "reward_total": info.get("reward_total", reward_value),
            "reward_objective_term": info.get("reward_objective_term", ""),
            "reward_CL_CD_term": info.get("reward_CL_CD_term", ""),
            "reward_CM_penalty": info.get("reward_CM_penalty", ""),
            "reward_tc_penalty": info.get("reward_tc_penalty", ""),
            "reward_local_thickness_penalty": info.get("reward_local_thickness_penalty", ""),
            "reward_invalid_geometry_penalty": info.get("reward_invalid_geometry_penalty", ""),
            "reward_solver_error_penalty": info.get("reward_solver_error_penalty", ""),
            "reward_action_penalty": info.get("reward_action_penalty", ""),
            "penalty_total": info.get("penalty_total", ""),
            "action_l2_penalty_raw": info.get("action_l2_penalty_raw", ""),

            "CM_lower_violation": info.get("CM_lower_violation", ""),
            "CM_upper_violation": info.get("CM_upper_violation", ""),
            "tc_lower_violation": info.get("tc_lower_violation", ""),
            "tc_upper_violation": info.get("tc_upper_violation", ""),
            "local_thickness_violation": info.get("local_thickness_violation", ""),

            "is_CM_feasible": info.get("is_CM_feasible", ""),
            "is_tc_feasible": info.get("is_tc_feasible", ""),
            "is_geometry_valid": info.get("is_geometry_valid", ""),

            "done": done,
            "done_reason": row_done_reason,
            "solver_status": info.get("solver_status", ""),
            "solver_error_message": info.get("solver_error_message", ""),

            # XFOIL training diagnostics
            "xfoil_converged": info.get("xfoil_converged", ""),
            "xfoil_error_message": info.get("xfoil_error_message", ""),
            "xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),
        }

        self.train_steps.append(row)
        self.last_step_row = dict(row)

        self._episode_reward += reward_value
        self._episode_penalty += penalty_total

        violation_now = (
            self._float(info.get("CM_lower_violation", 0.0)) > 0.0
            or self._float(info.get("CM_upper_violation", 0.0)) > 0.0
            or self._float(info.get("tc_lower_violation", 0.0)) > 0.0
            or self._float(info.get("tc_upper_violation", 0.0)) > 0.0
            or not bool(info.get("is_geometry_valid", True))
        )
        if violation_now:
            self._episode_violation_count += 1

        self._episode_step += 1

        if done:
            if self.cfg is not None:
                final_is_feasible = _is_feasible_info(info)
                final_is_strict_safe = _is_strict_safe_info(info, self.cfg)
            else:
                final_is_feasible = (
                    _is_constraint_feasible_info(info)
                    and _is_solver_success_info(info)
                )
                final_is_strict_safe = final_is_feasible

            done_reason = _standardize_done_reason(info, done=done)
            self.episode_rows.append(
                {
                    "episode_id": self._episode_id,
                    "initial_CL_CD": self._episode_initial_cl_cd,
                    "final_CL_CD": cl_cd,
                    "best_CL_CD": self._episode_best_cl_cd,
                    "final_CL": info.get("CL", ""),
                    "final_CD": info.get("CD", ""),
                    "final_CM": info.get("CM", ""),
                    "final_t_c": info.get("t_c", ""),
                    "total_reward": self._episode_reward,
                    "total_penalty": self._episode_penalty,
                    "constraint_violation_count": self._episode_violation_count,
                    "done_reason": done_reason,
                    "episode_length": self._episode_step,
                    "is_final_feasible": final_is_feasible,
                    "is_final_strict_safe": final_is_strict_safe,
                    "final_min_local_thickness": info.get("min_local_thickness", np.nan),
                    "final_local_thickness_violation": info.get("local_thickness_violation", np.nan),
                    "final_xfoil_converged": info.get("xfoil_converged", ""),
                    "final_xfoil_error_message": info.get("xfoil_error_message", ""),
                    "final_xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),
                }
            )

            self._episode_id += 1
            self._episode_step = 0
            self._episode_reward = 0.0
            self._episode_penalty = 0.0
            self._episode_violation_count = 0
            self._episode_best_cl_cd = -float("inf")
            self._episode_initial_cl_cd = None

    def _log_training_diagnostics(self) -> None:
        if self.num_timesteps % self.log_every != 0:
            return

        model = self.model

        if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
            return

        replay_size = model.replay_buffer.size()

        if replay_size < max(self.batch_size, 2):
            return

        if hasattr(model, "learning_starts") and self.num_timesteps < model.learning_starts:
            return

        try:
            batch = model.replay_buffer.sample(self.batch_size, env=None)

            observations = batch.observations
            actions = batch.actions
            rewards = batch.rewards
            next_observations = batch.next_observations
            dones = batch.dones

            with torch.no_grad():
                q1, q2 = model.critic(observations, actions)

                noise = torch.randn_like(actions) * float(model.target_policy_noise)
                noise = noise.clamp(
                    -float(model.target_noise_clip),
                    float(model.target_noise_clip),
                )

                next_actions = model.actor_target(next_observations)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)

                next_q1, next_q2 = model.critic_target(next_observations, next_actions)
                next_q_min = torch.min(next_q1, next_q2)

                target_q = rewards + (1.0 - dones) * float(model.gamma) * next_q_min

            critic_loss_q1 = float(F.mse_loss(q1, target_q).detach().cpu())
            critic_loss_q2 = float(F.mse_loss(q2, target_q).detach().cpu())

            actor_actions = model.actor(observations)
            actor_q1, _ = model.critic(observations, actor_actions)
            actor_loss = float((-actor_q1.mean()).detach().cpu())

            self.update_id += 1

            row = {
                "update_id": self.update_id,
                "critic_loss_Q1": critic_loss_q1,
                "critic_loss_Q2": critic_loss_q2,
                "actor_loss": actor_loss,
                "target_Q_mean": float(target_q.mean().detach().cpu()),
                "target_Q_std": float(target_q.std(unbiased=False).detach().cpu()),
                "Q1_mean": float(q1.mean().detach().cpu()),
                "Q2_mean": float(q2.mean().detach().cpu()),
                "policy_delay_step": getattr(model, "policy_delay", ""),
                "learning_rate_actor": self._current_lr(model.actor.optimizer),
                "learning_rate_critic": self._current_lr(model.critic.optimizer),
                "gradient_norm_actor": self._grad_norm(model.actor),
                "gradient_norm_critic": self._grad_norm(model.critic),
            }

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writerow(row)

        except Exception as exc:
            if self.verbose:
                print(f"[TD3TrainingDiagnosticsCallback] skipped logging due to: {exc}")

    def _on_step(self) -> bool:
        self._log_train_rollout_step()
        self._log_training_diagnostics()
        return True


class SACTrainingDiagnosticsCallback(TD3TrainingDiagnosticsCallback):
    """
    SAC için XAI uyumlu training diagnostics logger.

    TD3TrainingDiagnosticsCallback içindeki train rollout ve episode summary
    mantığını tekrar kullanır. SAC'e özel olarak replay buffer batch'i üzerinden:
      - Q1/Q2
      - target_Q
      - critic loss
      - actor loss
      - alpha / entropy / log_prob
    hesaplar.
    """

    def __init__(
        self,
        log_path: Path,
        cfg=None,
        log_every: int = 100,
        batch_size: int = 256,
        verbose: int = 0,
    ):
        super().__init__(
            log_path=log_path,
            cfg=cfg,
            log_every=log_every,
            batch_size=batch_size,
            verbose=verbose,
        )

        self.columns = [
            "update_id",
            "critic_loss_Q1",
            "critic_loss_Q2",
            "actor_loss",
            "alpha_loss",
            "alpha_value",
            "entropy_mean",
            "log_prob_mean",
            "target_Q_mean",
            "target_Q_std",
            "Q1_mean",
            "Q2_mean",
            "learning_rate_actor",
            "learning_rate_critic",
            "gradient_norm_actor",
            "gradient_norm_critic",
        ]

    def _get_alpha(self, model) -> torch.Tensor:
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            return torch.exp(model.log_ent_coef.detach())

        if hasattr(model, "ent_coef_tensor"):
            return model.ent_coef_tensor.detach()

        return torch.tensor(1.0, device=model.device)

    def _get_alpha_loss(self, model, log_prob: torch.Tensor) -> float:
        try:
            if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
                alpha_loss = -(
                    model.log_ent_coef * (log_prob + model.target_entropy).detach()
                ).mean()
                return float(alpha_loss.detach().cpu())
        except Exception:
            pass

        return float("nan")

    def _on_step(self) -> bool:
        # TD3 callback içindeki train_rollout_step mantığını kullan.
        self._log_train_rollout_step()

        if self.num_timesteps % self.log_every != 0:
            return True

        model = self.model

        if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
            return True

        replay_size = model.replay_buffer.size()

        if replay_size < max(self.batch_size, 2):
            return True

        try:
            batch = model.replay_buffer.sample(self.batch_size, env=None)

            observations = batch.observations
            actions = batch.actions
            rewards = batch.rewards.reshape(-1, 1)
            next_observations = batch.next_observations
            dones = batch.dones.reshape(-1, 1)

            alpha = self._get_alpha(model)

            with torch.no_grad():
                # SAC target:
                # y = r + gamma * (1-done) * (min(Q1', Q2') - alpha * log pi(a'|s'))
                next_actions, next_log_prob = model.actor.action_log_prob(next_observations)
                next_log_prob = next_log_prob.reshape(-1, 1)

                next_q1, next_q2 = model.critic_target(next_observations, next_actions)
                next_q = torch.min(next_q1, next_q2)

                target_q = rewards + (1.0 - dones) * model.gamma * (
                    next_q - alpha * next_log_prob
                )

                q1, q2 = model.critic(observations, actions)

            critic_loss_q1 = F.mse_loss(q1, target_q).item()
            critic_loss_q2 = F.mse_loss(q2, target_q).item()

            # Diagnostic actor loss:
            # J_pi = E[alpha * log pi(a|s) - min(Q1,Q2)]
            actor_actions, log_prob = model.actor.action_log_prob(observations)
            log_prob_col = log_prob.reshape(-1, 1)

            q1_pi, q2_pi = model.critic(observations, actor_actions)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * log_prob_col - q_pi).mean()

            entropy_mean = float((-log_prob).mean().detach().cpu())
            log_prob_mean = float(log_prob.mean().detach().cpu())

            alpha_loss = self._get_alpha_loss(model, log_prob)

            q1_mean = float(q1.mean().detach().cpu())
            q2_mean = float(q2.mean().detach().cpu())
            target_q_mean = float(target_q.mean().detach().cpu())
            target_q_std = float(target_q.std(unbiased=False).detach().cpu())

            self.update_id += 1

            row = {
                "update_id": self.update_id,
                "critic_loss_Q1": critic_loss_q1,
                "critic_loss_Q2": critic_loss_q2,
                "actor_loss": float(actor_loss.detach().cpu()),
                "alpha_loss": alpha_loss,
                "alpha_value": float(alpha.detach().cpu()),
                "entropy_mean": entropy_mean,
                "log_prob_mean": log_prob_mean,
                "target_Q_mean": target_q_mean,
                "target_Q_std": target_q_std,
                "Q1_mean": q1_mean,
                "Q2_mean": q2_mean,
                "learning_rate_actor": self._current_lr(model.actor.optimizer),
                "learning_rate_critic": self._current_lr(model.critic.optimizer),
                "gradient_norm_actor": self._grad_norm(model.actor),
                "gradient_norm_critic": self._grad_norm(model.critic),
            }

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writerow(row)

        except Exception as exc:
            if self.verbose:
                print(f"[SACTrainingDiagnosticsCallback] skipped logging due to: {exc}")

        return True


class PPOTrainingDiagnosticsCallback(BaseCallback):
    """
    PPO için XAI uyumlu training callback.

    PPO on-policy çalıştığı için replay buffer yoktur.
    Bu callback:
      - train_rollout_step_logs.csv için step bilgilerini RAM'de tutar.
      - train_episode_summary.csv için episode özetleri üretir.
      - training_update_logs.csv için PPO update metriklerini SB3 logger'dan yazar.
    """

    def __init__(
        self,
        log_path: Path,
        cfg=None,
        log_every: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.cfg = cfg
        self.log_every = int(log_every)
        self.update_id = 0

        self.columns = [
            "update_id",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "learning_rate",
            "gradient_norm",
            "mean_episode_reward",
            "mean_episode_length",
        ]

        self.train_steps = []
        self.episode_rows = []

        # Eski/ortak isimlerle uyumluluk
        self.ep_rows = self.episode_rows
        self.episodes = self.episode_rows
        self.episode_summaries = self.episode_rows
        self.train_episodes = self.episode_rows

        self._episode_id = 0
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_penalty = 0.0
        self._episode_violation_count = 0
        self._episode_best_cl_cd = -float("inf")
        self._episode_initial_cl_cd = None
        self._episode_action_norms = []

        self.last_info = {}
        self.last_step_row = {}

        self._last_logged_num_timesteps = -1

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    @staticmethod
    def _first(value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else default
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value[0]
        return value

    @staticmethod
    def _float(value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _bool(value) -> bool:
        try:
            if isinstance(value, np.ndarray):
                return bool(value.item()) if value.shape == () else bool(value[0])
            return bool(value)
        except Exception:
            return False

    def _current_lr(self) -> float:
        try:
            return float(self.model.policy.optimizer.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def _grad_norm(self, module: torch.nn.Module) -> float:
        total_sq = 0.0
        has_grad = False

        for param in module.parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = float(param.grad.detach().data.norm(2).cpu())
                total_sq += grad_norm ** 2

        if not has_grad:
            return float("nan")

        return float(total_sq ** 0.5)

    def _logger_value(self, key: str, default=np.nan):
        try:
            return self.model.logger.name_to_value.get(key, default)
        except Exception:
            return default

    def _write_training_update_row(self) -> None:
        """
        PPO train() bittikten sonra SB3 logger'da kalan son metrikleri yazar.
        _on_rollout_start içinde çağrılır; çünkü önceki rollout'un ardından train() tamamlanmış olur.
        """
        if self.num_timesteps == self._last_logged_num_timesteps:
            return

        policy_loss = self._logger_value("train/policy_gradient_loss", np.nan)
        value_loss = self._logger_value("train/value_loss", np.nan)
        entropy_loss = self._logger_value("train/entropy_loss", np.nan)

        approx_kl = self._logger_value("train/approx_kl", np.nan)
        clip_fraction = self._logger_value("train/clip_fraction", np.nan)
        explained_variance = self._logger_value("train/explained_variance", np.nan)

        # İlk rollout başında bu değerler henüz yoktur; boş satır yazmayalım.
        if pd.isna(policy_loss) and pd.isna(value_loss) and pd.isna(entropy_loss):
            return

        self.update_id += 1
        self._last_logged_num_timesteps = self.num_timesteps

        if self.episode_rows:
            recent_eps = self.episode_rows[-10:]
            mean_episode_reward = float(
                np.mean([row.get("total_reward", 0.0) for row in recent_eps])
            )
            mean_episode_length = float(
                np.mean([row.get("episode_length", 0.0) for row in recent_eps])
            )
        else:
            mean_episode_reward = float("nan")
            mean_episode_length = float("nan")

        row = {
            "update_id": self.update_id,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": explained_variance,
            "learning_rate": self._current_lr(),
            "gradient_norm": self._grad_norm(self.model.policy),
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
        }

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)

    def _on_rollout_start(self) -> None:
        # Önceki PPO update metriklerini yaz.
        self._write_training_update_row()

    def _log_train_rollout_step(self) -> None:
        infos = self.locals.get("infos", [{}])
        info = self._first(infos, default={}) or {}

        actions = self.locals.get("actions", np.zeros((1, 8), dtype=np.float32))
        action = np.asarray(self._first(actions, np.zeros(8)), dtype=np.float32).reshape(-1)

        dones = self.locals.get("dones", [False])
        done = self._bool(self._first(dones, False))

        rewards = self.locals.get("rewards", [0.0])
        reward = self._float(self._first(rewards, 0.0))

        if not info:
            return

        self.last_info = info

        if self._episode_initial_cl_cd is None:
            self._episode_initial_cl_cd = self._float(info.get("CL_CD", 0.0))

        self._episode_reward += reward
        self._episode_penalty += self._float(info.get("penalty_total", 0.0))

        cm_violation = int(
            self._float(info.get("CM_lower_violation", 0.0)) > 0.0
            or self._float(info.get("CM_upper_violation", 0.0)) > 0.0
        )
        tc_violation = int(
            self._float(info.get("tc_lower_violation", 0.0)) > 0.0
            or self._float(info.get("tc_upper_violation", 0.0)) > 0.0
        )

        self._episode_violation_count += int(cm_violation + tc_violation)

        cl_cd = self._float(info.get("CL_CD", 0.0))
        self._episode_best_cl_cd = max(self._episode_best_cl_cd, cl_cd)

        action_norm = self._float(info.get("action_norm", np.linalg.norm(action)))
        self._episode_action_norms.append(action_norm)
        row_done_reason = _standardize_done_reason(info, done=done)
        row = {
            "experiment_id": "",
            "algorithm": "PPO",
            "evaluator": self.cfg.evaluator if self.cfg is not None else "",
            "seed": self.cfg.seed if self.cfg is not None else "",
            "episode_id": self._episode_id,
            "step_id": self._episode_step,
            "global_step": self.num_timesteps,

            "AoA": self.cfg.aoa if self.cfg is not None else "",
            "Re": self.cfg.re if self.cfg is not None else "",
            "log10_Re": np.log10(self.cfg.re) if self.cfg is not None else "",

            "CL": info.get("CL", ""),
            "CD": info.get("CD", ""),
            "CM": info.get("CM", ""),
            "CL_CD": info.get("CL_CD", ""),
            "t_c": info.get("t_c", ""),

            "CL_pred": info.get("CL_pred", info.get("CL", "")),
            "CD_pred": info.get("CD_pred", info.get("CD", "")),
            "CM_pred": info.get("CM_pred", info.get("CM", "")),
            "CL_CD_pred": info.get("CL_CD_pred", info.get("CL_CD", "")),

            "is_CM_feasible": info.get("is_CM_feasible", ""),
            "is_tc_feasible": info.get("is_tc_feasible", ""),
            "is_geometry_valid": info.get("is_geometry_valid", ""),

            "action_norm": info.get("action_norm", action_norm),
            "upper_action_norm": info.get("upper_action_norm", ""),
            "lower_action_norm": info.get("lower_action_norm", ""),
            "action_max_abs": info.get("action_max_abs", ""),
            "action_saturation_count": info.get("action_saturation_count", ""),

            "reward_total": info.get("reward_total", reward),
            "reward_objective_term": info.get("reward_objective_term", ""),
            "reward_CL_CD_term": info.get("reward_CL_CD_term", ""),
            "reward_CM_penalty": info.get("reward_CM_penalty", 0.0),
            "reward_tc_penalty": info.get("reward_tc_penalty", 0.0),
            "reward_local_thickness_penalty": info.get("reward_local_thickness_penalty", 0.0),
            "reward_invalid_geometry_penalty": info.get("reward_invalid_geometry_penalty", 0.0),
            "reward_solver_error_penalty": info.get("reward_solver_error_penalty", 0.0),
            "reward_action_penalty": info.get("reward_action_penalty", 0.0),
            "penalty_total": info.get("penalty_total", 0.0),

            "CM_lower_violation": info.get("CM_lower_violation", 0.0),
            "CM_upper_violation": info.get("CM_upper_violation", 0.0),
            "tc_lower_violation": info.get("tc_lower_violation", 0.0),
            "tc_upper_violation": info.get("tc_upper_violation", 0.0),
            "local_thickness_violation": info.get("local_thickness_violation", 0.0),

            "max_thickness": info.get("max_thickness", ""),
            "x_max_thickness": info.get("x_max_thickness", ""),
            "max_camber": info.get("max_camber", ""),
            "x_max_camber": info.get("x_max_camber", ""),
            "leading_edge_radius_proxy": info.get("leading_edge_radius_proxy", ""),
            "trailing_edge_thickness": info.get("trailing_edge_thickness", ""),
            "upper_surface_curvature_mean": info.get("upper_surface_curvature_mean", ""),
            "lower_surface_curvature_mean": info.get("lower_surface_curvature_mean", ""),
            "surface_smoothness": info.get("surface_smoothness", ""),
            "min_local_thickness": info.get("min_local_thickness", ""),
            "area_proxy": info.get("area_proxy", ""),

            "done": done,
            "done_reason": row_done_reason,
            "solver_status": info.get("solver_status", ""),
            "solver_error_message": info.get("solver_error_message", ""),

            # XFOIL training diagnostics
            "xfoil_converged": info.get("xfoil_converged", ""),
            "xfoil_error_message": info.get("xfoil_error_message", ""),
            "xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),
        }

        prev_cst = np.asarray(info.get("prev_cst", np.zeros(8)), dtype=np.float32).reshape(-1)
        next_cst = np.asarray(info.get("next_cst", np.zeros(8)), dtype=np.float32).reshape(-1)
        delta_cst = np.asarray(info.get("delta_cst", np.zeros(8)), dtype=np.float32).reshape(-1)

        for i in range(4):
            row[f"state_CST_u{i+1}"] = float(prev_cst[i])
            row[f"state_CST_l{i+1}"] = float(prev_cst[4+i])

            row[f"action_u{i+1}"] = float(action[i])
            row[f"action_l{i+1}"] = float(action[4+i])

            row[f"next_CST_u{i+1}"] = float(next_cst[i])
            row[f"next_CST_l{i+1}"] = float(next_cst[4+i])

            row[f"delta_CST_u{i+1}"] = float(delta_cst[i])
            row[f"delta_CST_l{i+1}"] = float(delta_cst[4+i])

        self.train_steps.append(row)
        self.last_step_row = row

        self._episode_step += 1

        if done:
            final_is_cm_feasible = _as_bool(info.get("is_CM_feasible", False))
            final_is_tc_feasible = _as_bool(info.get("is_tc_feasible", False))
            final_is_geometry_valid = _as_bool(info.get("is_geometry_valid", False))

            if self.cfg is not None:
                final_is_feasible = _is_feasible_info(info)
                final_is_strict_safe = _is_strict_safe_info(info, self.cfg)
            else:
                final_is_feasible = (
                    _is_constraint_feasible_info(info)
                    and _is_solver_success_info(info)
                )
                final_is_strict_safe = final_is_feasible

            done_reason = _standardize_done_reason(info, done=done)

            self.episode_rows.append(
                {
                    "episode_id": self._episode_id,
                    "initial_CL_CD": self._episode_initial_cl_cd,
                    "final_CL_CD": info.get("CL_CD", np.nan),
                    "best_CL_CD": self._episode_best_cl_cd,
                    "final_CL": info.get("CL", np.nan),
                    "final_CD": info.get("CD", np.nan),
                    "final_CM": info.get("CM", np.nan),
                    "final_t_c": info.get("t_c", np.nan),
                    "total_reward": self._episode_reward,
                    "total_penalty": self._episode_penalty,
                    "constraint_violation_count": self._episode_violation_count,
                    "done_reason": done_reason,
                    "episode_length": self._episode_step,
                    "is_final_feasible": final_is_feasible,
                    "is_final_strict_safe": final_is_strict_safe,

                    # Local geometry safety
                    "final_min_local_thickness": info.get("min_local_thickness", np.nan),
                    "final_local_thickness_violation": info.get(
                        "local_thickness_violation",
                        np.nan,
                    ),

                    # XFOIL final solver status
                    "final_xfoil_converged": info.get("xfoil_converged", ""),
                    "final_xfoil_error_message": info.get("xfoil_error_message", ""),
                    "final_xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),
                }
            )

            self._episode_id += 1
            self._episode_step = 0
            self._episode_reward = 0.0
            self._episode_penalty = 0.0
            self._episode_violation_count = 0
            self._episode_best_cl_cd = -float("inf")
            self._episode_initial_cl_cd = None
            self._episode_action_norms = []

    def _on_step(self) -> bool:
        self._log_train_rollout_step()
        return True

    def _on_training_end(self) -> None:
        # Son PPO update metriklerini kaçırmamak için training sonunda tekrar yaz.
        self._write_training_update_row()


def _sac_alpha_value(model) -> float:
    try:
        if hasattr(model, "log_ent_coef") and model.log_ent_coef is not None:
            return float(torch.exp(model.log_ent_coef.detach()).cpu())

        if hasattr(model, "ent_coef_tensor"):
            return float(model.ent_coef_tensor.detach().cpu())
    except Exception:
        pass

    return float("nan")


def _sac_policy_diagnostics(
    model,
    obs_np: np.ndarray,
    action_np: np.ndarray,
    next_obs_np: np.ndarray,
    reward: float,
    done: bool,
) -> dict:
    device = model.device

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).reshape(1, -1)
    action_t = torch.as_tensor(action_np, dtype=torch.float32, device=device).reshape(1, -1)
    next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(1, -1)

    with torch.no_grad():
        mean_actions, log_std, _ = model.actor.get_action_dist_params(obs_t)
        actor_std = torch.exp(log_std)

        sampled_actions, log_prob = model.actor.action_log_prob(obs_t)
        deterministic_actions = model.actor(obs_t, deterministic=True)

        q1, q2 = model.critic(obs_t, action_t)
        q_min = torch.min(q1, q2)
        q_mean = 0.5 * (q1 + q2)

        q_disagreement_abs = torch.abs(q1 - q2)
        q_disagreement_relative = q_disagreement_abs / (torch.abs(q_mean) + 1e-8)

        alpha_value = _sac_alpha_value(model)
        alpha_t = torch.tensor(alpha_value, dtype=torch.float32, device=device)

        next_actions, next_log_prob = model.actor.action_log_prob(next_obs_t)
        next_log_prob = next_log_prob.reshape(1, 1)

        next_q1, next_q2 = model.critic_target(next_obs_t, next_actions)
        next_q = torch.min(next_q1, next_q2)

        reward_t = torch.tensor([[float(reward)]], dtype=torch.float32, device=device)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32, device=device)

        target_q = reward_t + (1.0 - done_t) * model.gamma * (
            next_q - alpha_t * next_log_prob
        )

        td_error_q1 = q1 - target_q
        td_error_q2 = q2 - target_q

    mean_np = mean_actions.detach().cpu().numpy().reshape(-1)
    log_std_np = log_std.detach().cpu().numpy().reshape(-1)
    std_np = actor_std.detach().cpu().numpy().reshape(-1)
    sampled_np = sampled_actions.detach().cpu().numpy().reshape(-1)
    deterministic_np = deterministic_actions.detach().cpu().numpy().reshape(-1)

    out = {}

    for i, name in enumerate(["u1", "u2", "u3", "u4", "l1", "l2", "l3", "l4"]):
        out[f"actor_mean_{name}"] = float(mean_np[i])
        out[f"actor_log_std_{name}"] = float(log_std_np[i])
        out[f"actor_std_{name}"] = float(std_np[i])
        out[f"sampled_action_{name}"] = float(sampled_np[i])
        out[f"deterministic_action_{name}"] = float(deterministic_np[i])

    out.update(
        {
            "log_prob_action": float(log_prob.reshape(-1)[0].detach().cpu()),
            "policy_entropy": float((-log_prob).reshape(-1)[0].detach().cpu()),
            "Q1_s_a": float(q1.reshape(-1)[0].detach().cpu()),
            "Q2_s_a": float(q2.reshape(-1)[0].detach().cpu()),
            "Q_min_s_a": float(q_min.reshape(-1)[0].detach().cpu()),
            "Q_mean_s_a": float(q_mean.reshape(-1)[0].detach().cpu()),
            "Q_disagreement_abs": float(q_disagreement_abs.reshape(-1)[0].detach().cpu()),
            "Q_disagreement_relative": float(q_disagreement_relative.reshape(-1)[0].detach().cpu()),
            "target_Q": float(target_q.reshape(-1)[0].detach().cpu()),
            "soft_value_target": float(
                (next_q - alpha_t * next_log_prob).reshape(-1)[0].detach().cpu()
            ),
            "td_error_Q1": float(td_error_q1.reshape(-1)[0].detach().cpu()),
            "td_error_Q2": float(td_error_q2.reshape(-1)[0].detach().cpu()),
            "alpha": float(alpha_value),
            "target_entropy": float(getattr(model, "target_entropy", np.nan)),
        }
    )

    return out


# def _safe_float_value(value, default=np.nan) -> float:
#     try:
#         if value is None:
#             return float(default)
#         if isinstance(value, str) and value.strip() == "":
#             return float(default)
#         return float(value)
#     except Exception:
#         return float(default)


# def _is_strict_feasible_info(info: dict, cfg: ExperimentConfig, eps: float = 1e-8) -> bool:
#     is_cm_feasible = _as_bool(info.get("is_CM_feasible", False))
#     is_tc_feasible = _as_bool(info.get("is_tc_feasible", False))
#     is_geometry_valid = _as_bool(info.get("is_geometry_valid", False))

#     local_violation = _safe_float_value(
#         info.get("local_thickness_violation", 0.0),
#         default=0.0,
#     )

#     min_local_thickness = _safe_float_value(
#         info.get("min_local_thickness", np.nan),
#         default=np.nan,
#     )

#     local_ok = local_violation <= eps

#     if np.isfinite(min_local_thickness):
#         local_ok = local_ok and (
#             min_local_thickness
#             >= float(cfg.geometry.min_local_thickness_required) - eps
#         )

#     return bool(
#         is_cm_feasible
#         and is_tc_feasible
#         and is_geometry_valid
#         and local_ok
#     )


def _ppo_policy_diagnostics(
    model,
    obs_np: np.ndarray,
    action_np: np.ndarray,
) -> dict:
    """
    PPO evaluation için policy distribution, log_prob, entropy ve V(s) çıkarır.
    """
    device = model.device

    obs_t = torch.as_tensor(
        obs_np,
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)

    action_t = torch.as_tensor(
        action_np,
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)

    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)

        values, log_prob, entropy = model.policy.evaluate_actions(
            obs_t,
            action_t,
        )

        sampled_action_t = dist.sample()

        # DiagGaussianDistribution için mean/std bilgisi
        try:
            mean_t = dist.distribution.mean
            std_t = dist.distribution.stddev
        except Exception:
            mean_t = torch.full_like(action_t, float("nan"))
            std_t = torch.full_like(action_t, float("nan"))

    mean_np = mean_t.detach().cpu().numpy().reshape(-1)
    std_np = std_t.detach().cpu().numpy().reshape(-1)
    sampled_np = sampled_action_t.detach().cpu().numpy().reshape(-1)
    deterministic_np = np.asarray(action_np, dtype=np.float32).reshape(-1)

    out = {}

    for i, name in enumerate(["u1", "u2", "u3", "u4", "l1", "l2", "l3", "l4"]):
        out[f"policy_mean_{name}"] = float(mean_np[i])
        out[f"policy_std_{name}"] = float(std_np[i])
        out[f"sampled_action_{name}"] = float(sampled_np[i])
        out[f"deterministic_action_{name}"] = float(deterministic_np[i])

    out.update(
        {
            "log_prob_action": float(log_prob.reshape(-1)[0].detach().cpu()),
            "entropy": float(entropy.reshape(-1)[0].detach().cpu())
            if entropy is not None
            else float("nan"),
            "value_V_s": float(values.reshape(-1)[0].detach().cpu()),
        }
    )

    return out


def _make_evaluator(cfg: ExperimentConfig):
    evaluator_name = cfg.evaluator.lower()

    if evaluator_name == "surrogate":
        return SurrogateEvaluator(
            checkpoint_path=cfg.surrogate_checkpoint_path,
            model_name=cfg.surrogate_model_name,
            scaler_json_path=cfg.scaler_json_path,
        )

    if evaluator_name == "xfoil":
        return XFOILEvaluator(
            executable_path=cfg.xfoil.executable_path,
            timeout_sec=cfg.xfoil.timeout_sec,
            max_iter=cfg.xfoil.max_iter,
            ppar_n=cfg.xfoil.ppar_n,
            mach=cfg.xfoil.mach,
            ncrit=cfg.xfoil.ncrit,
            xtr_top=cfg.xfoil.xtr_top,
            xtr_bottom=cfg.xfoil.xtr_bottom,
            n_points=cfg.xfoil.n_points,
            min_local_thickness_required=cfg.geometry.min_local_thickness_required,
            max_abs_surface_y=cfg.geometry.max_abs_surface_y,
        )

    raise ValueError(f"Unsupported evaluator: {cfg.evaluator}")


def _write_replay_sample(model: TD3, run_dir: Path, sample_n: int = 5000):
    rb = model.replay_buffer
    if rb is None or rb.size() == 0:
        pd.DataFrame(columns=["buffer_index","state_t","action_t","reward_t","next_state_t","done_t","sampling_weight"]).to_csv(run_dir / "replay_sample_logs.csv", index=False)
        return
    n = min(sample_n, rb.size())
    idxs = np.linspace(0, rb.size() - 1, n, dtype=int)
    rows = []
    for i in idxs:
        rows.append({
            "buffer_index": int(i),
            "state_t": np.asarray(rb.observations[i]).reshape(-1).tolist(),
            "action_t": np.asarray(rb.actions[i]).reshape(-1).tolist(),
            "reward_t": float(np.asarray(rb.rewards[i]).reshape(-1)[0]),
            "next_state_t": np.asarray(rb.next_observations[i]).reshape(-1).tolist(),
            "done_t": float(np.asarray(rb.dones[i]).reshape(-1)[0]),
            "sampling_weight": 1.0,
        })
    pd.DataFrame(rows).to_csv(run_dir / "replay_sample_logs.csv", index=False)


def _resolve_rl_checkpoint_path(cfg, run_dir: Path | None) -> Path:
    """
    RL checkpoint'i çözer.

    Öncelik sırası:
    1. Kullanıcı açıkça --rl-checkpoint-path verdiyse onu kullanır.
    2. --run-dir verilmişse o klasör içinde .zip model arar.
    3. Tercih edilen model adı td3_surrogate_s-1d.zip ise onu seçer.
    """

    # 1. Kullanıcı açıkça checkpoint path verdiyse onu kullan.
    explicit_path = str(getattr(cfg, "rl_checkpoint_path", "") or "").strip()

    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {path}")
        return path

    # 2. Run klasörü içinde model ara.
    if run_dir is not None:
        run_dir = Path(run_dir)

        preferred_path = run_dir / _rl_model_filename(cfg)
        if preferred_path.exists():
            return preferred_path

        candidates = sorted(run_dir.glob("*.zip"))

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple .zip model files found in {run_dir}. "
                f"Please specify --rl-checkpoint-path explicitly. "
                f"Candidates: {[str(c) for c in candidates]}"
            )

    raise FileNotFoundError(
        "RL checkpoint path could not be resolved. "
        "Provide --run-dir pointing to a training run folder or specify --rl-checkpoint-path."
    )


def _resolve_train_run_dir_from_checkpoint(rl_checkpoint_path: str, logs_root: Path = Path("logs")) -> Optional[Path]:
    ckpt = Path(rl_checkpoint_path).resolve()
    idx_path = ckpt.parent / "checkpoint_index.json"
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        run = data.get(str(ckpt))
        if run:
            p = Path(run)
            if p.exists():
                return p
    for meta in logs_root.glob("**/experiment_metadata.json"):
        with open(meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        if Path(m.get("rl_checkpoint_path", "")).resolve() == ckpt:
            return meta.parent
    return None


def _create_eval_run_dir(train_run_dir: Path) -> Path:
    eval_root = train_run_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dir = eval_root / f"run_{int(time.time())}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def train_td3(
    cfg: ExperimentConfig,
    base_logs: Path = Path("logs"),
) -> Path:
    run_dir = create_run_dir(base_logs, "td3")

    env = AirfoilEnv(cfg, _make_evaluator(cfg))

    td3_cfg = cfg.td3

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=td3_cfg.action_noise_sigma * np.ones(n_actions),
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=td3_cfg.learning_rate,
        buffer_size=td3_cfg.buffer_size,
        learning_starts=td3_cfg.learning_starts,
        batch_size=td3_cfg.batch_size,
        tau=td3_cfg.tau,
        gamma=td3_cfg.gamma,
        train_freq=td3_cfg.train_freq,
        gradient_steps=td3_cfg.gradient_steps,
        policy_delay=td3_cfg.policy_delay,
        target_policy_noise=td3_cfg.target_policy_noise,
        target_noise_clip=td3_cfg.target_noise_clip,
        action_noise=action_noise,
        seed=cfg.seed,
        verbose=1,
    )

    cb = TD3TrainingDiagnosticsCallback(
        log_path=run_dir / "training_update_logs.csv",
        cfg=cfg,
        log_every=100,
        batch_size=min(cfg.td3.batch_size, 256),
        verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=cfg.total_timesteps,
    callback=cb)
    train_wall_time = time.time() - t0

    rl_checkpoint_path = run_dir / _rl_model_filename(cfg)
    model.save(rl_checkpoint_path)

    # Metadata ve evaluation için config içine gerçek model path'ini yaz.
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    write_experiment_metadata(cfg, run_dir, normalization_stats={"source": cfg.scaler_json_path, "train_wall_time_sec": train_wall_time})

    pd.DataFrame(cb.train_steps).to_csv(run_dir / "train_rollout_step_logs.csv", index=False)
    pd.DataFrame(cb.ep_rows).to_csv(run_dir / "train_episode_summary.csv", index=False)
    _write_replay_sample(model, run_dir)
    last_info = getattr(cb, "last_info", {}) or {}
    last_step_row = getattr(cb, "last_step_row", {}) or {}

    def _pick_metric(key: str, default=np.nan):
        if isinstance(last_info, dict) and key in last_info:
            return last_info.get(key, default)
        if isinstance(last_step_row, dict) and key in last_step_row:
            return last_step_row.get(key, default)
        return default
    
    # train_metrics için kullanılacak son step bilgisini standartlaştır.
    # Öncelik last_step_row'da; boşsa last_info kullanılır.
    if isinstance(last_step_row, dict) and len(last_step_row) > 0:
        last_step = last_step_row
    elif isinstance(last_info, dict) and len(last_info) > 0:
        last_step = last_info
    else:
        last_step = {}

    # def _as_bool(value) -> bool:
    #     if isinstance(value, (bool, np.bool_)):
    #         return bool(value)

    #     if value is None:
    #         return False

    #     try:
    #         if pd.isna(value):
    #             return False
    #     except Exception:
    #         pass

    #     return str(value).strip().lower() in {"true", "1", "yes"}


    # def _safe_get(row: dict, key: str, default=np.nan):
    #     if not isinstance(row, dict):
    #         return default
    #     return row.get(key, default)


    # def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    #     if col not in df.columns:
    #         return pd.Series(dtype=float)
    #     return pd.to_numeric(df[col], errors="coerce")


    # def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    #     if col not in df.columns:
    #         return pd.Series(False, index=df.index)
    #     return df[col].map(_as_bool)
    
    train_steps_df = pd.DataFrame(cb.train_steps)

    # Training boyunca bulunan en iyi tasarımları iki ayrı kritere göre seçiyoruz:
    # 1) best_train_feasible_row      -> paper feasibility: CM + t/c + geometry valid
    # 2) best_train_strict_safe_row   -> paper feasibility + local thickness safety margin
    if len(train_steps_df) > 0 and "CL_CD" in train_steps_df.columns:
        train_steps_df = train_steps_df.copy()

        feasible_df = train_steps_df[
            _feasible_mask(train_steps_df)
        ].copy()

        strict_safe_df = train_steps_df[
            _strict_safe_mask(train_steps_df, cfg)
        ].copy()

        best_train_feasible_row = _best_row_by_cl_cd(feasible_df)
        best_train_strict_safe_row = _best_row_by_cl_cd(strict_safe_df)
    else:
        best_train_feasible_row = None
        best_train_strict_safe_row = None

    final_is_feasible = _is_feasible_info(last_step)
    final_is_strict_safe = _is_strict_safe_info(last_step, cfg)

    train_metrics = {
        "algorithm": "TD3",
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "train_wall_time_sec": train_wall_time,

        # Training son step / son state
        "final_CL": _safe_get(last_step, "CL"),
        "final_CD": _safe_get(last_step, "CD"),
        "final_CL_CD": _safe_get(last_step, "CL_CD"),
        "final_CM": _safe_get(last_step, "CM"),
        "final_t_c": _safe_get(last_step, "t_c"),
        "final_done_reason": _safe_get(last_step, "done_reason", ""),
        "final_solver_status": _safe_get(last_step, "solver_status", ""),
        "final_solver_error_message": _safe_get(last_step, "solver_error_message", ""),
        "final_xfoil_converged": _safe_get(last_step, "xfoil_converged", ""),
        "final_xfoil_error_message": _safe_get(last_step, "xfoil_error_message", ""),
        "final_xfoil_runtime_ms": _safe_get(last_step, "xfoil_runtime_ms", np.nan),
        "final_is_CM_feasible": _safe_get(last_step, "is_CM_feasible", ""),
        "final_is_tc_feasible": _safe_get(last_step, "is_tc_feasible", ""),
        "final_is_geometry_valid": _safe_get(last_step, "is_geometry_valid", ""),
        "final_is_feasible": final_is_feasible,
        "final_is_strict_safe": final_is_strict_safe,
        "final_min_local_thickness": _safe_get(last_step, "min_local_thickness", np.nan),
        "final_local_thickness_violation": _safe_get(last_step, "local_thickness_violation", np.nan),
        "final_is_local_thickness_feasible": _is_local_margin_safe_info(last_step, cfg),

        # Training boyunca bulunan en iyi feasible step
        "best_train_feasible_CL": (
            _safe_get(best_train_feasible_row, "CL")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CD": (
            _safe_get(best_train_feasible_row, "CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CL_CD": (
            _safe_get(best_train_feasible_row, "CL_CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CM": (
            _safe_get(best_train_feasible_row, "CM")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_t_c": (
            _safe_get(best_train_feasible_row, "t_c")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_episode_id": (
            int(_safe_get(best_train_feasible_row, "episode_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),
        "best_train_feasible_step_id": (
            int(_safe_get(best_train_feasible_row, "step_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),
        "has_train_feasible_design": best_train_feasible_row is not None,
        "best_train_strict_safe_CL": (
            _safe_get(best_train_strict_safe_row, "CL")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CD": (
            _safe_get(best_train_strict_safe_row, "CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CL_CD": (
            _safe_get(best_train_strict_safe_row, "CL_CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CM": (
            _safe_get(best_train_strict_safe_row, "CM")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_t_c": (
            _safe_get(best_train_strict_safe_row, "t_c")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_episode_id": (
            int(_safe_get(best_train_strict_safe_row, "episode_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "best_train_strict_safe_step_id": (
            int(_safe_get(best_train_strict_safe_row, "step_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "has_train_strict_safe_design": best_train_strict_safe_row is not None,

        # Log kontrol bilgileri
        "logged_train_steps": len(getattr(cb, "train_steps", [])),
        "logged_train_episodes": len(getattr(cb, "ep_rows", [])),
        "logged_training_diagnostics": getattr(cb, "update_id", 0),

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,
    }
    pd.DataFrame([train_metrics]).to_csv(run_dir / "train_metrics.csv", index=False)
    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "xai_v1",
                "run_type": "train",
                "files": [
                    "experiment_metadata.json",
                    "training_update_logs.csv",
                    "train_rollout_step_logs.csv",
                    "train_episode_summary.csv",
                    "replay_sample_logs.csv",
                    "train_metrics.csv",
                    rl_checkpoint_path.name, 
                ],
            },
            f,
            indent=2,
        )

    # evaluate ayrı komutla çalıştırılacak; train sadece eğitim artefact'larını üretir.
    return run_dir

def train_sac(
    cfg: ExperimentConfig,
    base_logs: Path = Path("logs"),
) -> Path:
    cfg.algorithm = "sac"

    run_dir = create_run_dir(base_logs, "sac")
    env = AirfoilEnv(cfg, _make_evaluator(cfg))

    sac_cfg = cfg.sac

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_cfg.learning_rate,
        buffer_size=sac_cfg.buffer_size,
        learning_starts=sac_cfg.learning_starts,
        batch_size=sac_cfg.batch_size,
        tau=sac_cfg.tau,
        gamma=sac_cfg.gamma,
        train_freq=sac_cfg.train_freq,
        gradient_steps=sac_cfg.gradient_steps,
        ent_coef=sac_cfg.ent_coef,
        target_entropy=sac_cfg.target_entropy,
        verbose=1,
        seed=cfg.seed,
    )

    callback = SACTrainingDiagnosticsCallback(
        log_path=run_dir / "training_update_logs.csv",
        cfg=cfg,
        log_every=100,
        batch_size=min(sac_cfg.batch_size, 256),
        verbose=1,
    )

    t0 = time.time()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        log_interval=4,
    )

    train_wall_time = time.time() - t0

    rl_checkpoint_path = run_dir / _rl_model_filename(cfg)
    model.save(rl_checkpoint_path)
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    write_experiment_metadata(
        cfg=cfg,
        run_dir=run_dir,
        normalization_stats={
            "observation": "8 CST + AoA + Re_norm + log10_Re + CL + CD + CM + CL_CD + t_c",
            "surrogate_input": "8 CST + AoA + Re/log10(Re), depending on TorchScript wrapper",
            "action_scale": cfg.action_scale,
        },
    )

    train_steps_df = pd.DataFrame(callback.train_steps)
    train_steps_df.to_csv(run_dir / "train_rollout_step_logs.csv", index=False)

    pd.DataFrame(callback.episode_rows).to_csv(
        run_dir / "train_episode_summary.csv",
        index=False,
    )

    _write_replay_sample(model, run_dir)

    last_step = callback.last_step_row or {}

    final_is_feasible = _is_feasible_info(last_step)
    final_is_strict_safe = _is_strict_safe_info(last_step, cfg)

    # Training boyunca bulunan en iyi tasarımları iki ayrı kritere göre seçiyoruz:
    # 1) best_train_feasible_row      -> paper feasibility: CM + t/c + geometry valid
    # 2) best_train_strict_safe_row   -> paper feasibility + local thickness safety margin
    if len(train_steps_df) > 0 and "CL_CD" in train_steps_df.columns:
        train_steps_df = train_steps_df.copy()

        feasible_df = train_steps_df[
            _feasible_mask(train_steps_df)
        ].copy()

        strict_safe_df = train_steps_df[
            _strict_safe_mask(train_steps_df, cfg)
        ].copy()

        best_train_feasible_row = _best_row_by_cl_cd(feasible_df)
        best_train_strict_safe_row = _best_row_by_cl_cd(strict_safe_df)
    else:
        best_train_feasible_row = None
        best_train_strict_safe_row = None

    train_metrics = {
        "algorithm": "SAC",
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "train_wall_time_sec": train_wall_time,

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,

        "final_CL": _safe_get(last_step, "CL"),
        "final_CD": _safe_get(last_step, "CD"),
        "final_CL_CD": _safe_get(last_step, "CL_CD"),
        "final_CM": _safe_get(last_step, "CM"),
        "final_t_c": _safe_get(last_step, "t_c"),
        "final_done_reason": _safe_get(last_step, "done_reason", ""),
        "final_solver_status": _safe_get(last_step, "solver_status", ""),
        "final_solver_error_message": _safe_get(last_step, "solver_error_message", ""),
        "final_xfoil_converged": _safe_get(last_step, "xfoil_converged", ""),
        "final_xfoil_error_message": _safe_get(last_step, "xfoil_error_message", ""),
        "final_xfoil_runtime_ms": _safe_get(last_step, "xfoil_runtime_ms", np.nan),
        "final_is_CM_feasible": _safe_get(last_step, "is_CM_feasible", ""),
        "final_is_tc_feasible": _safe_get(last_step, "is_tc_feasible", ""),
        "final_is_geometry_valid": _safe_get(last_step, "is_geometry_valid", ""),
        "final_is_feasible": final_is_feasible,
        "final_is_strict_safe": final_is_strict_safe,
        "final_min_local_thickness": _safe_get(last_step, "min_local_thickness", np.nan),
        "final_local_thickness_violation": _safe_get(last_step, "local_thickness_violation", np.nan),
        "final_is_local_thickness_feasible": _is_local_margin_safe_info(last_step, cfg),

        "best_train_feasible_CL": (
            _safe_get(best_train_feasible_row, "CL")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CD": (
            _safe_get(best_train_feasible_row, "CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CL_CD": (
            _safe_get(best_train_feasible_row, "CL_CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CM": (
            _safe_get(best_train_feasible_row, "CM")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_t_c": (
            _safe_get(best_train_feasible_row, "t_c")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_episode_id": (
            int(_safe_get(best_train_feasible_row, "episode_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),
        "best_train_feasible_step_id": (
            int(_safe_get(best_train_feasible_row, "step_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),
        "has_train_feasible_design": best_train_feasible_row is not None,
        "best_train_strict_safe_CL": (
            _safe_get(best_train_strict_safe_row, "CL")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CD": (
            _safe_get(best_train_strict_safe_row, "CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CL_CD": (
            _safe_get(best_train_strict_safe_row, "CL_CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CM": (
            _safe_get(best_train_strict_safe_row, "CM")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_t_c": (
            _safe_get(best_train_strict_safe_row, "t_c")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_episode_id": (
            int(_safe_get(best_train_strict_safe_row, "episode_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "best_train_strict_safe_step_id": (
            int(_safe_get(best_train_strict_safe_row, "step_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "has_train_strict_safe_design": best_train_strict_safe_row is not None,

        "logged_train_steps": len(callback.train_steps),
        "logged_train_episodes": len(callback.episode_rows),
        "logged_training_diagnostics": callback.update_id,
    }

    pd.DataFrame([train_metrics]).to_csv(
        run_dir / "train_metrics.csv",
        index=False,
    )

    manifest = {
        "algorithm": "SAC",
        "evaluator": cfg.evaluator,
        "run_dir": str(run_dir),
        "rl_checkpoint_file": rl_checkpoint_path.name,
        "files": [
            "experiment_metadata.json",
            "training_update_logs.csv",
            "train_rollout_step_logs.csv",
            "train_episode_summary.csv",
            "replay_sample_logs.csv",
            "train_metrics.csv",
            rl_checkpoint_path.name,
        ],
    }

    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return run_dir


def train_ppo(
    cfg: ExperimentConfig,
    base_logs: Path = Path("logs"),
) -> Path:
    cfg.algorithm = "ppo"

    run_dir = create_run_dir(base_logs, "ppo")

    env = AirfoilEnv(cfg, _make_evaluator(cfg))

    ppo_cfg = cfg.ppo

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=_linear_schedule(
            ppo_cfg.learning_rate_start,
            ppo_cfg.learning_rate_end,
        ),
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        normalize_advantage=ppo_cfg.normalize_advantage,
        verbose=1,
        seed=cfg.seed,
    )

    callback = PPOTrainingDiagnosticsCallback(
        log_path=run_dir / "training_update_logs.csv",
        cfg=cfg,
        verbose=1,
    )

    t0 = time.time()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        log_interval=4,
    )

    train_wall_time = time.time() - t0

    rl_checkpoint_path = run_dir / _rl_model_filename(cfg)
    model.save(rl_checkpoint_path)
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    write_experiment_metadata(
        cfg=cfg,
        run_dir=run_dir,
        normalization_stats={
            "observation": "8 CST + AoA + Re_norm + log10_Re + CL + CD + CM + CL_CD + t_c",
            "surrogate_input": "8 CST + AoA + Re/log10(Re), depending on TorchScript wrapper",
            "action_scale": cfg.action_scale,
        },
    )

    train_steps_df = pd.DataFrame(callback.train_steps)
    train_steps_df.to_csv(run_dir / "train_rollout_step_logs.csv", index=False)

    pd.DataFrame(callback.episode_rows).to_csv(
        run_dir / "train_episode_summary.csv",
        index=False,
    )

    last_step = callback.last_step_row or {}

    final_is_feasible = _is_feasible_info(last_step)
    final_is_strict_safe = _is_strict_safe_info(last_step, cfg)

    # Training boyunca bulunan en iyi tasarımları iki ayrı kritere göre seçiyoruz:
    # 1) best_train_feasible_row      -> paper feasibility: CM + t/c + geometry valid
    # 2) best_train_strict_safe_row   -> paper feasibility + local thickness safety margin
    if len(train_steps_df) > 0 and "CL_CD" in train_steps_df.columns:
        train_steps_df = train_steps_df.copy()

        feasible_df = train_steps_df[
            _feasible_mask(train_steps_df)
        ].copy()

        strict_safe_df = train_steps_df[
            _strict_safe_mask(train_steps_df, cfg)
        ].copy()

        best_train_feasible_row = _best_row_by_cl_cd(feasible_df)
        best_train_strict_safe_row = _best_row_by_cl_cd(strict_safe_df)
    else:
        best_train_feasible_row = None
        best_train_strict_safe_row = None

    train_metrics = {
        "algorithm": "PPO",
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "train_wall_time_sec": train_wall_time,

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,

        "final_CL": _safe_get(last_step, "CL"),
        "final_CD": _safe_get(last_step, "CD"),
        "final_CL_CD": _safe_get(last_step, "CL_CD"),
        "final_CM": _safe_get(last_step, "CM"),
        "final_t_c": _safe_get(last_step, "t_c"),
        "final_done_reason": _safe_get(last_step, "done_reason", ""),
        "final_solver_status": _safe_get(last_step, "solver_status", ""),
        "final_solver_error_message": _safe_get(last_step, "solver_error_message", ""),
        "final_xfoil_converged": _safe_get(last_step, "xfoil_converged", ""),
        "final_xfoil_error_message": _safe_get(last_step, "xfoil_error_message", ""),
        "final_xfoil_runtime_ms": _safe_get(last_step, "xfoil_runtime_ms", np.nan),
        "final_is_CM_feasible": _safe_get(last_step, "is_CM_feasible", ""),
        "final_is_tc_feasible": _safe_get(last_step, "is_tc_feasible", ""),
        "final_is_geometry_valid": _safe_get(last_step, "is_geometry_valid", ""),
        "final_is_feasible": final_is_feasible,
        "final_is_strict_safe": final_is_strict_safe,

        "best_train_feasible_CL": (
            _safe_get(best_train_feasible_row, "CL")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CD": (
            _safe_get(best_train_feasible_row, "CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CL_CD": (
            _safe_get(best_train_feasible_row, "CL_CD")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_CM": (
            _safe_get(best_train_feasible_row, "CM")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_t_c": (
            _safe_get(best_train_feasible_row, "t_c")
            if best_train_feasible_row is not None
            else np.nan
        ),
        "best_train_feasible_episode_id": (
            int(_safe_get(best_train_feasible_row, "episode_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),
        "best_train_feasible_step_id": (
            int(_safe_get(best_train_feasible_row, "step_id", -1))
            if best_train_feasible_row is not None
            else -1
        ),

        "final_min_local_thickness": _safe_get(last_step, "min_local_thickness", np.nan),
        "final_local_thickness_violation": _safe_get(last_step, "local_thickness_violation", np.nan),
        "final_is_local_thickness_feasible": _is_local_margin_safe_info(last_step, cfg),

        "has_train_feasible_design": best_train_feasible_row is not None,
        "best_train_strict_safe_CL": (
            _safe_get(best_train_strict_safe_row, "CL")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CD": (
            _safe_get(best_train_strict_safe_row, "CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CL_CD": (
            _safe_get(best_train_strict_safe_row, "CL_CD")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_CM": (
            _safe_get(best_train_strict_safe_row, "CM")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_t_c": (
            _safe_get(best_train_strict_safe_row, "t_c")
            if best_train_strict_safe_row is not None
            else np.nan
        ),
        "best_train_strict_safe_episode_id": (
            int(_safe_get(best_train_strict_safe_row, "episode_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "best_train_strict_safe_step_id": (
            int(_safe_get(best_train_strict_safe_row, "step_id", -1))
            if best_train_strict_safe_row is not None
            else -1
        ),
        "has_train_strict_safe_design": best_train_strict_safe_row is not None,

        "logged_train_steps": len(callback.train_steps),
        "logged_train_episodes": len(callback.episode_rows),
        "logged_training_diagnostics": callback.update_id,
    }

    pd.DataFrame([train_metrics]).to_csv(
        run_dir / "train_metrics.csv",
        index=False,
    )

    manifest = {
        "algorithm": "PPO",
        "evaluator": cfg.evaluator,
        "run_dir": str(run_dir),
        "rl_checkpoint_file": rl_checkpoint_path.name,
        "files": [
            "experiment_metadata.json",
            "training_update_logs.csv",
            "train_rollout_step_logs.csv",
            "train_episode_summary.csv",
            "train_metrics.csv",
            rl_checkpoint_path.name,
        ],
    }

    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return run_dir


def evaluate_td3(
    cfg: ExperimentConfig,
    run_dir: Optional[Path],
    episodes: int = 10,
    aoa_sweep: str = "-2,0,2,4,6,8",
):
    # Kullanıcı --run-dir verirse bunu train run klasörü olarak kabul ediyoruz.
    train_run_dir = Path(run_dir) if run_dir is not None else None

    # Model path'ini train run klasöründen veya explicit --rl-checkpoint-path'ten çöz.
    rl_checkpoint_path = _resolve_rl_checkpoint_path(cfg, train_run_dir)
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    # Evaluation çıktıları train run altında ayrı eval/run_* klasörüne yazılsın.
    if train_run_dir is not None:
        eval_run_dir = _create_eval_run_dir(train_run_dir)
    else:
        # Eğer sadece explicit checkpoint path verilmişse,
        # checkpoint'in bulunduğu klasörü train run gibi kabul et.
        eval_run_dir = _create_eval_run_dir(rl_checkpoint_path.parent)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = TD3.load(str(rl_checkpoint_path), env=env)

    run_dir = eval_run_dir

    rollout_cols = ["experiment_id","algorithm","evaluator","seed","episode_id","step_id","global_step",
                    "state_CST_u1","state_CST_u2","state_CST_u3","state_CST_u4","state_CST_l1","state_CST_l2","state_CST_l3","state_CST_l4",
                    "AoA","Re","log10_Re","CL","CD","CM","CL_CD","t_c",
                    "action_u1","action_u2","action_u3","action_u4","action_l1","action_l2","action_l3","action_l4",
                    "action_norm","upper_action_norm","lower_action_norm","action_max_abs","action_saturation_count",
                    "next_CST_u1","next_CST_u2","next_CST_u3","next_CST_u4","next_CST_l1","next_CST_l2","next_CST_l3","next_CST_l4",
                    "delta_CST_u1","delta_CST_u2","delta_CST_u3","delta_CST_u4","delta_CST_l1","delta_CST_l2","delta_CST_l3","delta_CST_l4",
                    "CL_pred","CD_pred","CM_pred","CL_CD_pred","is_CM_feasible","is_tc_feasible","is_geometry_valid",
                    "reward_total","reward_objective_term","reward_CL_CD_term","reward_CM_penalty","reward_tc_penalty","reward_local_thickness_penalty","reward_invalid_geometry_penalty","reward_solver_error_penalty","reward_action_penalty","penalty_total","action_l2_penalty_raw",
                    "CM_lower_violation","CM_upper_violation","tc_lower_violation","tc_upper_violation","local_thickness_violation", "max_thickness",
                    "x_max_thickness",
                    "max_camber",
                    "x_max_camber",
                    "leading_edge_radius_proxy",
                    "trailing_edge_thickness",
                    "upper_surface_curvature_mean",
                    "lower_surface_curvature_mean",
                    "surface_smoothness",
                    "min_local_thickness",
                    "area_proxy",
                    "done","truncated","terminated","done_reason","solver_status","solver_error_message","xfoil_converged",
                    "xfoil_error_message",
                    "xfoil_runtime_ms",]
    policy_cols = ["episode_id","step_id","actor_action_u1","actor_action_u2","actor_action_u3","actor_action_u4","actor_action_l1","actor_action_l2","actor_action_l3","actor_action_l4","Q1","Q2","Q_min","Q_disagreement","target_Q","td_error_Q1","td_error_Q2"]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    best_record = None
    best_strict_safe_record = None
    eval_start = time.time()
    aero_wall_time_sec = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        penalties = 0.0
        action_norms = []
        best_clcd = -1e9
        initial_clcd = float(obs[14])
        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_t = model.policy.obs_to_tensor(obs)[0]
            # NOTE:
            # obs_to_tensor() validates input against *observation_space*.
            # Passing an action vector here causes a shape mismatch on SB3
            # (expected obs shape=(16,), got action shape=(8,)).
            # Build action tensor manually for critic forward pass.
            act_t = torch.as_tensor(action, dtype=torch.float32, device=obs_t.device).reshape(1, -1)
            q1, q2 = model.critic(obs_t, act_t)
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            row_done_reason = str(info.get("done_reason", ""))

            if done and row_done_reason == "max_episode_steps" and _is_feasible_info(info):
                row_done_reason = "max_episode_steps_feasible"

            with torch.no_grad():
                next_obs_t = model.policy.obs_to_tensor(nxt)[0]
                next_action_t = model.actor_target(next_obs_t)
                next_q1, next_q2 = model.critic_target(next_obs_t, next_action_t)
                next_q_min = torch.min(next_q1, next_q2)

                target_q_t = torch.as_tensor(
                    [[reward]],
                    dtype=torch.float32,
                    device=next_q_min.device,
                )

                if not done:
                    target_q_t = target_q_t + float(model.gamma) * next_q_min

            q1v = float(q1.detach().cpu().numpy().ravel()[0])
            q2v = float(q2.detach().cpu().numpy().ravel()[0])
            target_qv = float(target_q_t.detach().cpu().numpy().ravel()[0])

            td_error_q1 = target_qv - q1v
            td_error_q2 = target_qv - q2v

            norm = float(np.linalg.norm(action)); upper = float(np.linalg.norm(action[:4])); lower = float(np.linalg.norm(action[4:]))
            row = {"experiment_id": run_dir.name, "algorithm": "TD3", "evaluator": cfg.evaluator, "seed": cfg.seed, "episode_id": ep, "step_id": step, "global_step": global_step,
                   "AoA": cfg.aoa, "Re": cfg.re, "log10_Re": np.log10(cfg.re), "CL": info["CL"], "CD": info["CD"], "CM": info["CM"], "CL_CD": info["CL_CD"], "t_c": info["t_c"],
                   "CL_pred": info["CL"], "CD_pred": info["CD"], "CM_pred": info["CM"], "CL_CD_pred": info["CL_CD"],
                   "is_CM_feasible": info["is_CM_feasible"], "is_tc_feasible": info["is_tc_feasible"], "is_geometry_valid": info["is_geometry_valid"],
                   "reward_total": reward,
                    "reward_objective_term": info.get("reward_objective_term", ""),
                    "reward_CL_CD_term": info.get("reward_CL_CD_term", info.get("reward_objective_term", "")),
                    "reward_CM_penalty": info.get("reward_CM_penalty", 0.0),
                    "reward_tc_penalty": info.get("reward_tc_penalty", 0.0),
                    "reward_local_thickness_penalty": info.get("reward_local_thickness_penalty", 0.0),
                    "reward_invalid_geometry_penalty": info.get("reward_invalid_geometry_penalty", 0.0),
                    "reward_solver_error_penalty": info.get("reward_solver_error_penalty", 0.0),
                    "reward_action_penalty": info.get("reward_action_penalty", 0.0),
                    "penalty_total": info.get("penalty_total", 0.0),
                    "action_l2_penalty_raw": info.get("action_l2_penalty_raw", 0.0),
                   "CM_lower_violation": info["CM_lower_violation"], "CM_upper_violation": info["CM_upper_violation"], "tc_lower_violation": info["tc_lower_violation"], "tc_upper_violation": info["tc_upper_violation"], "local_thickness_violation": info.get("local_thickness_violation", 0.0), "max_thickness": info.get("max_thickness", ""),
                    "x_max_thickness": info.get("x_max_thickness", ""),
                    "max_camber": info.get("max_camber", ""),
                    "x_max_camber": info.get("x_max_camber", ""),
                    "leading_edge_radius_proxy": info.get("leading_edge_radius_proxy", ""),
                    "trailing_edge_thickness": info.get("trailing_edge_thickness", ""),
                    "upper_surface_curvature_mean": info.get("upper_surface_curvature_mean", ""),
                    "lower_surface_curvature_mean": info.get("lower_surface_curvature_mean", ""),
                    "surface_smoothness": info.get("surface_smoothness", ""),
                    "min_local_thickness": info.get("min_local_thickness", ""),
                    "area_proxy": info.get("area_proxy", ""),
                   "done": done, "truncated": truncated, "terminated": terminated, "done_reason": row_done_reason,"solver_status": info.get("solver_status", ""),
                    "solver_error_message": info.get("solver_error_message", ""),"xfoil_converged": info.get("xfoil_converged", ""),
                    "xfoil_error_message": info.get("xfoil_error_message", ""),
                    "xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),
                   "action_norm": norm, "upper_action_norm": upper, "lower_action_norm": lower, "action_max_abs": float(np.max(np.abs(action))), "action_saturation_count": int(np.sum(np.abs(action) >= 0.999))}
            for i in range(4):
                row[f"state_CST_u{i+1}"] = float(obs[i]); row[f"state_CST_l{i+1}"] = float(obs[4+i])
                row[f"action_u{i+1}"] = float(action[i]); row[f"action_l{i+1}"] = float(action[4+i])
                row[f"next_CST_u{i+1}"] = float(info["next_cst"][i]); row[f"next_CST_l{i+1}"] = float(info["next_cst"][4+i])
                row[f"delta_CST_u{i+1}"] = float(info["delta_cst"][i]); row[f"delta_CST_l{i+1}"] = float(info["delta_cst"][4+i])
            rollout.log(row)

            q1v = float(q1.detach().cpu().numpy().ravel()[0]); q2v = float(q2.detach().cpu().numpy().ravel()[0])
            policy.log({"episode_id": ep, "step_id": step, **{f"actor_action_u{i+1}": float(action[i]) for i in range(4)}, **{f"actor_action_l{i+1}": float(action[4+i]) for i in range(4)},
                        "Q1": q1v, "Q2": q2v, "Q_min": min(q1v, q2v), "Q_disagreement": abs(q1v-q2v), "target_Q": target_qv,
                        "td_error_Q1": td_error_q1,
                        "td_error_Q2": td_error_q2,})

            total_reward += reward
            penalties += float(info.get("penalty_total", 0.0))
            aero_wall_time_sec += float(info.get("aero_wall_time_step_sec", 0.0))
            action_norms.append(norm)
            best_clcd = max(best_clcd, info["CL_CD"])
            cl_cd_value = _safe_float_value(info.get("CL_CD", -np.inf), -np.inf)
            cd_value = _safe_float_value(info.get("CD", np.inf), np.inf)

            numeric_ok = (
                np.isfinite(cl_cd_value)
                and np.isfinite(cd_value)
                and cd_value > 0.0
            )

            solver_ok = str(info.get("solver_status", "ok")) in {"ok", "valid", ""}

            is_step_feasible = (
                _is_feasible_info(info)
                and numeric_ok
            )

            is_step_strict_safe = (
                is_step_feasible
                and _is_local_margin_safe_info(info, cfg)
            )

            if is_step_feasible:
                if (
                    best_record is None
                    or cl_cd_value > float(best_record.get("CL_CD", -np.inf))
                ):
                    best_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }

            if is_step_strict_safe:
                if (
                    best_strict_safe_record is None
                    or cl_cd_value > float(best_strict_safe_record.get("CL_CD", -np.inf))
                ):
                    best_strict_safe_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }
            obs = nxt
            step += 1
            global_step += 1
            last_info = info
        final_is_cm_feasible = _as_bool(last_info.get("is_CM_feasible", False))
        final_is_tc_feasible = _as_bool(last_info.get("is_tc_feasible", False))
        final_is_geometry_valid = _as_bool(last_info.get("is_geometry_valid", False))

        final_is_feasible = _is_feasible_info(last_info)
        final_is_strict_safe = _is_strict_safe_info(last_info, cfg)

        cm_violation = int(not final_is_cm_feasible)
        tc_violation = int(not final_is_tc_feasible)
        invalid_geometry = int(not final_is_geometry_valid)
        solver_error = int(not _is_solver_success_info(last_info))
        done_reason = str(last_info.get("done_reason", ""))

        if done_reason == "max_episode_steps" and final_is_feasible:
            done_reason = "max_episode_steps_feasible"
        summaries.append({
            "experiment_id": run_dir.name,
            "algorithm": "TD3",
            "seed": cfg.seed,
            "episode_id": ep,

            "initial_CL_CD": initial_clcd,
            "final_CL_CD": last_info["CL_CD"],
            "best_CL_CD": best_clcd,

            "final_CL": last_info["CL"],
            "final_CD": last_info["CD"],
            "final_CM": last_info["CM"],
            "final_t_c": last_info["t_c"],

            "total_reward": total_reward,
            "total_penalty": penalties,

            "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
            "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,

            # Sadece aerodinamik/kalınlık constraint ihlalleri
            "constraint_violation_count": int(cm_violation + tc_violation),

            "CM_violation_count": cm_violation,
            "tc_violation_count": tc_violation,
            "invalid_geometry_count": invalid_geometry,
            "solver_error_count": solver_error,

            "done_reason": done_reason,
            "episode_length": step,

            # Nihai feasibility: CM + t/c + geometri birlikte sağlanmalı
            "is_final_feasible": final_is_feasible,
            "is_final_strict_safe": final_is_strict_safe,
            "final_min_local_thickness": last_info.get("min_local_thickness", np.nan),
            "final_local_thickness_violation": last_info.get("local_thickness_violation", np.nan),
        })

    rollout.close(); policy.close()
    pd.DataFrame(summaries).to_csv(run_dir / "episode_summary.csv", index=False)

    # only current evaluator AoA sweep (no cross-validation)
    sweep_rows = []
    if best_record is not None:
        aoa_values = [float(x) for x in aoa_sweep.split(",") if x.strip()]
        cst = best_record["cst"]
        evaltor = _make_evaluator(cfg)
        for a in aoa_values:
            t0 = time.time()
            out = evaltor.evaluate(cst, a, cfg.re)
            runtime_ms = (time.time() - t0) * 1000.0
            aero_wall_time_sec += runtime_ms / 1000.0
            ratio = out.cl / max(out.cd, cfg.cd_lower_bound)
            geom = out.geometry_features or {}

            min_local_thickness = float(
                geom.get("min_local_thickness", np.nan)
            )

            max_thickness = float(
                geom.get("max_thickness", out.tc)
            )

            local_thickness_violation = max(
                0.0,
                float(cfg.geometry.min_local_thickness_required) - min_local_thickness,
            )
            sweep_rows.append({
                "algorithm": "TD3", "evaluator": cfg.evaluator, "episode_id": best_record["episode_id"], "step_id": best_record["step_id"],
                "AoA": a, "Re": cfg.re,
                **{f"CST_u{i+1}": float(cst[i]) for i in range(4)}, **{f"CST_l{i+1}": float(cst[4+i]) for i in range(4)},
                "CL_pred": out.cl, "CD_pred": out.cd, "CM_pred": out.cm, "CL_CD_pred": ratio, "t_c": float(out.tc),
                "is_geometry_valid": bool(out.is_geometry_valid),
                "min_local_thickness": min_local_thickness,
                "max_thickness": max_thickness,
                "local_thickness_violation": float(local_thickness_violation),
                "solver_status": out.solver_status,
                "solver_error_message": out.solver_error_message,
                "xfoil_converged": True if cfg.evaluator == "xfoil" else "", "xfoil_iterations": "", "xfoil_error_message": "", "runtime_ms": runtime_ms,
            })
    sweep_filename = f"{cfg.evaluator.lower()}_aoa_sweep_logs.csv"

    sweep_cols = [
        "algorithm",
        "evaluator",
        "episode_id",
        "step_id",
        "AoA",
        "Re",

        "CST_u1",
        "CST_u2",
        "CST_u3",
        "CST_u4",
        "CST_l1",
        "CST_l2",
        "CST_l3",
        "CST_l4",

        "CL_pred",
        "CD_pred",
        "CM_pred",
        "CL_CD_pred",

        "t_c",
        "is_geometry_valid",
        "min_local_thickness",
        "max_thickness",
        "local_thickness_violation",
        "solver_status",
        "solver_error_message",

        "xfoil_converged",
        "xfoil_iterations",
        "xfoil_error_message",
        "runtime_ms",
    ]

    pd.DataFrame(sweep_rows, columns=sweep_cols).to_csv(
        run_dir / sweep_filename,
        index=False,
    )

    summary_df = pd.DataFrame(summaries)
    best_idx = summary_df["best_CL_CD"].astype(float).idxmax() if len(summary_df) else None
    best_row = summary_df.loc[best_idx] if best_idx is not None else None

    if len(summary_df) > 0:
        final_mean_CL_CD = pd.to_numeric(
            summary_df.get("final_CL_CD", np.nan),
            errors="coerce",
        ).mean()

        final_feasible_episode_count = int(
            summary_df.get("is_final_feasible", pd.Series(False, index=summary_df.index))
            .map(_as_bool)
            .sum()
        )
        solver_error_episode_count = int(
            pd.to_numeric(
                summary_df.get("solver_error_count", 0),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        final_strict_safe_episode_count = int(
            summary_df.get(
                "is_final_strict_safe",
                pd.Series(False, index=summary_df.index),
            )
            .map(_as_bool)
            .sum()
        )

        done_reason_series = summary_df.get(
            "done_reason",
            pd.Series("", index=summary_df.index),
        ).astype(str)

        invalid_geometry_episode_count = int(
            done_reason_series.str.startswith("invalid_geometry").sum()
        )

        negative_local_thickness_episode_count = int(
            (done_reason_series == "invalid_geometry_negative_local_thickness").sum()
        )
    else:
        final_mean_CL_CD = np.nan
        final_feasible_episode_count = 0
        invalid_geometry_episode_count = 0
        final_strict_safe_episode_count = 0
        negative_local_thickness_episode_count = 0
        final_strict_safe_episode_count = 0
    eval_metrics = {
        "algorithm": "TD3",
        "evaluator": cfg.evaluator,
        "aero_wall_time_sec": aero_wall_time_sec,
        "eval_wall_time_sec": time.time() - eval_start,
        "episodes": episodes,

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,

        # Evaluation final-state stabilitesi
        "final_mean_CL_CD": final_mean_CL_CD,
        "final_feasible_episode_count": final_feasible_episode_count,
        "invalid_geometry_episode_count": invalid_geometry_episode_count,

        "negative_local_thickness_episode_count": negative_local_thickness_episode_count,

        # Deterministic evaluation boyunca bulunan en iyi feasible tasarım
        "best_feasible_CL": (
            float(best_record["CL"]) if best_record is not None else np.nan
        ),
        "best_feasible_CD": (
            float(best_record["CD"]) if best_record is not None else np.nan
        ),
        "best_feasible_CL_CD": (
            float(best_record["CL_CD"]) if best_record is not None else np.nan
        ),
        "best_feasible_CM": (
            float(best_record["CM"]) if best_record is not None else np.nan
        ),
        "best_feasible_t_c": (
            float(best_record["t_c"]) if best_record is not None else np.nan
        ),
        "best_feasible_episode_id": (
            int(best_record["episode_id"]) if best_record is not None else -1
        ),
        "best_feasible_step_id": (
            int(best_record["step_id"]) if best_record is not None else -1
        ),
        "has_feasible_design": bool(best_record is not None),
        "final_strict_safe_episode_count": final_strict_safe_episode_count,
        "best_strict_safe_CL": (
            float(best_strict_safe_record["CL"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CD": (
            float(best_strict_safe_record["CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CL_CD": (
            float(best_strict_safe_record["CL_CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CM": (
            float(best_strict_safe_record["CM"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_t_c": (
            float(best_strict_safe_record["t_c"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_episode_id": (
            int(best_strict_safe_record["episode_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "best_strict_safe_step_id": (
            int(best_strict_safe_record["step_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "has_strict_safe_design": bool(best_strict_safe_record is not None),
        "solver_error_episode_count": solver_error_episode_count,
    }
    pd.DataFrame([eval_metrics]).to_csv(run_dir / "eval_metrics.csv", index=False)

    with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({"eval_wall_time_sec": eval_metrics["eval_wall_time_sec"], "aero_wall_time_sec": aero_wall_time_sec, "episodes": episodes, "aoa_sweep": aoa_sweep}, f, indent=2)
    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "xai_v1",
                "run_type": "evaluate",
                "source_checkpoint": cfg.rl_checkpoint_path,
                "files": [
                    "rollout_step_logs.csv",
                    "policy_outputs.csv",
                    "episode_summary.csv",
                    sweep_filename,
                    "eval_metrics.csv",
                    "eval_summary.json",
                    rl_checkpoint_path.name,
                ],
            },
            f,
            indent=2,
        )

def evaluate_sac(
    cfg: ExperimentConfig,
    run_dir: Optional[Path],
    episodes: int = 10,
    aoa_sweep: str = "-2,0,2,4,6,8",
):
    """
    SAC deterministic evaluation + XAI logging.

    Üretilen ana dosyalar:
      - rollout_step_logs.csv
      - policy_outputs.csv
      - episode_summary.csv
      - eval_metrics.csv
      - <evaluator>_aoa_sweep_logs.csv
      - eval_summary.json
      - xai_manifest.json
    """

    cfg.algorithm = "sac"

    # Kullanıcı --run-dir verirse bunu train run klasörü kabul ediyoruz.
    train_run_dir = Path(run_dir) if run_dir is not None else None

    # Model path'ini train run klasöründen veya explicit --rl-checkpoint-path'ten çöz.
    rl_checkpoint_path = _resolve_rl_checkpoint_path(cfg, train_run_dir)
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    # Evaluation çıktıları train run altında ayrı eval/run_* klasörüne yazılsın.
    if train_run_dir is not None:
        eval_run_dir = _create_eval_run_dir(train_run_dir)
    else:
        # Eğer sadece explicit checkpoint path verilmişse,
        # checkpoint'in bulunduğu klasörü train run gibi kabul et.
        eval_run_dir = _create_eval_run_dir(rl_checkpoint_path.parent)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = SAC.load(str(rl_checkpoint_path), env=env)

    run_dir = eval_run_dir

    rollout_cols = [
        "experiment_id",
        "algorithm",
        "evaluator",
        "seed",
        "episode_id",
        "step_id",
        "global_step",

        "state_CST_u1",
        "state_CST_u2",
        "state_CST_u3",
        "state_CST_u4",
        "state_CST_l1",
        "state_CST_l2",
        "state_CST_l3",
        "state_CST_l4",

        "AoA",
        "Re",
        "log10_Re",
        "CL",
        "CD",
        "CM",
        "CL_CD",
        "t_c",

        "action_u1",
        "action_u2",
        "action_u3",
        "action_u4",
        "action_l1",
        "action_l2",
        "action_l3",
        "action_l4",

        "action_norm",
        "upper_action_norm",
        "lower_action_norm",
        "action_max_abs",
        "action_saturation_count",

        "next_CST_u1",
        "next_CST_u2",
        "next_CST_u3",
        "next_CST_u4",
        "next_CST_l1",
        "next_CST_l2",
        "next_CST_l3",
        "next_CST_l4",

        "delta_CST_u1",
        "delta_CST_u2",
        "delta_CST_u3",
        "delta_CST_u4",
        "delta_CST_l1",
        "delta_CST_l2",
        "delta_CST_l3",
        "delta_CST_l4",

        "CL_pred",
        "CD_pred",
        "CM_pred",
        "CL_CD_pred",

        "is_CM_feasible",
        "is_tc_feasible",
        "is_geometry_valid",

        "reward_total",
        "reward_objective_term",
        "reward_CL_CD_term",
        "reward_CM_penalty",
        "reward_tc_penalty",
        "reward_local_thickness_penalty",
        "reward_invalid_geometry_penalty",
        "reward_solver_error_penalty",
        "reward_action_penalty",
        "penalty_total",
        "action_l2_penalty_raw",

        "CM_lower_violation",
        "CM_upper_violation",
        "tc_lower_violation",
        "tc_upper_violation",
        "local_thickness_violation",

        "max_thickness",
        "x_max_thickness",
        "max_camber",
        "x_max_camber",
        "leading_edge_radius_proxy",
        "trailing_edge_thickness",
        "upper_surface_curvature_mean",
        "lower_surface_curvature_mean",
        "surface_smoothness",
        "min_local_thickness",
        "area_proxy",

        "done",
        "truncated",
        "terminated",
        "done_reason",
        "solver_status",
        "solver_error_message",
        "xfoil_converged",
        "xfoil_error_message",
        "xfoil_runtime_ms",
    ]

    policy_cols = [
        "experiment_id",
        "algorithm",
        "evaluator",
        "seed",
        "episode_id",
        "step_id",
        "global_step",
    ]

    for prefix in [
        "actor_mean",
        "actor_log_std",
        "actor_std",
        "sampled_action",
        "deterministic_action",
    ]:
        for name in ["u1", "u2", "u3", "u4", "l1", "l2", "l3", "l4"]:
            policy_cols.append(f"{prefix}_{name}")

    policy_cols += [
        "log_prob_action",
        "policy_entropy",
        "Q1_s_a",
        "Q2_s_a",
        "Q_min_s_a",
        "Q_mean_s_a",
        "Q_disagreement_abs",
        "Q_disagreement_relative",
        "target_Q",
        "soft_value_target",
        "td_error_Q1",
        "td_error_Q2",
        "alpha",
        "target_entropy",
    ]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    best_record = None
    best_strict_safe_record = None
    eval_start = time.time()
    aero_wall_time_sec = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)

        done = False
        step = 0
        total_reward = 0.0
        penalties = 0.0
        action_norms = []
        best_clcd = -np.inf
        initial_clcd = float(obs[14])
        last_info = None

        while not done:
            # SAC evaluation deterministic yapılır.
            # XAI için ayrıca policy distribution bilgisi policy_outputs.csv'ye yazılır.
            action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            row_done_reason = _standardize_done_reason(info, done=done)

            norm = float(np.linalg.norm(action))
            upper = float(np.linalg.norm(action[:4]))
            lower = float(np.linalg.norm(action[4:]))

            row = {
                "experiment_id": run_dir.name,
                "algorithm": "SAC",
                "evaluator": cfg.evaluator,
                "seed": cfg.seed,
                "episode_id": ep,
                "step_id": step,
                "global_step": global_step,

                "AoA": float(cfg.aoa),
                "Re": float(cfg.re),
                "log10_Re": float(np.log10(cfg.re)),

                "CL": float(info["CL"]),
                "CD": float(info["CD"]),
                "CM": float(info["CM"]),
                "CL_CD": float(info["CL_CD"]),
                "t_c": float(info["t_c"]),

                "CL_pred": float(info["CL"]),
                "CD_pred": float(info["CD"]),
                "CM_pred": float(info["CM"]),
                "CL_CD_pred": float(info["CL_CD"]),

                "is_CM_feasible": info.get("is_CM_feasible", False),
                "is_tc_feasible": info.get("is_tc_feasible", False),
                "is_geometry_valid": info.get("is_geometry_valid", False),

                "reward_total": float(reward),
                "reward_objective_term": info.get("reward_objective_term", ""),
                "reward_CL_CD_term": info.get(
                    "reward_CL_CD_term",
                    info.get("reward_objective_term", ""),
                ),
                "reward_CM_penalty": info.get("reward_CM_penalty", 0.0),
                "reward_tc_penalty": info.get("reward_tc_penalty", 0.0),
                "reward_local_thickness_penalty": info.get(
                    "reward_local_thickness_penalty",
                    0.0,
                ),
                "reward_invalid_geometry_penalty": info.get(
                    "reward_invalid_geometry_penalty",
                    0.0,
                ),
                "reward_solver_error_penalty": info.get(
                    "reward_solver_error_penalty",
                    0.0,
                ),
                "reward_action_penalty": info.get("reward_action_penalty", 0.0),
                "penalty_total": info.get("penalty_total", 0.0),
                "action_l2_penalty_raw": info.get("action_l2_penalty_raw", 0.0),

                "CM_lower_violation": info.get("CM_lower_violation", 0.0),
                "CM_upper_violation": info.get("CM_upper_violation", 0.0),
                "tc_lower_violation": info.get("tc_lower_violation", 0.0),
                "tc_upper_violation": info.get("tc_upper_violation", 0.0),
                "local_thickness_violation": info.get(
                    "local_thickness_violation",
                    0.0,
                ),

                "max_thickness": info.get("max_thickness", ""),
                "x_max_thickness": info.get("x_max_thickness", ""),
                "max_camber": info.get("max_camber", ""),
                "x_max_camber": info.get("x_max_camber", ""),
                "leading_edge_radius_proxy": info.get(
                    "leading_edge_radius_proxy",
                    "",
                ),
                "trailing_edge_thickness": info.get(
                    "trailing_edge_thickness",
                    "",
                ),
                "upper_surface_curvature_mean": info.get(
                    "upper_surface_curvature_mean",
                    "",
                ),
                "lower_surface_curvature_mean": info.get(
                    "lower_surface_curvature_mean",
                    "",
                ),
                "surface_smoothness": info.get("surface_smoothness", ""),
                "min_local_thickness": info.get("min_local_thickness", ""),
                "area_proxy": info.get("area_proxy", ""),

                "done": done,
                "truncated": bool(truncated),
                "terminated": bool(terminated),
                "done_reason": row_done_reason,
                "solver_status": info.get("solver_status", ""),
                "solver_error_message": info.get("solver_error_message", ""),
                "xfoil_converged": info.get("xfoil_converged", ""),
                "xfoil_error_message": info.get("xfoil_error_message", ""),
                "xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),

                "action_norm": norm,
                "upper_action_norm": upper,
                "lower_action_norm": lower,
                "action_max_abs": float(np.max(np.abs(action))),
                "action_saturation_count": int(np.sum(np.abs(action) >= 0.999)),
            }

            for i in range(4):
                row[f"state_CST_u{i + 1}"] = float(obs[i])
                row[f"state_CST_l{i + 1}"] = float(obs[4 + i])

                row[f"action_u{i + 1}"] = float(action[i])
                row[f"action_l{i + 1}"] = float(action[4 + i])

                row[f"next_CST_u{i + 1}"] = float(info["next_cst"][i])
                row[f"next_CST_l{i + 1}"] = float(info["next_cst"][4 + i])

                row[f"delta_CST_u{i + 1}"] = float(info["delta_cst"][i])
                row[f"delta_CST_l{i + 1}"] = float(info["delta_cst"][4 + i])

            rollout.log(row)

            policy_extra = _sac_policy_diagnostics(
                model=model,
                obs_np=obs,
                action_np=action,
                next_obs_np=next_obs,
                reward=reward,
                done=done,
            )

            policy_row = {
                "experiment_id": run_dir.name,
                "algorithm": "SAC",
                "evaluator": cfg.evaluator,
                "seed": cfg.seed,
                "episode_id": ep,
                "step_id": step,
                "global_step": global_step,
                **policy_extra,
            }

            policy.log(policy_row)

            total_reward += float(reward)
            penalties += float(info.get("penalty_total", 0.0))
            aero_wall_time_sec += float(info.get("aero_wall_time_step_sec", 0.0))
            action_norms.append(norm)
            best_clcd = max(best_clcd, float(info.get("CL_CD", -np.inf)))

            cl_cd_value = _safe_float_value(info.get("CL_CD", -np.inf), -np.inf)
            cd_value = _safe_float_value(info.get("CD", np.inf), np.inf)

            numeric_ok = (
                np.isfinite(cl_cd_value)
                and np.isfinite(cd_value)
                and cd_value > 0.0
            )

            

            is_step_feasible = (
                _is_feasible_info(info)
                and numeric_ok
            )

            is_step_strict_safe = (
                is_step_feasible
                and _is_local_margin_safe_info(info, cfg)
            )

            if is_step_feasible:
                if (
                    best_record is None
                    or cl_cd_value > float(best_record.get("CL_CD", -np.inf))
                ):
                    best_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }

            if is_step_strict_safe:
                if (
                    best_strict_safe_record is None
                    or cl_cd_value > float(best_strict_safe_record.get("CL_CD", -np.inf))
                ):
                    best_strict_safe_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }

            obs = next_obs
            step += 1
            global_step += 1
            last_info = info

        if last_info is None:
            continue

        final_is_cm_feasible = _as_bool(last_info.get("is_CM_feasible", False))
        final_is_tc_feasible = _as_bool(last_info.get("is_tc_feasible", False))
        final_is_geometry_valid = _as_bool(last_info.get("is_geometry_valid", False))

        final_is_feasible = _is_feasible_info(last_info)
        final_is_strict_safe = _is_strict_safe_info(last_info, cfg)

        cm_violation = int(not final_is_cm_feasible)
        tc_violation = int(not final_is_tc_feasible)
        invalid_geometry = int(not final_is_geometry_valid)
        solver_error = int(not _is_solver_success_info(last_info))

        done_reason = _standardize_done_reason(last_info, done=True)

        summaries.append(
            {
                "experiment_id": run_dir.name,
                "algorithm": "SAC",
                "seed": cfg.seed,
                "episode_id": ep,

                "initial_CL_CD": initial_clcd,
                "final_CL_CD": last_info["CL_CD"],
                "best_CL_CD": best_clcd,

                "final_CL": last_info["CL"],
                "final_CD": last_info["CD"],
                "final_CM": last_info["CM"],
                "final_t_c": last_info["t_c"],

                "total_reward": total_reward,
                "total_penalty": penalties,

                "mean_action_norm": (
                    float(np.mean(action_norms)) if action_norms else 0.0
                ),
                "max_action_norm": (
                    float(np.max(action_norms)) if action_norms else 0.0
                ),

                "constraint_violation_count": int(cm_violation + tc_violation),
                "CM_violation_count": cm_violation,
                "tc_violation_count": tc_violation,
                "invalid_geometry_count": invalid_geometry,
                "solver_error_count": solver_error,

                "done_reason": done_reason,
                "episode_length": step,
                "is_final_feasible": final_is_feasible,
                "is_final_strict_safe": final_is_strict_safe,
                "final_min_local_thickness": last_info.get("min_local_thickness", np.nan),
                "final_local_thickness_violation": last_info.get("local_thickness_violation", np.nan),
            }
        )

    rollout.close()
    policy.close()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "episode_summary.csv", index=False)

    # AoA sweep: yalnızca mevcut evaluator ile yapılır.
    # Surrogate evaluator ise surrogate_aoa_sweep_logs.csv,
    # ileride gerçek XFOIL evaluator aktifse xfoil_aoa_sweep_logs.csv üretir.
    sweep_rows = []

    if best_record is not None:
        aoa_values = [float(x) for x in aoa_sweep.split(",") if x.strip()]
        cst = best_record["cst"]
        evaltor = _make_evaluator(cfg)

        for a in aoa_values:
            t0 = time.time()
            out = evaltor.evaluate(cst, a, cfg.re)
            runtime_ms = (time.time() - t0) * 1000.0

            aero_wall_time_sec += runtime_ms / 1000.0
            ratio = float(out.cl / max(out.cd, cfg.cd_lower_bound))
            geom = out.geometry_features or {}

            min_local_thickness = float(
                geom.get("min_local_thickness", np.nan)
            )

            max_thickness = float(
                geom.get("max_thickness", out.tc)
            )

            local_thickness_violation = max(
                0.0,
                float(cfg.geometry.min_local_thickness_required) - min_local_thickness,
            )

            sweep_rows.append(
                {
                    "algorithm": "SAC",
                    "evaluator": cfg.evaluator,
                    "episode_id": best_record["episode_id"],
                    "step_id": best_record["step_id"],
                    "AoA": a,
                    "Re": cfg.re,

                    **{
                        f"CST_u{i + 1}": float(cst[i])
                        for i in range(4)
                    },
                    **{
                        f"CST_l{i + 1}": float(cst[4 + i])
                        for i in range(4)
                    },

                    "CL_pred": float(out.cl),
                    "CD_pred": float(out.cd),
                    "CM_pred": float(out.cm),
                    "CL_CD_pred": ratio,

                    "t_c": float(out.tc),
                    "is_geometry_valid": bool(out.is_geometry_valid),
                    "min_local_thickness": min_local_thickness,
                    "max_thickness": max_thickness,
                    "local_thickness_violation": float(local_thickness_violation),
                    "solver_status": out.solver_status,
                    "solver_error_message": out.solver_error_message,

                    "xfoil_converged": (
                        bool(out.solver_status == "ok")
                        if cfg.evaluator == "xfoil"
                        else ""
                    ),
                    "xfoil_iterations": "",
                    "xfoil_error_message": (
                        out.solver_error_message
                        if cfg.evaluator == "xfoil"
                        else ""
                    ),
                    "runtime_ms": runtime_ms,
                }
            )

    sweep_filename = f"{cfg.evaluator.lower()}_aoa_sweep_logs.csv"

    sweep_cols = [
        "algorithm",
        "evaluator",
        "episode_id",
        "step_id",
        "AoA",
        "Re",

        "CST_u1",
        "CST_u2",
        "CST_u3",
        "CST_u4",
        "CST_l1",
        "CST_l2",
        "CST_l3",
        "CST_l4",

        "CL_pred",
        "CD_pred",
        "CM_pred",
        "CL_CD_pred",

        "t_c",
        "is_geometry_valid",
        "min_local_thickness",
        "max_thickness",
        "local_thickness_violation",
        "solver_status",
        "solver_error_message",

        "xfoil_converged",
        "xfoil_iterations",
        "xfoil_error_message",
        "runtime_ms",
    ]

    pd.DataFrame(sweep_rows, columns=sweep_cols).to_csv(
        run_dir / sweep_filename,
        index=False,
    )

    if len(summary_df) > 0:
        final_mean_CL_CD = pd.to_numeric(
            summary_df.get("final_CL_CD", np.nan),
            errors="coerce",
        ).mean()

        final_feasible_episode_count = int(
            summary_df.get(
                "is_final_feasible",
                pd.Series(False, index=summary_df.index),
            )
            .map(_as_bool)
            .sum()
        )
        solver_error_episode_count = int(
            pd.to_numeric(
                summary_df.get("solver_error_count", 0),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        final_strict_safe_episode_count = int(
            summary_df.get(
                "is_final_strict_safe",
                pd.Series(False, index=summary_df.index),
            )
            .map(_as_bool)
            .sum()
        )

        done_reason_series = summary_df.get(
            "done_reason",
            pd.Series("", index=summary_df.index),
        ).astype(str)

        invalid_geometry_episode_count = int(
            done_reason_series.str.startswith("invalid_geometry").sum()
        )

        negative_local_thickness_episode_count = int(
            (
                done_reason_series
                == "invalid_geometry_negative_local_thickness"
            ).sum()
        )
    else:
        final_mean_CL_CD = np.nan
        final_feasible_episode_count = 0
        invalid_geometry_episode_count = 0
        negative_local_thickness_episode_count = 0
        final_strict_safe_episode_count = 0
        solver_error_episode_count = 0
    eval_metrics = {
        "algorithm": "SAC",
        "evaluator": cfg.evaluator,
        "aero_wall_time_sec": aero_wall_time_sec,
        "eval_wall_time_sec": time.time() - eval_start,
        "episodes": episodes,

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,

        "final_mean_CL_CD": final_mean_CL_CD,
        "final_feasible_episode_count": final_feasible_episode_count,
        "invalid_geometry_episode_count": invalid_geometry_episode_count,
        "negative_local_thickness_episode_count": negative_local_thickness_episode_count,

        "best_feasible_CL": (
            float(best_record["CL"]) if best_record is not None else np.nan
        ),
        "best_feasible_CD": (
            float(best_record["CD"]) if best_record is not None else np.nan
        ),
        "best_feasible_CL_CD": (
            float(best_record["CL_CD"]) if best_record is not None else np.nan
        ),
        "best_feasible_CM": (
            float(best_record["CM"]) if best_record is not None else np.nan
        ),
        "best_feasible_t_c": (
            float(best_record["t_c"]) if best_record is not None else np.nan
        ),
        "best_feasible_episode_id": (
            int(best_record["episode_id"]) if best_record is not None else -1
        ),
        "best_feasible_step_id": (
            int(best_record["step_id"]) if best_record is not None else -1
        ),
        "has_feasible_design": bool(best_record is not None),
        "final_strict_safe_episode_count": final_strict_safe_episode_count,
        "best_strict_safe_CL": (
            float(best_strict_safe_record["CL"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CD": (
            float(best_strict_safe_record["CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CL_CD": (
            float(best_strict_safe_record["CL_CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CM": (
            float(best_strict_safe_record["CM"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_t_c": (
            float(best_strict_safe_record["t_c"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_episode_id": (
            int(best_strict_safe_record["episode_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "best_strict_safe_step_id": (
            int(best_strict_safe_record["step_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "has_strict_safe_design": bool(best_strict_safe_record is not None),
        "solver_error_episode_count": solver_error_episode_count,
    }

    pd.DataFrame([eval_metrics]).to_csv(
        run_dir / "eval_metrics.csv",
        index=False,
    )

    with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": "SAC",
                "evaluator": cfg.evaluator,
                "eval_wall_time_sec": eval_metrics["eval_wall_time_sec"],
                "aero_wall_time_sec": aero_wall_time_sec,
                "episodes": episodes,
                "aoa_sweep": aoa_sweep,
                "source_checkpoint": str(rl_checkpoint_path),
            },
            f,
            indent=2,
        )

    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "xai_v1",
                "algorithm": "SAC",
                "evaluator": cfg.evaluator,
                "run_type": "evaluate",
                "run_dir": str(run_dir),
                "source_checkpoint": str(rl_checkpoint_path),
                "files": [
                    "eval_metrics.csv",
                    "episode_summary.csv",
                    "rollout_step_logs.csv",
                    "policy_outputs.csv",
                    sweep_filename,
                    "eval_summary.json",
                ],
            },
            f,
            indent=2,
        )


def evaluate_ppo(
    cfg: ExperimentConfig,
    run_dir: Optional[Path],
    episodes: int = 10,
    aoa_sweep: str = "-2,0,2,4,6,8",
):
    cfg.algorithm = "ppo"

    train_run_dir = Path(run_dir) if run_dir is not None else None

    rl_checkpoint_path = _resolve_rl_checkpoint_path(cfg, train_run_dir)
    cfg.rl_checkpoint_path = str(rl_checkpoint_path)

    if train_run_dir is not None:
        eval_run_dir = _create_eval_run_dir(train_run_dir)
    else:
        eval_run_dir = _create_eval_run_dir(rl_checkpoint_path.parent)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = PPO.load(str(rl_checkpoint_path), env=env)

    run_dir = eval_run_dir

    rollout_cols = [
        "experiment_id", "algorithm", "evaluator", "seed",
        "episode_id", "step_id", "global_step",

        "state_CST_u1", "state_CST_u2", "state_CST_u3", "state_CST_u4",
        "state_CST_l1", "state_CST_l2", "state_CST_l3", "state_CST_l4",

        "AoA", "Re", "log10_Re",
        "CL", "CD", "CM", "CL_CD", "t_c",

        "action_u1", "action_u2", "action_u3", "action_u4",
        "action_l1", "action_l2", "action_l3", "action_l4",

        "action_norm", "upper_action_norm", "lower_action_norm",
        "action_max_abs", "action_saturation_count",

        "next_CST_u1", "next_CST_u2", "next_CST_u3", "next_CST_u4",
        "next_CST_l1", "next_CST_l2", "next_CST_l3", "next_CST_l4",

        "delta_CST_u1", "delta_CST_u2", "delta_CST_u3", "delta_CST_u4",
        "delta_CST_l1", "delta_CST_l2", "delta_CST_l3", "delta_CST_l4",

        "CL_pred", "CD_pred", "CM_pred", "CL_CD_pred",

        "is_CM_feasible", "is_tc_feasible", "is_geometry_valid",

        "reward_total", "reward_objective_term", "reward_CL_CD_term",
        "reward_CM_penalty", "reward_tc_penalty",
        "reward_local_thickness_penalty",
        "reward_invalid_geometry_penalty",
        "reward_solver_error_penalty",
        "reward_action_penalty",
        "penalty_total", "action_l2_penalty_raw",

        "CM_lower_violation", "CM_upper_violation",
        "tc_lower_violation", "tc_upper_violation",
        "local_thickness_violation",

        "max_thickness", "x_max_thickness",
        "max_camber", "x_max_camber",
        "leading_edge_radius_proxy", "trailing_edge_thickness",
        "upper_surface_curvature_mean", "lower_surface_curvature_mean",
        "surface_smoothness", "min_local_thickness", "area_proxy",

        "done", "truncated", "terminated", "done_reason",
        "solver_status", "solver_error_message", "xfoil_converged",
        "xfoil_error_message",
        "xfoil_runtime_ms",
    ]

    policy_cols = [
        "experiment_id", "algorithm", "evaluator", "seed",
        "episode_id", "step_id", "global_step",
    ]

    for prefix in [
        "policy_mean",
        "policy_std",
        "sampled_action",
        "deterministic_action",
    ]:
        for name in ["u1", "u2", "u3", "u4", "l1", "l2", "l3", "l4"]:
            policy_cols.append(f"{prefix}_{name}")

    policy_cols += [
        "log_prob_action",
        "entropy",
        "value_V_s",
        "reward",
        "return_G_t",
        "advantage_A_t",
        "normalized_advantage",
        "td_error_delta",
    ]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    best_record = None
    best_strict_safe_record = None
    eval_start = time.time()
    aero_wall_time_sec = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)

        done = False
        step = 0
        total_reward = 0.0
        penalties = 0.0
        action_norms = []
        best_clcd = -1e9
        initial_clcd = float(obs[14])
        last_info = None

        episode_policy_rows = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            policy_extra = _ppo_policy_diagnostics(
                model=model,
                obs_np=obs,
                action_np=action,
            )

            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            row_done_reason = str(info.get("done_reason", ""))

            if done and row_done_reason == "max_episode_steps" and _is_feasible_info(info):
                row_done_reason = "max_episode_steps_feasible"
            norm = float(np.linalg.norm(action))
            upper = float(np.linalg.norm(action[:4]))
            lower = float(np.linalg.norm(action[4:]))

            row = {
                "experiment_id": run_dir.name,
                "algorithm": "PPO",
                "evaluator": cfg.evaluator,
                "seed": cfg.seed,
                "episode_id": ep,
                "step_id": step,
                "global_step": global_step,

                "AoA": cfg.aoa,
                "Re": cfg.re,
                "log10_Re": np.log10(cfg.re),

                "CL": info["CL"],
                "CD": info["CD"],
                "CM": info["CM"],
                "CL_CD": info["CL_CD"],
                "t_c": info["t_c"],

                "CL_pred": info.get("CL_pred", info["CL"]),
                "CD_pred": info.get("CD_pred", info["CD"]),
                "CM_pred": info.get("CM_pred", info["CM"]),
                "CL_CD_pred": info.get("CL_CD_pred", info["CL_CD"]),

                "is_CM_feasible": info["is_CM_feasible"],
                "is_tc_feasible": info["is_tc_feasible"],
                "is_geometry_valid": info["is_geometry_valid"],

                "reward_total": reward,
                "reward_objective_term": info.get("reward_objective_term", ""),
                "reward_CL_CD_term": info.get("reward_CL_CD_term", ""),
                "reward_CM_penalty": info.get("reward_CM_penalty", 0.0),
                "reward_tc_penalty": info.get("reward_tc_penalty", 0.0),
                "reward_local_thickness_penalty": info.get("reward_local_thickness_penalty", 0.0),
                "reward_invalid_geometry_penalty": info.get("reward_invalid_geometry_penalty", 0.0),
                "reward_solver_error_penalty": info.get("reward_solver_error_penalty", 0.0),
                "reward_action_penalty": info.get("reward_action_penalty", 0.0),
                "penalty_total": info.get("penalty_total", 0.0),
                "action_l2_penalty_raw": info.get("action_l2_penalty_raw", 0.0),

                "CM_lower_violation": info.get("CM_lower_violation", 0.0),
                "CM_upper_violation": info.get("CM_upper_violation", 0.0),
                "tc_lower_violation": info.get("tc_lower_violation", 0.0),
                "tc_upper_violation": info.get("tc_upper_violation", 0.0),
                "local_thickness_violation": info.get("local_thickness_violation", 0.0),

                "max_thickness": info.get("max_thickness", ""),
                "x_max_thickness": info.get("x_max_thickness", ""),
                "max_camber": info.get("max_camber", ""),
                "x_max_camber": info.get("x_max_camber", ""),
                "leading_edge_radius_proxy": info.get("leading_edge_radius_proxy", ""),
                "trailing_edge_thickness": info.get("trailing_edge_thickness", ""),
                "upper_surface_curvature_mean": info.get("upper_surface_curvature_mean", ""),
                "lower_surface_curvature_mean": info.get("lower_surface_curvature_mean", ""),
                "surface_smoothness": info.get("surface_smoothness", ""),
                "min_local_thickness": info.get("min_local_thickness", ""),
                "area_proxy": info.get("area_proxy", ""),

                "done": done,
                "truncated": truncated,
                "terminated": terminated,
                "done_reason": row_done_reason,
                "solver_status": info.get("solver_status", ""),
                "solver_error_message": info.get("solver_error_message", ""),
                "xfoil_converged": info.get("xfoil_converged", ""),
                "xfoil_error_message": info.get("xfoil_error_message", ""),
                "xfoil_runtime_ms": info.get("xfoil_runtime_ms", ""),

                "action_norm": norm,
                "upper_action_norm": upper,
                "lower_action_norm": lower,
                "action_max_abs": float(np.max(np.abs(action))),
                "action_saturation_count": int(np.sum(np.abs(action) >= 0.999)),
            }

            for i in range(4):
                row[f"state_CST_u{i+1}"] = float(obs[i])
                row[f"state_CST_l{i+1}"] = float(obs[4+i])

                row[f"action_u{i+1}"] = float(action[i])
                row[f"action_l{i+1}"] = float(action[4+i])

                row[f"next_CST_u{i+1}"] = float(info["next_cst"][i])
                row[f"next_CST_l{i+1}"] = float(info["next_cst"][4+i])

                row[f"delta_CST_u{i+1}"] = float(info["delta_cst"][i])
                row[f"delta_CST_l{i+1}"] = float(info["delta_cst"][4+i])

            rollout.log(row)

            policy_row = {
                "experiment_id": run_dir.name,
                "algorithm": "PPO",
                "evaluator": cfg.evaluator,
                "seed": cfg.seed,
                "episode_id": ep,
                "step_id": step,
                "global_step": global_step,
                **policy_extra,
                "reward": float(reward),
                "return_G_t": np.nan,
                "advantage_A_t": np.nan,
                "normalized_advantage": np.nan,
                "td_error_delta": np.nan,
            }

            episode_policy_rows.append(policy_row)

            total_reward += reward
            penalties += float(info.get("penalty_total", 0.0))
            aero_wall_time_sec += float(info.get("aero_wall_time_step_sec", 0.0))
            action_norms.append(norm)
            best_clcd = max(best_clcd, info["CL_CD"])

            cl_cd_value = _safe_float_value(info.get("CL_CD", -np.inf), -np.inf)
            cd_value = _safe_float_value(info.get("CD", np.inf), np.inf)

            numeric_ok = (
                np.isfinite(cl_cd_value)
                and np.isfinite(cd_value)
                and cd_value > 0.0
            )

            solver_ok = str(info.get("solver_status", "ok")) in {"ok", "valid", ""}

            is_step_feasible = (
                _is_feasible_info(info)
                and numeric_ok
            )

            is_step_strict_safe = (
                is_step_feasible
                and _is_local_margin_safe_info(info, cfg)
            )

            if is_step_feasible:
                if (
                    best_record is None
                    or cl_cd_value > float(best_record.get("CL_CD", -np.inf))
                ):
                    best_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }

            if is_step_strict_safe:
                if (
                    best_strict_safe_record is None
                    or cl_cd_value > float(best_strict_safe_record.get("CL_CD", -np.inf))
                ):
                    best_strict_safe_record = {
                        "episode_id": ep,
                        "step_id": step,
                        "cst": info["next_cst"].copy(),
                        **info,
                    }

            obs = nxt
            step += 1
            global_step += 1
            last_info = info

        # Episode bittikten sonra PPO return/advantage hesapla.
        returns = []
        G = 0.0
        gamma = float(cfg.ppo.gamma)

        for pr in reversed(episode_policy_rows):
            G = float(pr["reward"]) + gamma * G
            returns.append(G)

        returns = list(reversed(returns))
        values = np.array(
            [float(pr.get("value_V_s", 0.0)) for pr in episode_policy_rows],
            dtype=np.float32,
        )

        advantages = np.array(returns, dtype=np.float32) - values

        if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
            normalized_advantages = (
                advantages - float(np.mean(advantages))
            ) / (float(np.std(advantages)) + 1e-8)
        else:
            normalized_advantages = np.zeros_like(advantages)

        for idx, pr in enumerate(episode_policy_rows):
            pr["return_G_t"] = float(returns[idx])
            pr["advantage_A_t"] = float(advantages[idx])
            pr["normalized_advantage"] = float(normalized_advantages[idx])

            if idx < len(episode_policy_rows) - 1:
                next_v = float(episode_policy_rows[idx + 1].get("value_V_s", 0.0))
                pr["td_error_delta"] = (
                    float(pr["reward"])
                    + gamma * next_v
                    - float(pr.get("value_V_s", 0.0))
                )
            else:
                pr["td_error_delta"] = (
                    float(pr["reward"])
                    - float(pr.get("value_V_s", 0.0))
                )

            policy.log(pr)

        final_is_cm_feasible = _as_bool(last_info.get("is_CM_feasible", False))
        final_is_tc_feasible = _as_bool(last_info.get("is_tc_feasible", False))
        final_is_geometry_valid = _as_bool(last_info.get("is_geometry_valid", False))

        final_is_feasible = _is_feasible_info(last_info)
        final_is_strict_safe = _is_strict_safe_info(last_info, cfg)

        cm_violation = int(not final_is_cm_feasible)
        tc_violation = int(not final_is_tc_feasible)
        invalid_geometry = int(not final_is_geometry_valid)
        solver_error = int(not _is_solver_success_info(last_info))

        done_reason = str(last_info.get("done_reason", ""))

        if done_reason == "max_episode_steps" and final_is_feasible:
            done_reason = "max_episode_steps_feasible"

        summaries.append(
            {
                "experiment_id": run_dir.name,
                "algorithm": "PPO",
                "seed": cfg.seed,
                "episode_id": ep,
                "initial_CL_CD": initial_clcd,
                "final_CL_CD": last_info["CL_CD"],
                "best_CL_CD": best_clcd,
                "final_CL": last_info["CL"],
                "final_CD": last_info["CD"],
                "final_CM": last_info["CM"],
                "final_t_c": last_info["t_c"],
                "total_reward": total_reward,
                "total_penalty": penalties,
                "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
                "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
                "constraint_violation_count": int(cm_violation + tc_violation),
                "CM_violation_count": cm_violation,
                "tc_violation_count": tc_violation,
                "invalid_geometry_count": invalid_geometry,
                "solver_error_count": solver_error,
                "done_reason": done_reason,
                "episode_length": step,
                "is_final_feasible": final_is_feasible,
                "is_final_strict_safe": final_is_strict_safe,
                "final_min_local_thickness": last_info.get("min_local_thickness", np.nan),
                "final_local_thickness_violation": last_info.get("local_thickness_violation", np.nan),
            }
        )

    rollout.close()
    policy.close()

    pd.DataFrame(summaries).to_csv(run_dir / "episode_summary.csv", index=False)

    # AoA sweep: sadece mevcut evaluator ile yapılır.
    sweep_rows = []

    if best_record is not None:
        aoa_values = [float(x) for x in aoa_sweep.split(",") if x.strip()]
        cst = best_record["cst"]
        evaltor = _make_evaluator(cfg)

        for a in aoa_values:
            t0 = time.time()
            out = evaltor.evaluate(cst, a, cfg.re)
            runtime_ms = (time.time() - t0) * 1000.0
            aero_wall_time_sec += runtime_ms / 1000.0

            ratio = out.cl / max(out.cd, cfg.cd_lower_bound)
            geom = out.geometry_features or {}

            min_local_thickness = float(geom.get("min_local_thickness", 0.0))
            min_required = float(cfg.geometry.min_local_thickness_required)
            local_violation = max(0.0, min_required - min_local_thickness)

            sweep_rows.append(
                {
                    "algorithm": "PPO",
                    "evaluator": cfg.evaluator,
                    "episode_id": best_record["episode_id"],
                    "step_id": best_record["step_id"],
                    "AoA": a,
                    "Re": cfg.re,
                    **{f"CST_u{i+1}": float(cst[i]) for i in range(4)},
                    **{f"CST_l{i+1}": float(cst[4+i]) for i in range(4)},
                    "CL_pred": out.cl,
                    "CD_pred": out.cd,
                    "CM_pred": out.cm,
                    "CL_CD_pred": ratio,
                    "t_c": out.tc,
                    "is_geometry_valid": out.is_geometry_valid,
                    "min_local_thickness": geom.get("min_local_thickness", ""),
                    "max_thickness": geom.get("max_thickness", ""),
                    "local_thickness_violation": local_violation,
                    "solver_status": out.solver_status,
                    "solver_error_message": out.solver_error_message,
                    "xfoil_converged": True if cfg.evaluator == "xfoil" else "",
                    "xfoil_iterations": "",
                    "xfoil_error_message": "",
                    "runtime_ms": runtime_ms,
                }
            )

    sweep_filename = f"{cfg.evaluator.lower()}_aoa_sweep_logs.csv"
    pd.DataFrame(sweep_rows).to_csv(run_dir / sweep_filename, index=False)

    summary_df = pd.DataFrame(summaries)

    if len(summary_df) > 0:
        final_mean_CL_CD = pd.to_numeric(
            summary_df.get("final_CL_CD", np.nan),
            errors="coerce",
        ).mean()

        final_feasible_episode_count = int(
            summary_df.get(
                "is_final_feasible",
                pd.Series(False, index=summary_df.index),
            )
            .map(_as_bool)
            .sum()
        )
        solver_error_episode_count = int(
            pd.to_numeric(
                summary_df.get("solver_error_count", 0),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        final_strict_safe_episode_count = int(
            summary_df.get(
                "is_final_strict_safe",
                pd.Series(False, index=summary_df.index),
            )
            .map(_as_bool)
            .sum()
        )

        done_reason_series = summary_df.get(
            "done_reason",
            pd.Series("", index=summary_df.index),
        ).astype(str)

        invalid_geometry_episode_count = int(
            done_reason_series.str.startswith("invalid_geometry").sum()
        )

        negative_local_thickness_episode_count = int(
            (done_reason_series == "invalid_geometry_negative_local_thickness").sum()
        )
    else:
        final_mean_CL_CD = np.nan
        final_feasible_episode_count = 0
        invalid_geometry_episode_count = 0
        negative_local_thickness_episode_count = 0
        final_strict_safe_episode_count = 0
        final_strict_safe_episode_count = 0
    eval_metrics = {
        "algorithm": "PPO",
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "action_scale": cfg.action_scale,
        "w_action": cfg.reward_weights.w_action,
        "AoA": cfg.aoa,
        "Re": cfg.re,

        "aero_wall_time_sec": aero_wall_time_sec,
        "eval_wall_time_sec": time.time() - eval_start,
        "episodes": episodes,

        "rl_checkpoint_path": str(rl_checkpoint_path),
        "rl_checkpoint_filename": rl_checkpoint_path.name,

        "final_mean_CL_CD": final_mean_CL_CD,
        "final_feasible_episode_count": final_feasible_episode_count,
        "invalid_geometry_episode_count": invalid_geometry_episode_count,
        "negative_local_thickness_episode_count": negative_local_thickness_episode_count,

        "best_feasible_CL": float(best_record["CL"]) if best_record is not None else np.nan,
        "best_feasible_CD": float(best_record["CD"]) if best_record is not None else np.nan,
        "best_feasible_CL_CD": float(best_record["CL_CD"]) if best_record is not None else np.nan,
        "best_feasible_CM": float(best_record["CM"]) if best_record is not None else np.nan,
        "best_feasible_t_c": float(best_record["t_c"]) if best_record is not None else np.nan,
        "best_feasible_episode_id": int(best_record["episode_id"]) if best_record is not None else -1,
        "best_feasible_step_id": int(best_record["step_id"]) if best_record is not None else -1,
        "has_feasible_design": bool(best_record is not None),
        "final_strict_safe_episode_count": final_strict_safe_episode_count,
        "best_strict_safe_CL": (
            float(best_strict_safe_record["CL"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CD": (
            float(best_strict_safe_record["CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CL_CD": (
            float(best_strict_safe_record["CL_CD"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_CM": (
            float(best_strict_safe_record["CM"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_t_c": (
            float(best_strict_safe_record["t_c"])
            if best_strict_safe_record is not None
            else np.nan
        ),
        "best_strict_safe_episode_id": (
            int(best_strict_safe_record["episode_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "best_strict_safe_step_id": (
            int(best_strict_safe_record["step_id"])
            if best_strict_safe_record is not None
            else -1
        ),
        "has_strict_safe_design": bool(best_strict_safe_record is not None),
        "solver_error_episode_count": solver_error_episode_count,
    }

    pd.DataFrame([eval_metrics]).to_csv(run_dir / "eval_metrics.csv", index=False)

    with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_wall_time_sec": eval_metrics["eval_wall_time_sec"],
                "aero_wall_time_sec": aero_wall_time_sec,
                "episodes": episodes,
                "aoa_sweep": aoa_sweep,
            },
            f,
            indent=2,
        )

    with open(run_dir / "xai_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "xai_v1",
                "run_type": "evaluate",
                "algorithm": "PPO",
                "evaluator": cfg.evaluator,
                "source_checkpoint": cfg.rl_checkpoint_path,
                "files": [
                    "rollout_step_logs.csv",
                    "policy_outputs.csv",
                    "episode_summary.csv",
                    sweep_filename,
                    "eval_metrics.csv",
                    "eval_summary.json",
                ],
            },
            f,
            indent=2,
        )
