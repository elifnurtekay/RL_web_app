from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from stable_baselines3 import PPO, SAC, TD3

from .solver_interface import AerodynamicOptimizer, OptimizationInput

from rl_airfoil.config.schema import (
    Constraints,
    ExperimentConfig,
    GeometryConfig,
    PPOHyperparameters,
    RewardWeights,
    SACHyperparameters,
    TD3Hyperparameters,
    XFOILConfig,
)
from rl_airfoil.core.env import AirfoilEnv
from rl_airfoil.evaluators.surrogate import SurrogateEvaluator
from rl_airfoil.evaluators.xfoil import XFOILEvaluator
from rl_airfoil.geometry.cst import cst_surface, compute_cst_geometry_features


class RealRLModelOptimizer(AerodynamicOptimizer):
    """
    Django arayüzü için gerçek PPO / TD3 / SAC inference katmanı.

    Bu sınıf run klasörü kullanmaz.
    Her model için:
    - .zip policy checkpoint dosyasını yükler.
    - metadata JSON dosyasından eğitim parametrelerini okur.
    - surrogate_s1d.pt + scalers.json ile aerodinamik çıktıları hesaplar.
    - AirfoilEnv üzerinde deterministic policy rollout yapar.
    """

    MODEL_CLASSES = {
        "PPO": PPO,
        "TD3": TD3,
        "SAC": SAC,
    }

    _MODEL_CACHE = {}

    def optimize(self, optimization_input: OptimizationInput) -> dict:
        algorithm = str(optimization_input.model).upper().strip()

        if algorithm not in self.MODEL_CLASSES:
            raise ValueError("Unsupported model. Use PPO, TD3, or SAC.")

        cfg = self._build_config_from_metadata_json(
            algorithm=algorithm,
            optimization_input=optimization_input,
        )

        evaluator = self._make_evaluator(cfg)
        env = AirfoilEnv(cfg, evaluator)

        model = self._load_model(
            algorithm=algorithm,
            checkpoint_path=cfg.rl_checkpoint_path,
        )

        rollout_start = time.time()

        obs, _ = env.reset(seed=cfg.seed)
        initial_cst = np.asarray(env.cst, dtype=np.float32).reshape(8)

        records = []
        best_feasible_record = None
        best_any_record = None

        for step_id in range(int(cfg.episode_max_steps)):
            action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            next_cst = np.asarray(
                info.get("next_cst", env.cst),
                dtype=np.float32,
            ).reshape(8)

            record = {
                "step_id": int(step_id),
                "cst": next_cst.copy(),
                "action": np.asarray(action, dtype=np.float32).reshape(8),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": bool(done),
                "info": dict(info),
                "is_feasible": self._is_feasible_info(info),
                "is_strict_safe": self._is_strict_safe_info(info, cfg),
            }

            records.append(record)

            cl_cd_value = self._safe_float(
                info.get("CL_CD"),
                default=-math.inf,
            )

            if best_any_record is None:
                best_any_record = record
            else:
                best_any_cl_cd = self._safe_float(
                    best_any_record["info"].get("CL_CD"),
                    default=-math.inf,
                )

                if cl_cd_value > best_any_cl_cd:
                    best_any_record = record

            if record["is_feasible"]:
                if best_feasible_record is None:
                    best_feasible_record = record
                else:
                    best_feasible_cl_cd = self._safe_float(
                        best_feasible_record["info"].get("CL_CD"),
                        default=-math.inf,
                    )

                    if cl_cd_value > best_feasible_cl_cd:
                        best_feasible_record = record

            obs = next_obs

            if done:
                break

        if not records:
            raise RuntimeError("Policy rollout produced no step.")

        selected_record = best_feasible_record or best_any_record or records[-1]
        optimized_cst = selected_record["cst"]

        rollout_wall_time_sec = time.time() - rollout_start

        return self._format_result(
            algorithm=algorithm,
            cfg=cfg,
            initial_cst=initial_cst,
            optimized_cst=optimized_cst,
            selected_record=selected_record,
            records=records,
            rollout_wall_time_sec=rollout_wall_time_sec,
        )

    def _build_config_from_metadata_json(
        self,
        algorithm: str,
        optimization_input: OptimizationInput,
    ) -> ExperimentConfig:
        artifacts = getattr(settings, "RL_MODEL_ARTIFACTS", {})

        if algorithm not in artifacts:
            raise FileNotFoundError(
                f"settings.RL_MODEL_ARTIFACTS içinde {algorithm} tanımlı değil."
            )

        model_path = Path(artifacts[algorithm]["model_path"]).expanduser().resolve()
        metadata_path = Path(artifacts[algorithm]["metadata_path"]).expanduser().resolve()

        surrogate_path = Path(settings.RL_SURROGATE_CHECKPOINT_PATH).expanduser().resolve()
        scaler_path = Path(settings.RL_SCALER_JSON_PATH).expanduser().resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"{algorithm} model dosyası bulunamadı: {model_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"{algorithm} metadata JSON bulunamadı: {metadata_path}")

        if not surrogate_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint bulunamadı: {surrogate_path}")

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler JSON bulunamadı: {scaler_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        metadata_algorithm = str(meta.get("algorithm", algorithm)).upper()

        if metadata_algorithm != algorithm:
            raise ValueError(
                f"Seçilen model {algorithm}, fakat metadata içinde {metadata_algorithm} yazıyor. "
                f"Yanlış metadata dosyası seçilmiş olabilir: {metadata_path}"
            )

        cst = self._input_to_cst8(
            optimization_input=optimization_input,
            meta=meta,
        )

        cfg = ExperimentConfig()

        cfg.algorithm = algorithm.lower()
        cfg.evaluator = str(
            getattr(settings, "RL_WEB_EVALUATOR", meta.get("evaluator", "surrogate"))
        ).lower()

        cfg.surrogate_model_name = str(meta.get("surrogate_model_name", "S-3D"))
        cfg.surrogate_checkpoint_path = str(surrogate_path)
        cfg.scaler_json_path = str(scaler_path)
        cfg.rl_checkpoint_path = str(model_path)

        cfg.seed = int(meta.get("seed", 42))
        cfg.total_timesteps = int(meta.get("total_timesteps", cfg.total_timesteps))
        cfg.episode_max_steps = int(meta.get("episode_max_steps", cfg.episode_max_steps))

        cfg.action_scale = float(meta.get("action_scale", cfg.action_scale))
        cfg.action_range = tuple(meta.get("action_range", cfg.action_range))
        cfg.cst_bounds = tuple(meta.get("cst_bounds", cfg.cst_bounds))

        cfg.initial_cst = tuple(float(x) for x in cst)

        # Kullanıcının arayüzde verdiği CST başlangıcı birebir kullanılsın.
        # Random başlangıç gürültüsü kapatılır.
        cfg.initial_cst_noise_std = 0.0

        # AoA ve Reynolds arayüzden alınır.
        cfg.aoa = float(optimization_input.aoa)
        cfg.re = float(optimization_input.reynolds)

        cfg.cd_lower_bound = float(meta.get("cd_lower_bound", cfg.cd_lower_bound))
        cfg.invalid_geometry_penalty = float(
            meta.get("invalid_geometry_penalty", cfg.invalid_geometry_penalty)
        )
        cfg.solver_error_penalty = float(
            meta.get("solver_error_penalty", cfg.solver_error_penalty)
        )

        reward_raw = meta.get("reward_weights", {})

        cfg.reward_weights = RewardWeights(
            w1=float(reward_raw.get("w1", cfg.reward_weights.w1)),
            w2=float(reward_raw.get("w2", cfg.reward_weights.w2)),
            w3=float(reward_raw.get("w3", cfg.reward_weights.w3)),
            w_local_thickness=float(
                reward_raw.get(
                    "w_local_thickness",
                    cfg.reward_weights.w_local_thickness,
                )
            ),
            w_action=float(reward_raw.get("w_action", cfg.reward_weights.w_action)),
        )

        constraints_raw = meta.get("constraints", {})
        cm_bounds = constraints_raw.get("CM", cfg.constraints.cm_bounds)
        tc_bounds = constraints_raw.get("t/c", cfg.constraints.tc_bounds)

        cfg.constraints = Constraints(
            cm_bounds=tuple(float(x) for x in cm_bounds),
            tc_bounds=tuple(float(x) for x in tc_bounds),
        )

        geometry_raw = meta.get("geometry_config", {})

        cfg.geometry = GeometryConfig(
            n_points=int(geometry_raw.get("n_points", cfg.geometry.n_points)),
            min_local_thickness_required=float(
                geometry_raw.get(
                    "min_local_thickness_required",
                    cfg.geometry.min_local_thickness_required,
                )
            ),
            max_abs_surface_y=float(
                geometry_raw.get(
                    "max_abs_surface_y",
                    cfg.geometry.max_abs_surface_y,
                )
            ),
        )

        xfoil_raw = meta.get("xfoil_config", {})

        cfg.xfoil = XFOILConfig(
            executable_path=str(
                xfoil_raw.get("executable_path", cfg.xfoil.executable_path)
            ),
            timeout_sec=float(xfoil_raw.get("timeout_sec", cfg.xfoil.timeout_sec)),
            max_iter=int(xfoil_raw.get("max_iter", cfg.xfoil.max_iter)),
            ppar_n=int(xfoil_raw.get("ppar_n", cfg.xfoil.ppar_n)),
            mach=float(xfoil_raw.get("mach", cfg.xfoil.mach)),
            ncrit=float(xfoil_raw.get("ncrit", cfg.xfoil.ncrit)),
            xtr_top=float(xfoil_raw.get("xtr_top", cfg.xfoil.xtr_top)),
            xtr_bottom=float(xfoil_raw.get("xtr_bottom", cfg.xfoil.xtr_bottom)),
            n_points=int(xfoil_raw.get("n_points", cfg.xfoil.n_points)),
        )

        td3_raw = meta.get("td3_hyperparameters", {})

        cfg.td3 = TD3Hyperparameters(
            learning_rate=float(td3_raw.get("learning_rate", cfg.td3.learning_rate)),
            buffer_size=int(td3_raw.get("buffer_size", cfg.td3.buffer_size)),
            learning_starts=int(td3_raw.get("learning_starts", cfg.td3.learning_starts)),
            batch_size=int(td3_raw.get("batch_size", cfg.td3.batch_size)),
            tau=float(td3_raw.get("tau", cfg.td3.tau)),
            gamma=float(td3_raw.get("gamma", cfg.td3.gamma)),
            train_freq=int(td3_raw.get("train_freq", cfg.td3.train_freq)),
            gradient_steps=int(td3_raw.get("gradient_steps", cfg.td3.gradient_steps)),
            policy_delay=int(td3_raw.get("policy_delay", cfg.td3.policy_delay)),
            target_policy_noise=float(
                td3_raw.get("target_policy_noise", cfg.td3.target_policy_noise)
            ),
            target_noise_clip=float(
                td3_raw.get("target_noise_clip", cfg.td3.target_noise_clip)
            ),
            action_noise_sigma=float(
                td3_raw.get("action_noise_sigma", cfg.td3.action_noise_sigma)
            ),
        )

        sac_raw = meta.get("sac_hyperparameters", {})

        cfg.sac = SACHyperparameters(
            learning_rate=float(sac_raw.get("learning_rate", cfg.sac.learning_rate)),
            buffer_size=int(sac_raw.get("buffer_size", cfg.sac.buffer_size)),
            learning_starts=int(sac_raw.get("learning_starts", cfg.sac.learning_starts)),
            batch_size=int(sac_raw.get("batch_size", cfg.sac.batch_size)),
            tau=float(sac_raw.get("tau", cfg.sac.tau)),
            gamma=float(sac_raw.get("gamma", cfg.sac.gamma)),
            train_freq=int(sac_raw.get("train_freq", cfg.sac.train_freq)),
            gradient_steps=int(sac_raw.get("gradient_steps", cfg.sac.gradient_steps)),
            ent_coef=sac_raw.get("ent_coef", cfg.sac.ent_coef),
            target_entropy=sac_raw.get("target_entropy", cfg.sac.target_entropy),
        )

        ppo_raw = meta.get("ppo_hyperparameters", {})

        cfg.ppo = PPOHyperparameters(
            learning_rate_start=float(
                ppo_raw.get("learning_rate_start", cfg.ppo.learning_rate_start)
            ),
            learning_rate_end=float(
                ppo_raw.get("learning_rate_end", cfg.ppo.learning_rate_end)
            ),
            n_steps=int(ppo_raw.get("n_steps", cfg.ppo.n_steps)),
            batch_size=int(ppo_raw.get("batch_size", cfg.ppo.batch_size)),
            n_epochs=int(ppo_raw.get("n_epochs", cfg.ppo.n_epochs)),
            gamma=float(ppo_raw.get("gamma", cfg.ppo.gamma)),
            gae_lambda=float(ppo_raw.get("gae_lambda", cfg.ppo.gae_lambda)),
            clip_range=float(ppo_raw.get("clip_range", cfg.ppo.clip_range)),
            ent_coef=float(ppo_raw.get("ent_coef", cfg.ppo.ent_coef)),
            vf_coef=float(ppo_raw.get("vf_coef", cfg.ppo.vf_coef)),
            max_grad_norm=float(ppo_raw.get("max_grad_norm", cfg.ppo.max_grad_norm)),
            normalize_advantage=bool(
                ppo_raw.get("normalize_advantage", cfg.ppo.normalize_advantage)
            ),
        )

        return cfg

    def _input_to_cst8(
        self,
        optimization_input: OptimizationInput,
        meta: dict,
    ) -> np.ndarray:
        upper = [float(x) for x in optimization_input.upper_weights]
        lower = [float(x) for x in optimization_input.lower_weights]

        if len(upper) != 4:
            raise ValueError(
                f"Upper Surface Weights tam olarak 4 değer içermeli. Gelen: {len(upper)}"
            )

        if len(lower) != 4:
            raise ValueError(
                f"Lower Surface Weights tam olarak 4 değer içermeli. Gelen: {len(lower)}"
            )

        cst = np.asarray(upper + lower, dtype=np.float32).reshape(8)

        cst_bounds = meta.get("cst_bounds", [-0.35, 0.35])
        cst_min = float(cst_bounds[0])
        cst_max = float(cst_bounds[1])

        return np.clip(cst, cst_min, cst_max).astype(np.float32)

    def _load_model(self, algorithm: str, checkpoint_path: str):
        path = Path(checkpoint_path).expanduser().resolve()

        cache_key = (
            algorithm,
            str(path),
            path.stat().st_mtime,
        )

        if cache_key in self._MODEL_CACHE:
            return self._MODEL_CACHE[cache_key]

        model_cls = self.MODEL_CLASSES[algorithm]
        model = model_cls.load(str(path), device="cpu")

        self._MODEL_CACHE.clear()
        self._MODEL_CACHE[cache_key] = model

        return model

    def _make_evaluator(self, cfg: ExperimentConfig):
        if cfg.evaluator == "surrogate":
            return SurrogateEvaluator(
                checkpoint_path=cfg.surrogate_checkpoint_path,
                model_name=cfg.surrogate_model_name,
                scaler_json_path=cfg.scaler_json_path,
                device="cpu",
            )

        if cfg.evaluator == "xfoil":
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

    def _safe_float(self, value, default=np.nan) -> float:
        try:
            if value is None:
                return float(default)

            if isinstance(value, str) and value.strip() == "":
                return float(default)

            return float(value)

        except Exception:
            return float(default)

    def _as_bool(self, value) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if value is None:
            return False

        return str(value).strip().lower() in {"true", "1", "yes"}

    def _is_solver_success_info(self, info: dict) -> bool:
        solver_status = str(info.get("solver_status", "")).strip().lower()

        if solver_status not in {"", "ok", "valid", "non_positive_cd_clipped"}:
            return False

        xfoil_converged = str(info.get("xfoil_converged", "")).strip()

        if xfoil_converged == "":
            return True

        return self._as_bool(xfoil_converged)

    def _is_feasible_info(self, info: dict) -> bool:
        return bool(
            self._as_bool(info.get("is_CM_feasible", False))
            and self._as_bool(info.get("is_tc_feasible", False))
            and self._as_bool(info.get("is_geometry_valid", False))
            and self._is_solver_success_info(info)
        )

    def _is_strict_safe_info(
        self,
        info: dict,
        cfg: ExperimentConfig,
        eps: float = 1e-8,
    ) -> bool:
        if not self._is_feasible_info(info):
            return False

        local_violation = self._safe_float(
            info.get("local_thickness_violation", 0.0),
            default=0.0,
        )

        min_local_thickness = self._safe_float(
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

    def _geometry_from_cst(self, cst: np.ndarray, n_points: int = 201) -> dict:
        cst = np.asarray(cst, dtype=np.float64).reshape(8)
        x = np.linspace(0.0, 1.0, int(n_points), dtype=np.float64)

        upper_y = cst_surface(cst[:4], x)
        lower_y = cst_surface(cst[4:], x)

        return {
            "upper": [[float(xi), float(yi)] for xi, yi in zip(x, upper_y)],
            "lower": [[float(xi), float(yi)] for xi, yi in zip(x, lower_y)],
        }

    def _format_result(
        self,
        algorithm: str,
        cfg: ExperimentConfig,
        initial_cst: np.ndarray,
        optimized_cst: np.ndarray,
        selected_record: dict,
        records: list[dict],
        rollout_wall_time_sec: float,
    ) -> dict:
        info = selected_record["info"]

        cl = self._safe_float(info.get("CL"), 0.0)
        cd = self._safe_float(info.get("CD"), 1.0)
        cm = self._safe_float(info.get("CM"), 0.0)
        tc = self._safe_float(info.get("t_c"), 0.0)

        cl_cd = self._safe_float(
            info.get("CL_CD"),
            cl / max(cd, float(cfg.cd_lower_bound)),
        )

        cm_min, cm_max = cfg.constraints.cm_bounds
        tc_min, tc_max = cfg.constraints.tc_bounds

        cm_ok = bool(float(cm_min) <= cm <= float(cm_max))
        tc_ok = bool(float(tc_min) <= tc <= float(tc_max))
        geometry_ok = self._as_bool(info.get("is_geometry_valid", False))
        solver_ok = self._is_solver_success_info(info)

        feasible = bool(cm_ok and tc_ok and geometry_ok and solver_ok)

        geom_features = compute_cst_geometry_features(
            optimized_cst,
            n_points=cfg.geometry.n_points,
            min_local_thickness_required=cfg.geometry.min_local_thickness_required,
            max_abs_surface_y=cfg.geometry.max_abs_surface_y,
        ).to_dict()

        action = np.asarray(selected_record["action"], dtype=np.float32).reshape(8)

        return {
            "status": "ok",
            "model": algorithm,
            "metrics": {
                "cl": round(float(cl), 4),
                "cd": round(float(cd), 6),
                "cl_cd": round(float(cl_cd), 2),
                "cm": round(float(cm), 4),
                "tc": round(float(tc), 4),
                "is_feasible": feasible,
                "is_strict_safe": bool(selected_record["is_strict_safe"]),
            },
            "constraints": {
                "cm": {
                    "value": float(cm),
                    "min": float(cm_min),
                    "max": float(cm_max),
                    "satisfied": cm_ok,
                },
                "tc": {
                    "value": float(tc),
                    "min": float(tc_min),
                    "max": float(tc_max),
                    "satisfied": tc_ok,
                },
            },
            "decision_logic": [
                f"{algorithm} trained policy checkpoint loaded",
                "Training parameters restored from metadata JSON",
                "User CST8 geometry used as initial airfoil state",
                "Deterministic policy rollout executed",
                "Best feasible CL/CD candidate selected" if feasible else "No fully feasible candidate found; best available candidate returned",
                f"CM constraint {'satisfied' if cm_ok else 'violated'}",
                f"t/c constraint {'satisfied' if tc_ok else 'violated'}",
            ],
            "pipeline": [
                f"{algorithm} model file loaded",
                "Metadata JSON loaded",
                "AirfoilEnv initialized with JSON parameters",
                "Surrogate aerodynamic evaluator initialized",
                "Policy generated deterministic CST actions",
                "Aerodynamic metrics computed",
                "Constraint verification completed",
                "Optimized geometry generated",
            ],
            "geometry": {
                "initial": self._geometry_from_cst(
                    initial_cst,
                    n_points=cfg.geometry.n_points,
                ),
                "optimized": self._geometry_from_cst(
                    optimized_cst,
                    n_points=cfg.geometry.n_points,
                ),
            },
            "rl_diagnostics": {
                "algorithm": algorithm,
                "model_path": str(cfg.rl_checkpoint_path),
                "surrogate_path": str(cfg.surrogate_checkpoint_path),
                "selected_step_id": int(selected_record["step_id"]),
                "rollout_steps": int(len(records)),
                "episode_max_steps": int(cfg.episode_max_steps),
                "action_scale": float(cfg.action_scale),
                "rollout_wall_time_sec": round(float(rollout_wall_time_sec), 4),
                "done_reason": str(info.get("done_reason", "")),
                "reward": float(selected_record["reward"]),
                "action_norm": float(np.linalg.norm(action)),
                "upper_action_norm": float(np.linalg.norm(action[:4])),
                "lower_action_norm": float(np.linalg.norm(action[4:])),
                "action_max_abs": float(np.max(np.abs(action))),
                "optimized_cst": [float(x) for x in optimized_cst],
                "geometry_features": {
                    key: (
                        float(value)
                        if isinstance(value, (int, float, np.floating))
                        else bool(value)
                    )
                    for key, value in geom_features.items()
                },
            },
        }