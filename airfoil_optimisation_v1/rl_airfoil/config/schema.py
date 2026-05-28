from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Dict, Tuple

import json
import subprocess
import time


@dataclass
class Constraints:
    # Paper ile uyumlu fiziksel kısıtlar
    cm_bounds: Tuple[float, float] = (-0.12, 0.02)
    tc_bounds: Tuple[float, float] = (0.08, 0.18)


@dataclass
class RewardWeights:
    # Objective term: log-scaled CL/CD
    w1: float = 1.0

    # CM constraint penalty weight
    w2: float = 10.0

    # t/c constraint penalty weight
    w3: float = 10.0

    # Local thickness penalty weight.
    # env.py içinde local thickness violation normalize edildiği için
    # bu değer artık daha anlamlı ve güçlü etki eder.
    w_local_thickness: float = 20.0

    # TD3/SAC gibi continuous-control algoritmalarında
    # actor'ın action saturation yapmasını azaltmak için kullanılır.
    #
    # action vektörü [-1, 1]^8 aralığındadır.
    # max sum(action^2) = 8
    # w_action = 0.05 ise maksimum action penalty = 0.40
    #
    # Bu değeri sabit bırakıyoruz.
    w_action: float = 0.05


@dataclass
class TD3Hyperparameters:
    # TD3 eğitim hiperparametreleri
    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2

    # Bu değerler sabit bırakılıyor.
    target_policy_noise: float = 0.03
    target_noise_clip: float = 0.07
    action_noise_sigma: float = 0.02

@dataclass
class SACHyperparameters:
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # SAC'in exploration mekanizması policy entropy üzerinden gelir.
    # "auto" alpha değerinin otomatik öğrenilmesini sağlar.
    ent_coef: str = "auto"
    target_entropy: str = "auto"

@dataclass
class PPOHyperparameters:
    # Paper ile uyumlu PPO ayarları:
    # learning rate lineer olarak 3e-4 -> 3e-5 azalacak.
    learning_rate_start: float = 3e-4
    learning_rate_end: float = 3e-5

    # PPO rollout/update ayarları
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10

    # PPO objective/GAE ayarları
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    # Loss ağırlıkları
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    normalize_advantage: bool = True

@dataclass
class GeometryConfig:
    # CST yüzeyi ve geometri özellikleri için örnekleme çözünürlüğü
    n_points: int = 201

    # Local thickness güvenlik eşiği.
    # min_local_thickness bu değerin altına düşerse penalty oluşur.
    # env.py içinde bu violation normalize edildiği için 0.005 daha güvenli bir eşiktir.
    min_local_thickness_required: float = 0.005

    # Çok uç yüzey değerlerini yakalamak için güvenlik sınırı.
    max_abs_surface_y: float = 0.75


@dataclass
class XFOILConfig:
    # XFOIL executable path.
    # Windows için örnek: C:/xfoil/xfoil.exe
    # Boş bırakılırsa sistem PATH veya XFOIL_PATH environment variable kullanılabilir.
    executable_path: str = "xfoil"

    # Her XFOIL çağrısı için maksimum bekleme süresi.
    timeout_sec: float = 30.0

    # XFOIL viscous solver iteration sayısı.
    max_iter: int = 100

    # Panel sayısı.
    ppar_n: int = 160

    # Düşük hız / incompressible analiz için 0.0 kalabilir.
    mach: float = 0.0

    # Boundary layer transition ayarı.
    ncrit: float = 9.0

    # Forced transition yoksa 1.0 / 1.0 bırakılır.
    xtr_top: float = 1.0
    xtr_bottom: float = 1.0

    # CST profil koordinat üretim çözünürlüğü.
    n_points: int = 201


@dataclass
class ExperimentConfig:
    algorithm: str = "td3"
    evaluator: str = "surrogate"

    surrogate_model_name: str = "S-1D"
    surrogate_checkpoint_path: str = "checkpoints/surrogate_s1d.pt"
    scaler_json_path: str = "checkpoints/scalers.json"

    # RL model artık checkpoints/ içine değil,
    # her eğitim run klasörünün içine kaydedilecek.
    # Bu nedenle default boş bırakılır.
    rl_checkpoint_path: str = ""

    seed: int = 42
    total_timesteps: int = 200_000
    episode_max_steps: int = 25

    action_range: Tuple[float, float] = (-1.0, 1.0)

    # Bu değer sabit bırakılıyor.
    action_scale: float = 0.003

    cst_bounds: Tuple[float, float] = (-0.35, 0.35)

    # Güvenli başlangıç airfoil'i.
    # Paper'daki CST8 uzayı ile uyumludur: 4 üst + 4 alt CST katsayısı.
    initial_cst: Tuple[float, float, float, float, float, float, float, float] = (
        0.20, 0.18, 0.14, 0.10,
        -0.12, -0.10, -0.08, -0.05,
    )

    initial_cst_noise_std: float = 0.005

    aoa: float = 2.0
    re: float = 1e6

    # CL/CD hesabında CD'nin sıfıra yaklaşması durumunda numerical stability için.
    cd_lower_bound: float = 1e-6

    # Invalid geometry veya solver error durumunda uygulanacak büyük cezalar.
    invalid_geometry_penalty: float = 100.0
    solver_error_penalty: float = 100.0

    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    constraints: Constraints = field(default_factory=Constraints)
    td3: TD3Hyperparameters = field(default_factory=TD3Hyperparameters)
    sac: SACHyperparameters = field(default_factory=SACHyperparameters)
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    xfoil: XFOILConfig = field(default_factory=XFOILConfig)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def create_run_dir(base: Path, algorithm: str) -> Path:
    """
    Eğitim veya deney çıktıları için benzersiz run klasörü oluşturur.

    Normalde run_id saniye bazlıdır; ancak aynı saniye içinde birden fazla
    deneme yapılırsa çakışma olmaması için sonuna sayaç eklenir.
    """
    base = Path(base)
    algorithm_dir = base / algorithm.lower()
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    run_dir = algorithm_dir / f"run_{timestamp}"

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    suffix = 1
    while True:
        candidate = algorithm_dir / f"run_{timestamp}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def write_experiment_metadata(
    cfg: ExperimentConfig,
    run_dir: Path,
    normalization_stats: Dict,
) -> None:
    """
    Deney metadata dosyasını yazar.

    XAI analizleri için kritik olan model yolu, reward ağırlıkları,
    constraint sınırları, TD3 hiperparametreleri ve geometri ayarları burada tutulur.
    """
    run_dir = Path(run_dir)

    metadata = {
        "experiment_id": run_dir.name,
        "algorithm": cfg.algorithm.upper(),
        "evaluator": cfg.evaluator,
        "surrogate_model_name": cfg.surrogate_model_name,
        "surrogate_checkpoint_path": cfg.surrogate_checkpoint_path,
        "scaler_json_path": cfg.scaler_json_path,
        "rl_checkpoint_path": cfg.rl_checkpoint_path,
        "rl_checkpoint_filename": (
            Path(cfg.rl_checkpoint_path).name
            if str(cfg.rl_checkpoint_path).strip()
            else ""
        ),

        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "episode_max_steps": cfg.episode_max_steps,

        "action_range": list(cfg.action_range),
        "action_scale": cfg.action_scale,
        "cst_bounds": list(cfg.cst_bounds),
        "initial_cst": list(cfg.initial_cst),
        "initial_cst_noise_std": cfg.initial_cst_noise_std,

        "AoA": cfg.aoa,
        "Re": cfg.re,
        "log10_Re": None if cfg.re <= 0 else float(__import__("math").log10(cfg.re)),

        "reward_weights": asdict(cfg.reward_weights),

        "constraints": {
            "CM": list(cfg.constraints.cm_bounds),
            "t/c": list(cfg.constraints.tc_bounds),
        },

        "cd_lower_bound": cfg.cd_lower_bound,
        "invalid_geometry_penalty": cfg.invalid_geometry_penalty,
        "solver_error_penalty": cfg.solver_error_penalty,

        "td3_hyperparameters": asdict(cfg.td3),
        "sac_hyperparameters": asdict(cfg.sac),
        "ppo_hyperparameters": asdict(cfg.ppo),
        "geometry_config": asdict(cfg.geometry),
        "xfoil_config": asdict(cfg.xfoil),

        "normalization_stats": normalization_stats,
        "code_commit_hash": git_commit_hash(),
    }

    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def _filter_dataclass_kwargs(cls, raw: Dict) -> Dict:
    """
    Metadata içinden sadece ilgili dataclass'ın kabul ettiği alanları seçer.
    Böylece metadata'ya ileride yeni alan eklenirse kod kırılmaz.
    """
    if not isinstance(raw, dict):
        return {}

    valid_keys = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid_keys}


def _tuple_float(value, fallback):
    """
    Metadata'dan gelen liste/tuple değerleri güvenli şekilde tuple[float, ...] yapar.
    """
    if value is None:
        return fallback

    try:
        return tuple(float(x) for x in value)
    except Exception:
        return fallback


def _tuple_int_or_float(value, fallback):
    """
    action_range, cst_bounds gibi tuple alanları için güvenli dönüştürücü.
    """
    if value is None:
        return fallback

    try:
        return tuple(value)
    except Exception:
        return fallback


def load_experiment_config_from_metadata(
    run_dir: Path,
    fallback_cfg: ExperimentConfig | None = None,
) -> ExperimentConfig:
    """
    Eğitim run klasöründeki experiment_metadata.json dosyasından
    evaluation için kullanılacak ExperimentConfig nesnesini yeniden kurar.

    Amaç:
    Evaluation sırasında training ile aynı environment/config değerlerini kullanmak.

    Bu fonksiyon özellikle şu hataları engeller:
      - training action_scale = 0.003 iken evaluation action_scale = 0.02 çalışması
      - training w_action = 0.05 iken evaluation w_action = 0.01 yazılması
      - yanlış surrogate_model_name veya checkpoint ile evaluation yapılması
      - farklı AoA/Re/constraint değerleriyle karşılaştırma yapılması
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "experiment_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"experiment_metadata.json bulunamadı: {metadata_path}\n"
            "Evaluation için --run-dir olarak train run klasörünü vermelisiniz. "
            "Örnek: .\\logs\\sac\\run_1779022429"
        )

    if fallback_cfg is None:
        fallback_cfg = ExperimentConfig()

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    reward_weights = RewardWeights(
        **_filter_dataclass_kwargs(
            RewardWeights,
            meta.get("reward_weights", {}),
        )
    )

    constraints_raw = meta.get("constraints", {})
    cm_bounds = _tuple_float(
        constraints_raw.get("CM"),
        fallback_cfg.constraints.cm_bounds,
    )
    tc_bounds = _tuple_float(
        constraints_raw.get("t/c"),
        fallback_cfg.constraints.tc_bounds,
    )

    constraints = Constraints(
        cm_bounds=cm_bounds,
        tc_bounds=tc_bounds,
    )

    td3 = TD3Hyperparameters(
        **_filter_dataclass_kwargs(
            TD3Hyperparameters,
            meta.get("td3_hyperparameters", {}),
        )
    )

    sac = SACHyperparameters(
        **_filter_dataclass_kwargs(
            SACHyperparameters,
            meta.get("sac_hyperparameters", {}),
        )
    )

    ppo = PPOHyperparameters(
        **_filter_dataclass_kwargs(
            PPOHyperparameters,
            meta.get("ppo_hyperparameters", {}),
        )
    )

    geometry = GeometryConfig(
        **_filter_dataclass_kwargs(
            GeometryConfig,
            meta.get("geometry_config", {}),
        )
    )

    xfoil = XFOILConfig(
        **_filter_dataclass_kwargs(
            XFOILConfig,
            meta.get("xfoil_config", {}),
        )
    )

    cfg = ExperimentConfig(
        algorithm=str(meta.get("algorithm", fallback_cfg.algorithm)).lower(),
        evaluator=str(meta.get("evaluator", fallback_cfg.evaluator)).lower(),

        surrogate_model_name=str(
            meta.get("surrogate_model_name", fallback_cfg.surrogate_model_name)
        ),
        surrogate_checkpoint_path=str(
            meta.get(
                "surrogate_checkpoint_path",
                fallback_cfg.surrogate_checkpoint_path,
            )
        ),
        scaler_json_path=str(
            meta.get("scaler_json_path", fallback_cfg.scaler_json_path)
        ),

        rl_checkpoint_path=str(
            meta.get("rl_checkpoint_path", fallback_cfg.rl_checkpoint_path)
        ),

        seed=int(meta.get("seed", fallback_cfg.seed)),
        total_timesteps=int(
            meta.get("total_timesteps", fallback_cfg.total_timesteps)
        ),
        episode_max_steps=int(
            meta.get("episode_max_steps", fallback_cfg.episode_max_steps)
        ),

        action_range=_tuple_int_or_float(
            meta.get("action_range"),
            fallback_cfg.action_range,
        ),
        action_scale=float(
            meta.get("action_scale", fallback_cfg.action_scale)
        ),
        cst_bounds=_tuple_int_or_float(
            meta.get("cst_bounds"),
            fallback_cfg.cst_bounds,
        ),

        initial_cst=_tuple_float(
            meta.get("initial_cst"),
            fallback_cfg.initial_cst,
        ),
        initial_cst_noise_std=float(
            meta.get(
                "initial_cst_noise_std",
                fallback_cfg.initial_cst_noise_std,
            )
        ),

        aoa=float(meta.get("AoA", fallback_cfg.aoa)),
        re=float(meta.get("Re", fallback_cfg.re)),

        cd_lower_bound=float(
            meta.get("cd_lower_bound", fallback_cfg.cd_lower_bound)
        ),
        invalid_geometry_penalty=float(
            meta.get(
                "invalid_geometry_penalty",
                fallback_cfg.invalid_geometry_penalty,
            )
        ),
        solver_error_penalty=float(
            meta.get(
                "solver_error_penalty",
                fallback_cfg.solver_error_penalty,
            )
        ),

        reward_weights=reward_weights,
        constraints=constraints,
        td3=td3,
        sac=sac,
        ppo=ppo,
        geometry=geometry,
        xfoil=xfoil,
    )

    return cfg