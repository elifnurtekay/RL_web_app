from __future__ import annotations
import argparse
from pathlib import Path
from rl_airfoil.config.schema import (
    ExperimentConfig,
    TD3Hyperparameters,
    SACHyperparameters,
    PPOHyperparameters,
    XFOILConfig,
    load_experiment_config_from_metadata,
)
from rl_airfoil.training.runner import (
    train_td3,
    evaluate_td3,
    train_sac,
    evaluate_sac,
    train_ppo,
    evaluate_ppo,
)


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--algorithm", default="td3", choices=["td3", "sac", "ppo"])
        sp.add_argument("--evaluator", default="surrogate", choices=["surrogate", "xfoil"])
        sp.add_argument("--surrogate-model-name", default="S-1D", choices=["S-1D", "S-2D", "S-3D"])
        sp.add_argument("--surrogate-checkpoint-path", default="checkpoints/surrogate_s1d.pt")
        sp.add_argument("--scaler-json-path", default="checkpoints/scalers.json")
        sp.add_argument("--rl-checkpoint-path", default="")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--total-timesteps", type=int, default=200000)
        sp.add_argument("--aoa", type=float, default=2.0)
        sp.add_argument("--re", type=float, default=1e6)
        sp.add_argument("--action-scale", type=float, default=0.003)
        sp.add_argument("--episode-max-steps", type=int, default=25)

        sp.add_argument("--td3-learning-rate", type=float, default=1e-3)
        sp.add_argument("--td3-buffer-size", type=int, default=100000)
        sp.add_argument("--td3-learning-starts", type=int, default=1000)
        sp.add_argument("--td3-batch-size", type=int, default=256)
        sp.add_argument("--td3-tau", type=float, default=0.005)
        sp.add_argument("--td3-gamma", type=float, default=0.99)
        sp.add_argument("--td3-policy-delay", type=int, default=2)
        sp.add_argument("--td3-target-policy-noise", type=float, default=0.2)
        sp.add_argument("--td3-target-noise-clip", type=float, default=0.5)
        sp.add_argument("--td3-action-noise-sigma", type=float, default=0.1)
        sp.add_argument("--sac-learning-rate", type=float, default=3e-4)
        sp.add_argument("--sac-buffer-size", type=int, default=100000)
        sp.add_argument("--sac-learning-starts", type=int, default=5000)
        sp.add_argument("--sac-batch-size", type=int, default=256)
        sp.add_argument("--sac-tau", type=float, default=0.005)
        sp.add_argument("--sac-gamma", type=float, default=0.99)
        sp.add_argument("--sac-train-freq", type=int, default=1)
        sp.add_argument("--sac-gradient-steps", type=int, default=1)
        sp.add_argument("--sac-ent-coef", default="auto")
        sp.add_argument("--sac-target-entropy", default="auto")
        # PPO hyperparameters
        sp.add_argument("--ppo-learning-rate-start", type=float, default=3e-4)
        sp.add_argument("--ppo-learning-rate-end", type=float, default=3e-5)
        sp.add_argument("--ppo-n-steps", type=int, default=1024)
        sp.add_argument("--ppo-batch-size", type=int, default=256)
        sp.add_argument("--ppo-n-epochs", type=int, default=10)
        sp.add_argument("--ppo-gamma", type=float, default=0.99)
        sp.add_argument("--ppo-gae-lambda", type=float, default=0.95)
        sp.add_argument("--ppo-clip-range", type=float, default=0.2)
        sp.add_argument("--ppo-ent-coef", type=float, default=0.0)
        sp.add_argument("--ppo-vf-coef", type=float, default=0.5)
        sp.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
        # XFOIL solver settings
        sp.add_argument("--xfoil-executable-path", default="")
        sp.add_argument("--xfoil-timeout-sec", type=float, default=30.0)
        sp.add_argument("--xfoil-max-iter", type=int, default=100)
        sp.add_argument("--xfoil-ppar-n", type=int, default=160)
        sp.add_argument("--xfoil-mach", type=float, default=0.0)
        sp.add_argument("--xfoil-ncrit", type=float, default=9.0)
        sp.add_argument("--xfoil-xtr-top", type=float, default=1.0)
        sp.add_argument("--xfoil-xtr-bottom", type=float, default=1.0)
        sp.add_argument("--xfoil-n-points", type=int, default=201)
        # Reward içinde action saturation'ı azaltmak için kullanılan ceza katsayısı
        sp.add_argument("--w-action", type=float, default=0.005)

    t = sub.add_parser("train")
    add_common(t)

    e = sub.add_parser("evaluate")
    add_common(e)
    e.add_argument("--run-dir", required=False, default=None)
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--aoa-sweep", default="-2,0,2,4,6,8")
    return p


def to_cfg(args):
    cfg = ExperimentConfig(
        algorithm=args.algorithm,
        evaluator=args.evaluator,
        surrogate_model_name=args.surrogate_model_name,
        surrogate_checkpoint_path=args.surrogate_checkpoint_path,
        scaler_json_path=args.scaler_json_path,
        rl_checkpoint_path=args.rl_checkpoint_path,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        aoa=args.aoa,
        re=args.re,
        episode_max_steps=args.episode_max_steps,
        action_scale=args.action_scale,
        td3=TD3Hyperparameters(
            learning_rate=args.td3_learning_rate,
            buffer_size=args.td3_buffer_size,
            learning_starts=args.td3_learning_starts,
            batch_size=args.td3_batch_size,
            tau=args.td3_tau,
            gamma=args.td3_gamma,
            policy_delay=args.td3_policy_delay,
            target_policy_noise=args.td3_target_policy_noise,
            target_noise_clip=args.td3_target_noise_clip,
            action_noise_sigma=args.td3_action_noise_sigma,
        ),
        sac=SACHyperparameters(
            learning_rate=args.sac_learning_rate,
            buffer_size=args.sac_buffer_size,
            learning_starts=args.sac_learning_starts,
            batch_size=args.sac_batch_size,
            tau=args.sac_tau,
            gamma=args.sac_gamma,
            train_freq=args.sac_train_freq,
            gradient_steps=args.sac_gradient_steps,
            ent_coef=args.sac_ent_coef,
            target_entropy=args.sac_target_entropy,
        ),
        ppo=PPOHyperparameters(
            learning_rate_start=args.ppo_learning_rate_start,
            learning_rate_end=args.ppo_learning_rate_end,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_n_epochs,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            clip_range=args.ppo_clip_range,
            ent_coef=args.ppo_ent_coef,
            vf_coef=args.ppo_vf_coef,
            max_grad_norm=args.ppo_max_grad_norm,
        ),
        xfoil=XFOILConfig(
        executable_path=(
            args.xfoil_executable_path
            if str(args.xfoil_executable_path).strip()
            else "xfoil"
        ),
        timeout_sec=args.xfoil_timeout_sec,
        max_iter=args.xfoil_max_iter,
        ppar_n=args.xfoil_ppar_n,
        mach=args.xfoil_mach,
        ncrit=args.xfoil_ncrit,
        xtr_top=args.xfoil_xtr_top,
        xtr_bottom=args.xfoil_xtr_bottom,
        n_points=args.xfoil_n_points,
    ),
    )

    cfg.reward_weights.w_action = args.w_action

    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = to_cfg(args)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------
    if args.command == "train":
        algo = cfg.algorithm.lower()

        if algo == "td3":
            run_dir = train_td3(cfg)
        elif algo == "sac":
            run_dir = train_sac(cfg)
        elif algo == "ppo":
            run_dir = train_ppo(cfg)
        else:
            raise ValueError(f"Unsupported algorithm for train: {cfg.algorithm}")

        print(run_dir)
        return

    # ------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------
    run_dir = Path(args.run_dir) if args.run_dir else None

    # Evaluation sırasında --run-dir verilmişse, eğitimde kullanılan
    # gerçek config değerleri experiment_metadata.json dosyasından yüklenir.
    #
    # Böylece şu hatalar engellenir:
    # - SAC checkpoint'inin yanlışlıkla TD3 loader ile açılması
    # - training action_scale=0.003 iken evaluation action_scale=0.02/0.03 çalışması
    # - yanlış surrogate_model_name ile değerlendirme yapılması
    # - reward_weights / constraints / AoA / Re uyumsuzluğu
    if run_dir is not None:
        user_checkpoint_override = bool(str(args.rl_checkpoint_path).strip())

        cfg = load_experiment_config_from_metadata(
            run_dir=run_dir,
            fallback_cfg=cfg,
        )

        # Normal kullanımda buna gerek yok.
        # Ama kullanıcı özellikle --rl-checkpoint-path verdiyse metadata'daki
        # checkpoint yerine bu dosya kullanılır.
        if user_checkpoint_override:
            cfg.rl_checkpoint_path = args.rl_checkpoint_path

        print(
            "[INFO] Evaluation config loaded from metadata: "
            f"{run_dir / 'experiment_metadata.json'}"
        )
        print(
            "[INFO] Loaded config summary: "
            f"algorithm={cfg.algorithm}, "
            f"evaluator={cfg.evaluator}, "
            f"surrogate_model_name={cfg.surrogate_model_name}, "
            f"action_scale={cfg.action_scale}, "
            f"w_action={cfg.reward_weights.w_action}, "
            f"AoA={cfg.aoa}, "
            f"Re={cfg.re}, "
            f"rl_checkpoint_path={cfg.rl_checkpoint_path}"
        )

    # Kritik nokta:
    # Burada args.algorithm kullanılmamalı.
    # Çünkü --run-dir verildiğinde gerçek algoritma metadata içinden gelir.
    algo = cfg.algorithm.lower()

    if algo == "td3":
        evaluate_td3(
            cfg,
            run_dir,
            episodes=args.episodes,
            aoa_sweep=args.aoa_sweep,
        )
    elif algo == "sac":
        evaluate_sac(
            cfg,
            run_dir,
            episodes=args.episodes,
            aoa_sweep=args.aoa_sweep,
        )
    elif algo == "ppo":
        evaluate_ppo(
            cfg,
            run_dir,
            episodes=args.episodes,
            aoa_sweep=args.aoa_sweep,
        )
    else:
        raise ValueError(f"Unsupported algorithm for evaluate: {cfg.algorithm}")

    print(str(run_dir) if run_dir else "auto-generated")
