from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla PPO (rsl-rl) for Safety-Gymnasium point navigation.")
    parser.add_argument("--config", type=str, default="config_pointnav_PPO_DynObs.json")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs_pointnav_ppo_rsl")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    dev = str(device).lower()
    if dev == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            print("[Warn] MPS is not available on this machine/runtime. Falling back to cpu.")
            return "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available on this machine/runtime. Falling back to cpu.")
        return "cpu"
    return dev


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    from cmdp_wrapper import CMDPConfig, RewardCostWrapper
    from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv
    from ppo_agent import RslPPOConfig, SafetyGymVecEnv, build_train_cfg

    device = resolve_device(args.device)
    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0))
    set_seed(seed)

    env_cfg = cfg.get("env", {})
    cmdp_cfg = cfg.get("cmdp", {})
    algo_cfg = cfg.get("algo", {})
    train_cfg = cfg.get("training", {})

    mujoco_cfg = MujocoPointNavConfig(
        env_id=str(env_cfg.get("env_id", "SafetyPointGoal1-v0")),
        max_steps=int(env_cfg.get("max_steps", 1000)),
        action_scale=float(env_cfg.get("action_scale", 1.0)),
        env_config_overrides=dict(env_cfg.get("env_config_overrides", {})),
        env_kwargs=dict(env_cfg.get("env_kwargs", {})),
    )
    cost_limits = tuple(cmdp_cfg.get("cost_limits", [25.0]))
    if "cost_coeffs" in cmdp_cfg:
        cost_coeffs = tuple(cmdp_cfg.get("cost_coeffs", [0.0] * len(cost_limits)))
    else:
        cost_coeff_scalar = float(cmdp_cfg.get("cost_coeff", 0.0))
        cost_coeffs = tuple([cost_coeff_scalar] * len(cost_limits))

    reward_cost_cfg = CMDPConfig(
        cost_limits=tuple(cmdp_cfg.get("cost_limits", [25.0])),
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=tuple(cmdp_cfg.get("cost_scales", [1.0])),
        cost_coeffs=cost_coeffs,
    )

    def env_factory() -> RewardCostWrapper:
        env = MujocoPointNavEnv(mujoco_cfg, render_mode=None)
        return RewardCostWrapper(env, cfg=reward_cost_cfg)

    vec_env = SafetyGymVecEnv(
        env_factory=env_factory,
        num_envs=int(args.num_envs),
        device=device,
        seed=seed,
        gamma=float(algo_cfg.get("gamma", 0.99)),
    )

    ppo_cfg = RslPPOConfig(
        seed=seed,
        max_iterations=int(
            args.max_iterations
            if args.max_iterations is not None
            else train_cfg.get("max_iterations", 300)
        ),
        save_interval=int(train_cfg.get("save_interval", 25)),
        num_steps_per_env=int(algo_cfg.get("num_steps_per_env", 24)),
        num_learning_epochs=int(algo_cfg.get("num_learning_epochs", 5)),
        num_mini_batches=int(algo_cfg.get("num_mini_batches", 4)),
        clip_param=float(algo_cfg.get("clip_param", 0.2)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        lam=float(algo_cfg.get("lam", 0.95)),
        entropy_coef=float(algo_cfg.get("entropy_coef", 0.0)),
        value_loss_coef=float(algo_cfg.get("value_loss_coef", 1.0)),
        learning_rate=float(algo_cfg.get("learning_rate", 3e-4)),
        max_grad_norm=float(algo_cfg.get("max_grad_norm", 1.0)),
        desired_kl=float(algo_cfg.get("desired_kl", 0.01)),
        schedule=str(algo_cfg.get("schedule", "adaptive")),
        hidden_dims=tuple(algo_cfg.get("hidden_dims", [256, 256, 256])),
        activation=str(algo_cfg.get("activation", "elu")),
        init_noise_std=float(algo_cfg.get("init_noise_std", 1.0)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = {
        "env": asdict(mujoco_cfg),
        "cmdp": asdict(reward_cost_cfg),
        "algo": asdict(ppo_cfg),
        "training": {
            "seed": seed,
            "device": device,
            "num_envs": int(args.num_envs),
            "max_iterations": int(ppo_cfg.max_iterations),
            "save_interval": int(ppo_cfg.save_interval),
            "log_dir": str(log_dir),
        },
    }
    (log_dir / "resolved_config.json").write_text(json.dumps(resolved_cfg, indent=2), encoding="utf-8")

    try:
        from rsl_rl.runners import OnPolicyRunner
    except Exception as exc:
        vec_env.close()
        raise ImportError(
            "rsl-rl is required for this trainer. Install it, e.g. `pip install rsl-rl`."
        ) from exc

    runner_cfg = build_train_cfg(
        cfg=ppo_cfg,
        experiment_name=str(Path(args.log_dir).name),
        run_name=run_name,
    )
    runner_log_dir: str | None = str(log_dir)
    try:
        import tensorboard  # noqa: F401
    except Exception:
        print("[Warn] tensorboard is not installed. Continuing without rsl-rl logging/checkpoint writer.")
        runner_log_dir = None

    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg,
        log_dir=runner_log_dir,
        device=device,
    )

    # Add richer loss logging keys and KL distance per iteration.
    # KL is computed between old rollout Gaussian params and updated policy on the same sampled observations.
    orig_update = runner.alg.update

    def _wrap_update():
        old_obs = None
        old_mu = None
        old_sigma = None
        try:
            st = runner.alg.storage
            old_obs = st.observations[0].clone()
            old_mu = st.mu[0].detach().clone()
            old_sigma = st.sigma[0].detach().clone()
        except Exception:
            old_obs = None
            old_mu = None
            old_sigma = None

        loss_dict = orig_update()

        kl_distance = None
        if old_obs is not None and old_mu is not None and old_sigma is not None:
            try:
                with torch.no_grad():
                    runner.alg.actor(old_obs, stochastic_output=True)
                    new_mu = runner.alg.actor.output_mean
                    new_sigma = runner.alg.actor.output_std
                    kl = torch.sum(
                        torch.log(new_sigma / old_sigma + 1.0e-5)
                        + (torch.square(old_sigma) + torch.square(old_mu - new_mu))
                        / (2.0 * torch.square(new_sigma))
                        - 0.5,
                        dim=-1,
                    )
                    kl_distance = float(torch.mean(kl).item())
            except Exception:
                kl_distance = None

        enriched = dict(loss_dict)
        if "surrogate" in loss_dict:
            enriched["surrogate_loss"] = float(loss_dict["surrogate"])
        if "value" in loss_dict:
            enriched["value_loss"] = float(loss_dict["value"])
        if "entropy" in loss_dict:
            enriched["entropy_loss"] = float(loss_dict["entropy"])
        if kl_distance is not None:
            enriched["kl_distance"] = float(kl_distance)
        return enriched

    runner.alg.update = _wrap_update

    runner.learn(
        num_learning_iterations=int(ppo_cfg.max_iterations),
        init_at_random_ep_len=True,
    )
    vec_env.close()


if __name__ == "__main__":
    main()
