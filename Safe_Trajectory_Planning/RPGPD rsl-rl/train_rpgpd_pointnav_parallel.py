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
    parser = argparse.ArgumentParser(description="Parallel RPGPD training for Safety-Gymnasium point navigation.")
    parser.add_argument("--config", type=str, default="config_pointnav_RPGPD_DynObs.json")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs_pointnav_rpgpd_rsl")
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start method for rollout workers",
    )
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


def _tuple_floats(xs: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if xs is None:
        return default
    vals = tuple(float(x) for x in xs)
    return vals if len(vals) > 0 else default


def _patch_runner_log_no_loss_for_diagnostics(runner: Any) -> None:
    """Log selected diagnostics as episode stats instead of algorithm losses."""
    import contextlib
    import io
    import re
    import sys as _sys

    if not hasattr(runner, "log"):
        return
    orig_log = runner.log

    def _is_diag_key(key: str) -> bool:
        return (
            key.startswith("dual_lambda_")
            or key.startswith("constraint_violation_")
            or key.startswith("discounted_cost_")
            or key == "kl_distance"
        )

    def _extract_diag_from_losses(losses: Any) -> Dict[str, float]:
        diag: Dict[str, float] = {}
        if isinstance(losses, dict):
            for key in list(losses.keys()):
                if _is_diag_key(str(key)):
                    diag[str(key)] = float(losses.pop(key))
        elif isinstance(losses, list):
            for item in losses:
                if isinstance(item, dict):
                    for key in list(item.keys()):
                        if _is_diag_key(str(key)):
                            diag[str(key)] = float(item.pop(key))
        return diag

    text_pattern = re.compile(
        r"(Mean\s+(?:kl_distance|dual_lambda_\d+|constraint_violation_\d+|discounted_cost_\d+))\s+loss:",
    )

    def patched_log(*args, **kwargs):
        locs = None
        if len(args) > 0 and isinstance(args[0], dict):
            locs = args[0]
        elif isinstance(kwargs.get("locs"), dict):
            locs = kwargs["locs"]

        if isinstance(locs, dict):
            diag = _extract_diag_from_losses(locs.get("losses", {}))
            if diag:
                ep_infos = locs.get("ep_infos")
                if ep_infos is None or not isinstance(ep_infos, list):
                    ep_infos = []
                    locs["ep_infos"] = ep_infos
                fake_ep = {k: torch.tensor([v], dtype=torch.float32) for k, v in diag.items()}
                ep_infos.append(fake_ep)

        out_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf):
            orig_log(*args, **kwargs)
        text = text_pattern.sub(r"\1:", out_buf.getvalue())
        _sys.stdout.write(text)
        _sys.stdout.flush()

    runner.log = patched_log


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    from cmdp_wrapper import CMDPConfig
    from mujoco_env import MujocoPointNavConfig
    from rpgpd_agent_parallel import ParallelSafetyGymVecEnv, RPGPDTrainerHook, RslRPGPDConfig, build_train_cfg

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

    cost_limits = _tuple_floats(cmdp_cfg.get("cost_limits", [25.0]), (25.0,))
    cost_scales = _tuple_floats(cmdp_cfg.get("cost_scales", [1.0] * len(cost_limits)), tuple([1.0] * len(cost_limits)))
    if len(cost_scales) != len(cost_limits):
        raise ValueError("cmdp.cost_scales length must match cmdp.cost_limits")
    reward_cost_cfg = CMDPConfig(
        cost_limits=cost_limits,
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=cost_scales,
    )

    vec_env = ParallelSafetyGymVecEnv(
        mujoco_cfg=mujoco_cfg,
        cmdp_cfg=reward_cost_cfg,
        num_envs=int(args.num_envs),
        device=device,
        seed=seed,
        gamma=float(algo_cfg.get("gamma", 0.99)),
        start_method=str(args.start_method),
    )

    rpgpd_cfg = RslRPGPDConfig(
        seed=seed,
        max_iterations=int(args.max_iterations if args.max_iterations is not None else train_cfg.get("max_iterations", 300)),
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
        hidden_dims=tuple(int(x) for x in algo_cfg.get("hidden_dims", [256, 256, 256])),
        activation=str(algo_cfg.get("activation", "elu")),
        init_noise_std=float(algo_cfg.get("init_noise_std", 1.0)),
        dual_lr=float(algo_cfg.get("dual_lr", 0.1)),
        dual_tau=float(algo_cfg.get("dual_tau", 0.0)),
        lambda_init=float(algo_cfg.get("lambda_init", 0.0)),
        lambda_max=float(algo_cfg.get("lambda_max", 1000.0)),
        cost_value_loss_coef=float(algo_cfg.get("cost_value_loss_coef", 1.0)),
        value_learning_rate=float(algo_cfg.get("value_learning_rate", 1e-3)),
        normalize_reward_advantage=bool(algo_cfg.get("normalize_reward_advantage", True)),
        normalize_cost_advantages=bool(algo_cfg.get("normalize_cost_advantages", True)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = {
        "env": asdict(mujoco_cfg),
        "cmdp": asdict(reward_cost_cfg),
        "algo": asdict(rpgpd_cfg),
        "training": {
            "seed": seed,
            "device": device,
            "num_envs": int(args.num_envs),
            "max_iterations": int(rpgpd_cfg.max_iterations),
            "save_interval": int(rpgpd_cfg.save_interval),
            "log_dir": str(log_dir),
            "start_method": str(args.start_method),
        },
    }
    (log_dir / "resolved_config.json").write_text(json.dumps(resolved_cfg, indent=2), encoding="utf-8")

    try:
        from rsl_rl.runners import OnPolicyRunner
    except Exception as exc:
        vec_env.close()
        raise ImportError("rsl-rl is required for this trainer. Install it, e.g. `pip install rsl-rl`.") from exc

    runner_cfg = build_train_cfg(
        cfg=rpgpd_cfg,
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
    _patch_runner_log_no_loss_for_diagnostics(runner)

    rpgpd_hook = RPGPDTrainerHook(
        runner=runner,
        vec_env=vec_env,
        cfg=rpgpd_cfg,
        cost_limits=np.asarray(cost_limits, dtype=np.float32),
        device=device,
    )
    rpgpd_hook.install()

    runner.learn(
        num_learning_iterations=int(rpgpd_cfg.max_iterations),
        init_at_random_ep_len=True,
    )
    vec_env.close()


if __name__ == "__main__":
    main()
