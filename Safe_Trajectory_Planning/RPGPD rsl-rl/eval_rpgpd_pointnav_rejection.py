from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tensordict import TensorDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RPGPD policy with rejection sampling on cost limit.")
    p.add_argument("--config", type=str, default="config_pointnav_RPGPD_DynObs.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=10, help="Accepted episodes to collect")
    p.add_argument("--max_attempts", type=int, default=1000, help="Upper bound on sampled episodes")
    p.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions")
    p.add_argument("--save_json", type=str, default=None, help="Optional summary JSON output path")
    return p.parse_args()


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
            print("[Warn] MPS unavailable. Using cpu.")
            return "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Using cpu.")
        return "cpu"
    return dev


def make_policy(config: Dict[str, Any], checkpoint: str, device: str, seed: int):
    from cmdp_wrapper import CMDPConfig, RewardCostWrapper
    from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv
    from rpgpd_agent import RslRPGPDConfig, SafetyGymVecEnv, build_train_cfg

    try:
        from rsl_rl.runners import OnPolicyRunner
    except Exception as exc:
        raise ImportError("rsl-rl is required for evaluation. Install `rsl-rl`.") from exc

    env_cfg = config.get("env", {})
    cmdp_cfg = config.get("cmdp", {})
    algo_cfg = config.get("algo", {})
    tr_cfg = config.get("training", {})

    mujoco_cfg = MujocoPointNavConfig(
        env_id=str(env_cfg.get("env_id", "SafetyPointGoal1-v0")),
        max_steps=int(env_cfg.get("max_steps", 1000)),
        action_scale=float(env_cfg.get("action_scale", 1.0)),
        env_config_overrides=dict(env_cfg.get("env_config_overrides", {})),
        env_kwargs=dict(env_cfg.get("env_kwargs", {})),
    )

    cost_limits = tuple(cmdp_cfg.get("cost_limits", [25.0]))
    reward_cost_cfg = CMDPConfig(
        cost_limits=cost_limits,
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=tuple(cmdp_cfg.get("cost_scales", [1.0] * len(cost_limits))),
    )

    def env_factory() -> RewardCostWrapper:
        env = MujocoPointNavEnv(mujoco_cfg, render_mode=None)
        return RewardCostWrapper(env, cfg=reward_cost_cfg)

    vec_env = SafetyGymVecEnv(env_factory=env_factory, num_envs=1, device=device, seed=seed, gamma=float(algo_cfg.get("gamma", 0.99)))
    rpgpd_cfg = RslRPGPDConfig(
        seed=seed,
        max_iterations=int(tr_cfg.get("max_iterations", 300)),
        save_interval=int(tr_cfg.get("save_interval", 25)),
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

    runner_cfg = build_train_cfg(rpgpd_cfg, experiment_name="eval", run_name="eval")
    runner = OnPolicyRunner(env=vec_env, train_cfg=runner_cfg, log_dir=None, device=device)
    runner.load(checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    vec_env.close()
    return policy


@torch.no_grad()
def rollout_episode(env: Any, policy: Any, device: str, max_steps: int, seed: int, stochastic: bool) -> Dict[str, Any]:
    obs_np, info = env.reset(seed=seed)
    ep_ret = 0.0
    ep_cost = np.zeros((len(info.get("cost_limits", [0.0])),), dtype=np.float32)

    for t in range(max_steps):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
        obs_td = TensorDict({"policy": obs}, batch_size=[1], device=device)
        act = policy(obs_td, stochastic_output=bool(stochastic))
        act_np = act.squeeze(0).detach().cpu().numpy()

        obs_np, rew, terminated, truncated, info = env.step(act_np)
        ep_ret += float(rew)
        costs = np.asarray(info.get("costs", [0.0]), dtype=np.float32).reshape(-1)
        if costs.shape[0] == ep_cost.shape[0]:
            ep_cost += costs

        if bool(terminated or truncated):
            return {
                "EpRet": float(ep_ret),
                "EpLen": int(t + 1),
                "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
                "EpCostVec": ep_cost.copy(),
            }

    return {
        "EpRet": float(ep_ret),
        "EpLen": int(max_steps),
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpCostVec": ep_cost.copy(),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    set_seed(int(args.seed))

    policy = make_policy(cfg, args.checkpoint, device, int(args.seed))

    from cmdp_wrapper import CMDPConfig, RewardCostWrapper
    from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv

    env_cfg = cfg.get("env", {})
    cmdp_cfg = cfg.get("cmdp", {})
    max_steps = int(env_cfg.get("max_steps", 1000))

    mujoco_cfg = MujocoPointNavConfig(
        env_id=str(env_cfg.get("env_id", "SafetyPointGoal1-v0")),
        max_steps=max_steps,
        action_scale=float(env_cfg.get("action_scale", 1.0)),
        env_config_overrides=dict(env_cfg.get("env_config_overrides", {})),
        env_kwargs=dict(env_cfg.get("env_kwargs", {})),
    )

    cost_limits = tuple(cmdp_cfg.get("cost_limits", [25.0]))
    cost_limit0 = float(cost_limits[0]) if len(cost_limits) > 0 else float("inf")
    reward_cost_cfg = CMDPConfig(
        cost_limits=cost_limits,
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=tuple(cmdp_cfg.get("cost_scales", [1.0] * len(cost_limits))),
    )

    eval_env = RewardCostWrapper(MujocoPointNavEnv(mujoco_cfg, render_mode=None), cfg=reward_cost_cfg)

    accepted: List[Dict[str, Any]] = []
    rejected = 0
    for attempt in range(int(args.max_attempts)):
        if len(accepted) >= int(args.num_episodes):
            break
        ep_seed = int(args.seed) + attempt
        out = rollout_episode(eval_env, policy, device, max_steps, ep_seed, bool(args.stochastic))
        if float(out["EpCost0"]) <= cost_limit0:
            accepted.append(out)
            print(
                f"[Eval] accept idx={len(accepted)-1:03d} seed={ep_seed} ret={out['EpRet']:.3f} "
                f"len={out['EpLen']} cost0={out['EpCost0']:.3f}"
            )
        else:
            rejected += 1

    eval_env.close()

    if len(accepted) == 0:
        print(
            f"[Eval] no accepted episodes after {args.max_attempts} attempts. "
            f"cost_limit0={cost_limit0:.3f}"
        )
        return

    rets = np.asarray([x["EpRet"] for x in accepted], dtype=np.float32)
    lens = np.asarray([x["EpLen"] for x in accepted], dtype=np.float32)
    c0 = np.asarray([x["EpCost0"] for x in accepted], dtype=np.float32)

    print(
        "[Eval] summary "
        f"accepted={len(accepted)} rejected={rejected} attempts={len(accepted)+rejected} "
        f"cost_limit0={cost_limit0:.3f} "
        f"return_mean={rets.mean():.3f} return_std={rets.std():.3f} "
        f"len_mean={lens.mean():.2f} cost0_mean={c0.mean():.3f} cost0_std={c0.std():.3f}"
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "accepted": int(len(accepted)),
            "rejected": int(rejected),
            "attempts": int(len(accepted) + rejected),
            "cost_limit0": float(cost_limit0),
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std()),
            "len_mean": float(lens.mean()),
            "cost0_mean": float(c0.mean()),
            "cost0_std": float(c0.std()),
            "episodes": accepted,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Eval] wrote {out_path}")


if __name__ == "__main__":
    main()
