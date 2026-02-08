"""
train_rpgpdppo_pointnav.py

Train RPG-PD integrated with PPO on MuJoCo Safety-Gymnasium point navigation.

Usage:
  python train_rpgpdppo_pointnav.py --config config_pointnav_RPGPD.json --device cpu --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure local repo code is imported before any site-packages copy.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mujoco_env import MujocoPointNavEnv, MujocoPointNavConfig
from cmdp_wrapper import RewardCostWrapper, CMDPConfig
from rpgpd_ppo_agent import RPGPDPPOAgent, RPGPDPPOTrainer, RPGPDPPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_pointnav_RPGPD.json", help="Path to JSON config")
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    p.add_argument("--seed", type=int, default=None, help="override seed in config")
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p.add_argument("--total_steps", type=int, default=None, help="Override total steps; epochs = total_steps/steps_per_epoch")

    # overrides
    p.add_argument("--steps_per_epoch", type=int, default=None, help="Deprecated: ignored in this trainer")
    p.add_argument("--num_roll_out", type=int, default=None)
    p.add_argument("--max_ep_len", type=int, default=None)
    p.add_argument("--checkpoint_every", type=int, default=1)
    # Parallel rollout workers (only used when num_roll_out is set)
    p.add_argument("--rollout_parallel", type=int, default=1)
    # Accepted for CLI compatibility (not used in this script)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--live", action="store_true", help="Render environment live during sequential rollout collection")
    p.add_argument(
        "--fixed_rollout_seeds",
        action="store_true",
        help="Disable per-rollout random seeding and use deterministic seed schedule",
    )

    # outputs
    p.add_argument("--save_dir", type=str, default="runs_pointnav_rpgpd")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_metric_plot(save_dir: Path, values: list[float], name: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    if len(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Training {ylabel} per Epoch")
    ax.grid(True, alpha=0.3)
    out = save_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_checkpoint(save_dir: Path, ep: int, agent: RPGPDPPOAgent, cfg: RPGPDPPOConfig, obs_dim: int, act_dim: int, num_costs: int) -> Path:
    ckpt = {
        "epoch": int(ep),
        "pi": agent.pi.state_dict(),
        "v_r": agent.v_r.state_dict(),
        "v_c": agent.v_c.state_dict(),
        "lambda": agent.lam,
        "cfg": asdict(cfg),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "num_costs": int(num_costs),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = save_dir / f"ckpt_{ts}_epoch_{ep:03d}.pt"
    torch.save(ckpt, out)
    return out


@torch.no_grad()
def _rollout_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect one episode using a fixed policy (used for parallel rollout).
    """
    torch.set_num_threads(1)
    cfg = args["cfg"]
    algo_cfg = RPGPDPPOConfig(**args["algo_cfg"])

    env_cfg = MujocoPointNavConfig(**cfg["env"])
    env = MujocoPointNavEnv(env_cfg, render_mode=None)
    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)

    agent = RPGPDPPOAgent(
        obs_dim=wrapped.obs_dim(),
        act_dim=wrapped.act_dim(),
        num_costs=len(cmdp_cfg.cost_limits),
        cfg=algo_cfg,
        device=args["device"],
    )
    agent.pi.load_state_dict(args["state_dicts"]["pi"])
    agent.v_r.load_state_dict(args["state_dicts"]["v_r"])
    agent.v_c.load_state_dict(args["state_dicts"]["v_c"])
    agent.pi.eval()
    agent.v_r.eval()
    agent.v_c.eval()

    obs, info = wrapped.reset(seed=args["seed"])
    max_steps = int(args["max_steps"])
    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0

    traj = {
        "obs": [],
        "act": [],
        "logp": [],
        "rew": [],
        "costs": [],
        "val_r": [],
        "val_c": [],
        "done": [],
    }

    for t in range(max_steps):
        act, logp, v_r, v_c = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, info = wrapped.step(act)
        costs = np.asarray(info["costs"], dtype=np.float32)
        done = bool(terminated or truncated or (t + 1 >= max_steps))

        traj["obs"].append(obs)
        traj["act"].append(act)
        traj["logp"].append(float(logp))
        traj["rew"].append(float(rew))
        traj["costs"].append(costs)
        traj["val_r"].append(float(v_r))
        traj["val_c"].append(v_c)
        traj["done"].append(done)

        ep_ret += float(rew)
        ep_cost += costs
        ep_len += 1
        obs = next_obs

        if done:
            break

    traj_np = {
        "obs": np.asarray(traj["obs"], dtype=np.float32),
        "act": np.asarray(traj["act"], dtype=np.float32),
        "logp": np.asarray(traj["logp"], dtype=np.float32),
        "rew": np.asarray(traj["rew"], dtype=np.float32),
        "costs": np.asarray(traj["costs"], dtype=np.float32),
        "val_r": np.asarray(traj["val_r"], dtype=np.float32),
        "val_c": np.asarray(traj["val_c"], dtype=np.float32),
        "done": np.asarray(traj["done"], dtype=np.float32),
    }

    return {
        "traj": traj_np,
        "ep_ret": float(ep_ret),
        "ep_cost": ep_cost.astype(np.float32),
        "ep_len": int(ep_len),
        "seed": int(args["seed"]),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    random_seed_each_rollout = not args.fixed_rollout_seeds
    seed_cfg = cfg.get("training", {}).get("seed", 0)
    seed = args.seed if args.seed is not None else seed_cfg
    if random_seed_each_rollout and args.seed is None:
        # Default behavior: new random base seed every run.
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    np.random.seed(seed)
    torch.manual_seed(seed)


    env_cfg = MujocoPointNavConfig(**cfg["env"])
    render_mode = "human" if args.live else None
    env = MujocoPointNavEnv(env_cfg, render_mode=render_mode)

    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)

    algo = cfg["algo"].copy()
    if args.max_ep_len is not None:
        algo["max_ep_len"] = int(args.max_ep_len)
    if args.num_roll_out is not None:
        algo["num_roll_out"] = int(args.num_roll_out)

    algo_cfg = RPGPDPPOConfig(
        gamma=float(algo["gamma"]),
        lam=float(algo["lam"]),
        clip_ratio=float(algo["clip_ratio"]),
        target_kl=float(algo["target_kl"]),
        pi_lr=float(algo["pi_lr"]),
        vf_lr=float(algo["vf_lr"]),
        train_pi_iters=int(algo["train_pi_iters"]),
        train_v_iters=int(algo["train_v_iters"]),
        max_grad_norm=float(algo["max_grad_norm"]),
        # kept for config compatibility; collection is controlled by num_roll_out * max_ep_len
        steps_per_epoch=int(algo.get("steps_per_epoch", algo["max_ep_len"])),
        max_ep_len=int(algo["max_ep_len"]),
        num_roll_out=int(algo.get("num_roll_out")) if algo.get("num_roll_out") is not None else None,
        hidden_sizes=tuple(int(x) for x in algo["hidden_sizes"]),
        entropy_coef=float(algo.get("entropy_coef", 0.0)),
        normalize_advantages=bool(algo.get("normalize_advantages", True)),
        dual_lr=float(algo.get("dual_lr", 0.5)),
        dual_tau=float(algo.get("dual_tau", 0.0)),
        lambda_init=float(algo.get("lambda_init", 0.0)),
        lambda_max=float(algo.get("lambda_max", 1000.0)),
        dual_scale_one_minus_gamma=bool(algo.get("dual_scale_one_minus_gamma", True)),
    )

    if algo_cfg.num_roll_out is None:
        algo_cfg.num_roll_out = 1

    if args.epochs is not None:
        epochs = int(args.epochs)
    else:
        total_steps = int(args.total_steps) if args.total_steps is not None else int(cfg["training"]["total_steps"])
        steps_per_epoch_eff = max(1, int(algo_cfg.num_roll_out) * int(algo_cfg.max_ep_len))
        epochs = max(1, total_steps // steps_per_epoch_eff)

    agent = RPGPDPPOAgent(
        obs_dim=wrapped.obs_dim(),
        act_dim=wrapped.act_dim(),
        num_costs=len(cmdp_cfg.cost_limits),
        cfg=algo_cfg,
        device=args.device,
    )
    trainer = RPGPDPPOTrainer(
        env=wrapped,
        agent=agent,
        cost_limits=np.asarray(cmdp_cfg.cost_limits, dtype=np.float32),
        cfg=algo_cfg,
    )

    # resume/init
    start_epoch = 1
    if args.init_checkpoint:
        ckpt = torch.load(Path(args.init_checkpoint).resolve(), map_location=args.device)
        agent.pi.load_state_dict(ckpt.get("pi", ckpt.get("pi_hat")))
        agent.v_r.load_state_dict(ckpt["v_r"])
        agent.v_c.load_state_dict(ckpt["v_c"])
        agent.lam = np.asarray(ckpt.get("lambda", ckpt.get("lambda_hat", agent.lam)), dtype=np.float32)
        print(f"[Init] Loaded weights from {args.init_checkpoint}")
    elif args.resume_checkpoint:
        ckpt = torch.load(Path(args.resume_checkpoint).resolve(), map_location=args.device)
        agent.pi.load_state_dict(ckpt.get("pi", ckpt.get("pi_hat")))
        agent.v_r.load_state_dict(ckpt["v_r"])
        agent.v_c.load_state_dict(ckpt["v_c"])
        agent.lam = np.asarray(ckpt.get("lambda", ckpt.get("lambda_hat", agent.lam)), dtype=np.float32)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] Loaded checkpoint {args.resume_checkpoint} (start_epoch={start_epoch})")

    print(
        f"[Train] epochs={epochs} episodes_per_epoch={algo_cfg.num_roll_out} "
        f"max_ep_len={algo_cfg.max_ep_len} device={args.device} seed={seed}"
    )
    if random_seed_each_rollout:
        print("[Seed] random_seed_each_rollout enabled: each episode reset uses a fresh random seed.")
    if args.live and algo_cfg.num_roll_out is not None and int(args.rollout_parallel) > 1:
        print("[Live] Live rendering is only supported in sequential rollout. Disabling rollout_parallel.")
        args.rollout_parallel = 1
    if int(args.rollout_parallel) > 1 and int(algo_cfg.num_roll_out) <= 1:
        print("[Rollout] rollout_parallel>1 with a single episode per epoch has no benefit; using sequential collection.")
        args.rollout_parallel = 1

    ret_hist: list[float] = []
    c0_hist: list[float] = []

    for ep in range(start_epoch, epochs + 1):
        trainer.buf.reset()
        if int(algo_cfg.num_roll_out) > 1 and int(args.rollout_parallel) > 1:
            rollout_device = args.device
            if args.device != "cpu":
                rollout_device = "cpu"
                print("[Rollout] Parallel rollout uses CPU workers; switching device to cpu.")

            state_dicts = {
                "pi": {k: v.detach().cpu() for k, v in agent.pi.state_dict().items()},
                "v_r": {k: v.detach().cpu() for k, v in agent.v_r.state_dict().items()},
                "v_c": {k: v.detach().cpu() for k, v in agent.v_c.state_dict().items()},
            }
            algo_cfg_dict = asdict(algo_cfg)
            rollout_total = int(algo_cfg.num_roll_out)
            if random_seed_each_rollout:
                seeds = [int(np.random.randint(0, 2**31 - 1)) for _ in range(rollout_total)]
            else:
                seeds = [seed + 1000 * ep + i for i in range(rollout_total)]
            worker_args = [
                {
                    "cfg": cfg,
                    "algo_cfg": algo_cfg_dict,
                    "state_dicts": state_dicts,
                    "seed": s,
                    "max_steps": algo_cfg.max_ep_len,
                    "device": rollout_device,
                }
                for s in seeds
            ]

            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(args.rollout_parallel)) as pool:
                summaries = pool.map(_rollout_worker, worker_args) if worker_args else []

            ep_rets: List[float] = []
            ep_costs_epoch: List[np.ndarray] = []
            ep_lens: List[int] = []
            for summ in summaries:
                traj = summ["traj"]
                for i in range(traj["obs"].shape[0]):
                    trainer.buf.store(
                        traj["obs"][i],
                        traj["act"][i],
                        traj["logp"][i],
                        traj["rew"][i],
                        traj["costs"][i],
                        traj["val_r"][i],
                        traj["val_c"][i],
                        bool(traj["done"][i]),
                    )
                last_val_r = 0.0
                last_val_c = np.zeros_like(summ["ep_cost"], dtype=np.float32)
                trainer.buf.finish_path(last_val_r=last_val_r, last_val_c=last_val_c)
                ep_rets.append(summ["ep_ret"])
                ep_costs_epoch.append(summ["ep_cost"])
                ep_lens.append(summ["ep_len"])

            collect = {
                "EpRetMean": float(np.mean(ep_rets)) if ep_rets else 0.0,
                "EpRetStd": float(np.std(ep_rets)) if ep_rets else 0.0,
                "EpLenMean": float(np.mean(ep_lens)) if ep_lens else 0.0,
            }
            if ep_costs_epoch:
                C = np.stack(ep_costs_epoch, axis=0)
                for i in range(C.shape[1]):
                    collect[f"EpCost{i}Mean"] = float(np.mean(C[:, i]))
                    collect[f"EpCost{i}Std"] = float(np.std(C[:, i]))
        else:
            collect = trainer.collect_epoch(
                seed=seed + ep,
                live=args.live,
                random_seed_each_rollout=random_seed_each_rollout,
            )
        upd = trainer.train_epoch()
        ret_hist.append(float(collect.get("EpRetMean", 0.0)))
        c0_hist.append(float(collect.get("EpCost0Mean", 0.0)))
        lam0 = float(agent.lam[0]) if len(agent.lam) > 0 else 0.0
        jc0_mean = float(upd.get("Jc_mean_0", upd.get("Jc_0", 0.0)))
        jc0_std = float(upd.get("Jc_std_0", 0.0))
        jr_mean = float(upd.get("Jr_mean", 0.0))
        jr_std = float(upd.get("Jr_std", 0.0))
        viol0 = float(upd.get("viol_0", 0.0))

        print(
            f"Epoch {ep:03d} | Ret {collect['EpRetMean']:.2f} (std {collect.get('EpRetStd', 0.0):.3f}) | "
            f"Jr {jr_mean:.3f} (std {jr_std:.3f}) | "
            f"C0 {collect.get('EpCost0Mean', 0.0):.2f} (std {collect.get('EpCost0Std', 0.0):.3f}) | "
            f"Jc0 {jc0_mean:.3f} (std {jc0_std:.3f}) | viol0 {viol0:.3f} | "
            f"KL {upd.get('kl', 0.0):.4f} | λ0 {lam0:.3f}"
        )

        if args.checkpoint_every > 0 and (ep % args.checkpoint_every == 0):
            ckpt_path = save_checkpoint(save_dir, ep, agent, algo_cfg, wrapped.obs_dim(), wrapped.act_dim(), len(cmdp_cfg.cost_limits))
            ckpt_tag = ckpt_path.stem
            save_metric_plot(save_dir, ret_hist, f"returns_{ckpt_tag}", "EpRetMean")
            save_metric_plot(save_dir, c0_hist, f"costs_{ckpt_tag}", "EpCost0Mean")
            print(f"  [Saved] {ckpt_path}")


if __name__ == "__main__":
    main()
