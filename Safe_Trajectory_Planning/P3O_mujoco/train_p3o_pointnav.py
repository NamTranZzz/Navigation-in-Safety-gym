"""
train_p3o_pointnav.py

Runnable entrypoint:
- Builds Safety-Gymnasium point navigation env (MuJoCo)
- Wraps it with RewardCostWrapper (reward + multi-cost CMDP)
- Trains P3O with fixed kappa=20 (paper default)

Usage:
  python train_p3o_pointnav.py --config config_pointnav_p3o.json --epochs 50 --device cpu
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from mujoco_env import MujocoPointNavEnv, MujocoPointNavConfig
from cmdp_wrapper import RewardCostWrapper, CMDPConfig
from p3o_agent import P3OAgent, P3OTrainer, P3OConfig
from eval_p3o_pointnav import rollout_frames, save_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_pointnav_p3o.json", help="Path to JSON config")
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    p.add_argument("--seed", type=int, default=None, help="override seed in config")
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs (each collects steps_per_epoch steps)")
    p.add_argument("--total_steps", type=int, default=None, help="Override total steps; epochs = total_steps/steps_per_epoch")

    # Common overrides
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--num_roll_out", type=int, default=None, help="Collect this many rollouts per epoch")
    p.add_argument("--max_ep_len", type=int, default=None, help="Maximum length for each rollout")
    p.add_argument("--rollout_parallel", type=int, default=1, help="Parallel rollout workers per epoch")
    p.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--eval_episodes", type=int, default=5, help="Number of eval rollouts per epoch")
    p.add_argument("--eval_parallel", type=int, default=1, help="Parallel eval workers (CPU)")
    p.add_argument("--eval_max_steps", type=int, default=None, help="Override eval max steps")

    # Outputs
    p.add_argument("--live", action="store_true", help="Render live during training collection (slow)")
    p.add_argument("--save_dir", type=str, default="runs_pointnav", help="Directory to save logs/checkpoints")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--init_checkpoint", type=str, default=None, help="Path to checkpoint .pt to init weights from")

    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metric_plot(save_dir: Path, values: List[float], name: str, ylabel: str) -> None:
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


def save_checkpoint(save_dir: Path, ep: int, agent: P3OAgent, cfg: P3OConfig, obs_dim: int, act_dim: int, num_costs: int) -> Path:
    ckpt = {
        "epoch": int(ep),
        "pi": agent.pi.state_dict(),
        "v_r": agent.v_r.state_dict(),
        "v_c": agent.v_c.state_dict(),
        "cfg": asdict(cfg),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "num_costs": int(num_costs),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = save_dir / f"ckpt_{ts}_epoch_{ep:03d}.pt"
    torch.save(ckpt, out)
    return out


def _build_eval_env(cfg: Dict[str, Any], render_mode: str | None = None) -> RewardCostWrapper:
    env_cfg = MujocoPointNavConfig(**cfg["env"])
    env = MujocoPointNavEnv(env_cfg, render_mode=render_mode)
    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)
    return wrapped


@torch.no_grad()
def _rollout_summary(env: RewardCostWrapper, agent: P3OAgent, max_steps: int, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0
    terminated = False

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(act)
        ep_ret += float(r)
        ep_cost += np.asarray(info["costs"], dtype=np.float32)
        ep_len += 1
        if terminated or truncated:
            break

    return {
        "EpRet": ep_ret,
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpLen": ep_len,
        "Terminated": bool(terminated),
    }


def _eval_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    cfg = args["cfg"]
    algo_cfg = P3OConfig(**args["algo_cfg"])
    env = _build_eval_env(cfg)
    agent = P3OAgent(
        obs_dim=env.obs_dim(),
        act_dim=env.act_dim(),
        num_costs=len(env.cfg.cost_limits),
        cfg=algo_cfg,
        device=args["device"],
    )
    agent.pi.load_state_dict(args["state_dicts"]["pi"])
    agent.v_r.load_state_dict(args["state_dicts"]["v_r"])
    agent.v_c.load_state_dict(args["state_dicts"]["v_c"])
    agent.pi.eval()
    agent.v_r.eval()
    agent.v_c.eval()
    summ = _rollout_summary(env, agent, max_steps=args["max_steps"], seed=args["seed"])
    summ["seed"] = int(args["seed"])
    return summ


@torch.no_grad()
def _rollout_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    cfg = args["cfg"]
    algo_cfg = P3OConfig(**args["algo_cfg"])
    env = _build_eval_env(cfg)
    agent = P3OAgent(
        obs_dim=env.obs_dim(),
        act_dim=env.act_dim(),
        num_costs=len(env.cfg.cost_limits),
        cfg=algo_cfg,
        device=args["device"],
    )
    agent.pi.load_state_dict(args["state_dicts"]["pi"])
    agent.v_r.load_state_dict(args["state_dicts"]["v_r"])
    agent.v_c.load_state_dict(args["state_dicts"]["v_c"])
    agent.pi.eval()
    agent.v_r.eval()
    agent.v_c.eval()

    obs, info = env.reset(seed=args["seed"])
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
        next_obs, rew, terminated, truncated, info = env.step(act)
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


@torch.no_grad()
def _collect_rollout_local(trainer: P3OTrainer, max_steps: int, seed: int) -> Dict[str, Any]:
    obs, info = trainer.env.reset(seed=seed)
    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0

    for t in range(max_steps):
        act, logp, v_r, v_c = trainer.agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, info = trainer.env.step(act)
        costs = np.asarray(info["costs"], dtype=np.float32)
        done = bool(terminated or truncated or (t + 1 >= max_steps))

        trainer.buf.store(obs, act, logp, rew, costs, v_r, v_c, done)

        ep_ret += float(rew)
        ep_cost += costs
        ep_len += 1
        obs = next_obs

        if done:
            break

    last_val_r = 0.0
    last_val_c = np.zeros_like(ep_cost, dtype=np.float32)
    trainer.buf.finish_path(last_val_r=last_val_r, last_val_c=last_val_c)

    return {"ep_ret": float(ep_ret), "ep_cost": ep_cost, "ep_len": int(ep_len), "seed": int(seed)}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)

    env_cfg = MujocoPointNavConfig(**cfg["env"])
    env = MujocoPointNavEnv(env_cfg, render_mode=("human" if args.live else None))

    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)

    algo_dict = cfg["algo"].copy()
    if args.steps_per_epoch is not None:
        algo_dict["steps_per_epoch"] = int(args.steps_per_epoch)
    if args.max_ep_len is not None:
        algo_dict["max_ep_len"] = int(args.max_ep_len)
    if args.num_roll_out is not None:
        algo_dict["num_roll_out"] = int(args.num_roll_out)
    algo_cfg = P3OConfig(
        gamma=float(algo_dict["gamma"]),
        lam=float(algo_dict["lam"]),
        clip_ratio=float(algo_dict["clip_ratio"]),
        kappa=float(algo_dict["kappa"]),
        target_kl=float(algo_dict["target_kl"]),
        pi_lr=float(algo_dict["pi_lr"]),
        vf_lr=float(algo_dict["vf_lr"]),
        train_pi_iters=int(algo_dict["train_pi_iters"]),
        train_v_iters=int(algo_dict["train_v_iters"]),
        max_grad_norm=float(algo_dict["max_grad_norm"]),
        steps_per_epoch=int(algo_dict["steps_per_epoch"]),
        max_ep_len=int(algo_dict["max_ep_len"]),
        num_roll_out=int(algo_dict.get("num_roll_out")) if algo_dict.get("num_roll_out") is not None else None,
        hidden_sizes=tuple(int(x) for x in algo_dict["hidden_sizes"]),
        entropy_coef=float(algo_dict.get("entropy_coef", 0.0)),
        normalize_advantages=bool(algo_dict.get("normalize_advantages", True)),
    )

    if args.epochs is not None:
        epochs = int(args.epochs)
    else:
        total_steps = int(args.total_steps) if args.total_steps is not None else int(cfg["training"]["total_steps"])
        epochs = max(1, total_steps // algo_cfg.steps_per_epoch)

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = P3OAgent(
        obs_dim=wrapped.obs_dim(),
        act_dim=wrapped.act_dim(),
        num_costs=len(cmdp_cfg.cost_limits),
        cfg=algo_cfg,
        device=args.device,
    )
    trainer = P3OTrainer(
        env=wrapped,
        agent=agent,
        cost_limits=np.asarray(cmdp_cfg.cost_limits, dtype=np.float32),
        cfg=algo_cfg,
    )

    eval_max_steps = int(args.eval_max_steps) if args.eval_max_steps is not None else int(env_cfg.max_steps)

    resolved = {
        "env": cfg["env"],
        "cmdp": cfg["cmdp"],
        "algo": {
            **cfg["algo"],
            **({"steps_per_epoch": algo_cfg.steps_per_epoch} if args.steps_per_epoch is not None else {}),
            **({"num_roll_out": int(algo_cfg.num_roll_out)} if algo_cfg.num_roll_out is not None else {}),
            **({"max_ep_len": int(algo_cfg.max_ep_len)} if args.max_ep_len is not None else {}),
        },
        "training": {
            **cfg.get("training", {}),
            "seed": seed,
            "epochs": epochs,
            "checkpoint_every": int(args.checkpoint_every),
            "eval_episodes": int(args.eval_episodes),
            "eval_parallel": int(args.eval_parallel),
            "eval_max_steps": int(eval_max_steps),
            "rollout_parallel": int(args.rollout_parallel),
        },
    }
    with open(save_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)

    print(f"[PointNav] obs_dim={wrapped.obs_dim()} act_dim={wrapped.act_dim()} costs={len(cmdp_cfg.cost_limits)}")
    print(f"[Train] epochs={epochs} steps_per_epoch={algo_cfg.steps_per_epoch} device={args.device} seed={seed}")
    print(
        "[Config] kappa={kappa} gamma={gamma} lam={lam} clip={clip}".format(
            kappa=algo_cfg.kappa,
            gamma=algo_cfg.gamma,
            lam=algo_cfg.lam,
            clip=algo_cfg.clip_ratio,
        )
    )
    if args.live and algo_cfg.num_roll_out is not None and int(args.rollout_parallel) > 1:
        print("[Live] Showing one rollout per epoch; remaining rollouts run in parallel without render.")

    ep_returns: List[float] = []
    ep_costs: List[float] = []
    last_ckpt_path: Path | None = None
    start_epoch = 1
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Use only one of --resume_checkpoint or --init_checkpoint.")
    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint).resolve()
        ckpt = torch.load(ckpt_path, map_location=args.device)
        agent.pi.load_state_dict(ckpt["pi"])
        agent.v_r.load_state_dict(ckpt["v_r"])
        agent.v_c.load_state_dict(ckpt["v_c"])
        print(f"[Init] Loaded checkpoint weights from {ckpt_path}")
    elif args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint).resolve()
        ckpt = torch.load(ckpt_path, map_location=args.device)
        agent.pi.load_state_dict(ckpt["pi"])
        agent.v_r.load_state_dict(ckpt["v_r"])
        agent.v_c.load_state_dict(ckpt["v_c"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        last_ckpt_path = ckpt_path
        print(f"[Resume] Loaded checkpoint {ckpt_path} (start_epoch={start_epoch})")
        if start_epoch > epochs:
            print(f"[Resume] start_epoch ({start_epoch}) > epochs ({epochs}); nothing to train.")
            return

    for ep in range(start_epoch, epochs + 1):
        if algo_cfg.num_roll_out is not None and int(args.rollout_parallel) > 1:
            rollout_device = args.device
            if args.rollout_parallel > 1 and args.device != "cpu":
                rollout_device = "cpu"
                print("[Rollout] Parallel rollout uses CPU workers; switching device to cpu.")

            state_dicts = {
                "pi": {k: v.detach().cpu() for k, v in agent.pi.state_dict().items()},
                "v_r": {k: v.detach().cpu() for k, v in agent.v_r.state_dict().items()},
                "v_c": {k: v.detach().cpu() for k, v in agent.v_c.state_dict().items()},
            }
            algo_cfg_dict = asdict(algo_cfg)
            rollout_total = int(algo_cfg.num_roll_out)
            seeds = [seed + 1000 * ep + i for i in range(rollout_total)]
            local_rollouts = 1 if args.live and rollout_total > 0 else 0
            local_seeds = seeds[:local_rollouts]
            worker_seeds = seeds[local_rollouts:]
            worker_args = [
                {
                    "cfg": cfg,
                    "algo_cfg": algo_cfg_dict,
                    "state_dicts": state_dicts,
                    "seed": s,
                    "max_steps": algo_cfg.max_ep_len,
                    "device": rollout_device,
                }
                for s in worker_seeds
            ]

            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(args.rollout_parallel)) as pool:
                summaries = pool.map(_rollout_worker, worker_args) if worker_args else []

            ep_rets = []
            ep_costs_epoch = []
            ep_lens = []
            for s in local_seeds:
                summ = _collect_rollout_local(trainer, max_steps=algo_cfg.max_ep_len, seed=s)
                ep_rets.append(summ["ep_ret"])
                ep_costs_epoch.append(summ["ep_cost"])
                ep_lens.append(summ["ep_len"])
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

            collect_stats = {
                "EpRetMean": float(np.mean(ep_rets)) if ep_rets else 0.0,
                "EpLenMean": float(np.mean(ep_lens)) if ep_lens else 0.0,
            }
            if ep_costs_epoch:
                C = np.stack(ep_costs_epoch, axis=0)
                for i in range(C.shape[1]):
                    collect_stats[f"EpCost{i}Mean"] = float(np.mean(C[:, i]))
        else:
            collect_stats = trainer.collect_epoch(seed=seed + ep)
        train_stats = trainer.train_epoch()
        ep_returns.append(float(collect_stats["EpRetMean"]))
        if "EpCost0Mean" in collect_stats:
            ep_costs.append(float(collect_stats["EpCost0Mean"]))
        else:
            ep_costs.append(0.0)

        line = f"Epoch {ep:03d} | Ret {collect_stats['EpRetMean']:.2f} | Len {collect_stats['EpLenMean']:.1f}"
        if "EpCost0Mean" in collect_stats:
            line += f" | C0 {collect_stats['EpCost0Mean']:.2f}"
        if "Jc_0" in train_stats:
            line += f" | Jc0 {train_stats['Jc_0']:.2f}"
        if "EpCost1Mean" in collect_stats:
            line += f" | C1 {collect_stats['EpCost1Mean']:.2f}"
        line += f" | KL {train_stats['kl']:.4f} | Pen {train_stats['penalty_sum']:.3f}"
        if "adv_c_term_sum" in train_stats:
            line += f" | AdvC {train_stats['adv_c_term_sum']:.3f}"
        if "constraint_violation_sum" in train_stats:
            line += f" | CViol {train_stats['constraint_violation_sum']:.3f}"
        if "pi_early_stop_iter" in train_stats:
            line += f" | early_stop@{train_stats['pi_early_stop_iter']}"
        print(line)

        if args.checkpoint_every > 0 and (ep % args.checkpoint_every == 0):
            ckpt_path = save_checkpoint(save_dir, ep, agent, algo_cfg, wrapped.obs_dim(), wrapped.act_dim(), len(cmdp_cfg.cost_limits))
            save_metric_plot(save_dir, ep_returns, "returns", "EpRetMean")
            if ep_costs:
                save_metric_plot(save_dir, ep_costs, "costs", "EpCost0Mean")
            last_ckpt_path = ckpt_path
            print(f"  [Saved] {ckpt_path}")

    if args.eval_episodes > 0:
        eval_device = args.device
        if args.eval_parallel > 1 and args.device != "cpu":
            eval_device = "cpu"
            print("[Eval] Parallel eval uses CPU workers; switching device to cpu.")

        state_dicts = {
            "pi": {k: v.detach().cpu() for k, v in agent.pi.state_dict().items()},
            "v_r": {k: v.detach().cpu() for k, v in agent.v_r.state_dict().items()},
            "v_c": {k: v.detach().cpu() for k, v in agent.v_c.state_dict().items()},
        }
        algo_cfg_dict = asdict(algo_cfg)
        seeds = [seed + 100000 + epochs * 1000 + i for i in range(int(args.eval_episodes))]
        worker_args = [
            {
                "cfg": cfg,
                "algo_cfg": algo_cfg_dict,
                "state_dicts": state_dicts,
                "seed": s,
                "max_steps": eval_max_steps,
                "device": eval_device,
            }
            for s in seeds
        ]

        if args.eval_parallel > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(args.eval_parallel)) as pool:
                summaries = pool.map(_eval_worker, worker_args)
        else:
            summaries = [_eval_worker(arg) for arg in worker_args]

        eval_ret = float(np.mean([s["EpRet"] for s in summaries])) if summaries else 0.0
        eval_cost = float(np.mean([s["EpCost0"] for s in summaries])) if summaries else 0.0
        eval_len = float(np.mean([s["EpLen"] for s in summaries])) if summaries else 0.0
        eval_term = float(np.mean([s["Terminated"] for s in summaries])) if summaries else 0.0
        eval_summary = {
            "num_episodes": int(len(summaries)),
            "EpRetMean": float(eval_ret),
            "EpCost0Mean": float(eval_cost),
            "EpLenMean": float(eval_len),
            "TermRate": float(eval_term),
            "episodes": summaries,
        }
        with open(save_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"[Eval] n={len(summaries)} Ret={eval_ret:.2f} Cost0={eval_cost:.2f} Len={eval_len:.1f} TermRate={eval_term:.2f}")
        print(f"[Eval] Saved summary to {save_dir / 'eval_summary.json'}")
        try:
            eval_env = _build_eval_env(cfg, render_mode="rgb_array")
            eval_agent = P3OAgent(
                obs_dim=eval_env.obs_dim(),
                act_dim=eval_env.act_dim(),
                num_costs=len(eval_env.cfg.cost_limits),
                cfg=algo_cfg,
                device=eval_device,
            )
            eval_agent.pi.load_state_dict(state_dicts["pi"])
            eval_agent.v_r.load_state_dict(state_dicts["v_r"])
            eval_agent.v_c.load_state_dict(state_dicts["v_c"])
            eval_agent.pi.eval()
            eval_agent.v_r.eval()
            eval_agent.v_c.eval()
            frames = rollout_frames(eval_env, eval_agent, max_steps=eval_max_steps, seed=seed + 999999, fps=30)
            if last_ckpt_path is not None:
                mp4_name = f"eval_rollout_{last_ckpt_path.stem}.mp4"
            else:
                mp4_name = "eval_rollout_final.mp4"
            mp4_path = save_dir / mp4_name
            save_frames(frames, mp4_path, fps=30)
            print(f"[Eval] Saved MP4 to {mp4_path}")
        except Exception as exc:
            print(f"[Eval] MP4 save failed: {exc}")

    print(f"Done. Outputs in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
