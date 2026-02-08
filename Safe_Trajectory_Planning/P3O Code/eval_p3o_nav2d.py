"""
eval_p3o_nav2d.py

Evaluate a saved checkpoint across multiple random initializations.

Usage:
  python eval_p3o_nav2d.py --config config_nav_p3o.json --checkpoint runs_nav2d/ckpt_epoch_001.pt
  python eval_p3o_nav2d.py --config config_nav_p3o.json --checkpoint runs_nav2d/ckpt_epoch_001.pt --num_episodes 10 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from nav_env import Nav2DEnv, Nav2DConfig
from cmdp_wrapper import RewardCostWrapper, CMDPConfig, reward_progress_to_goal, cost_distance_to_obstacles
from p3o_agent import P3OAgent, P3OConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_nav_p3o.json", help="Path to JSON config")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--seed", type=int, default=0, help="Base seed for evaluation episodes")
    p.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes")
    p.add_argument("--max_steps", type=int, default=None, help="Override eval max steps")
    p.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--save_dir", type=str, default=None, help="Directory to save eval outputs")
    p.add_argument("--save_gif", action="store_true", help="Save one evaluation rollout as GIF")
    p.add_argument("--save_mp4", action="store_true", help="Save one evaluation rollout as MP4 (requires ffmpeg)")
    p.add_argument("--save_mp4_all", action="store_true", help="Save all evaluation rollouts into a single MP4")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_and_agent(cfg: Dict[str, Any], checkpoint_path: str, device: str) -> Tuple[RewardCostWrapper, P3OAgent]:
    env_cfg = Nav2DConfig(**cfg["env"])
    env = Nav2DEnv(env_cfg)

    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        distance_threshold=float(cfg["cmdp"].get("distance_threshold", 1.0)),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(
        env,
        cfg=cmdp_cfg,
        reward_fn=reward_progress_to_goal,
        cost_fns=(lambda info, thr=cmdp_cfg.distance_threshold: cost_distance_to_obstacles(info, threshold=thr),),
    )

    algo_dict = cfg["algo"]
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
        hidden_sizes=tuple(int(x) for x in algo_dict["hidden_sizes"]),
        entropy_coef=float(algo_dict.get("entropy_coef", 0.0)),
        normalize_advantages=bool(algo_dict.get("normalize_advantages", True)),
    )

    agent = P3OAgent(obs_dim=wrapped.obs_dim(), act_dim=wrapped.act_dim(), num_costs=len(cmdp_cfg.cost_limits), cfg=algo_cfg, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.pi.load_state_dict(ckpt["pi"])
    agent.v_r.load_state_dict(ckpt["v_r"])
    agent.v_c.load_state_dict(ckpt["v_c"])
    agent.pi.eval()
    agent.v_r.eval()
    agent.v_c.eval()

    return wrapped, agent


@torch.no_grad()
def rollout_episode(env: RewardCostWrapper, agent: P3OAgent, max_steps: int, seed: int | None = None, deterministic: bool = True):
    obs, info = env.reset(seed=seed)
    traj: List[Dict[str, Any]] = []

    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(act)
        ep_ret += float(r)
        ep_cost += np.asarray(info["costs"], dtype=np.float32)

        traj.append({
            "t": int(info["t"]),
            "agent_pos": np.asarray(info["agent_pos"]).copy(),
            "agent_vel": np.asarray(info["agent_vel"]).copy(),
            "goal_pos": np.asarray(info["goal_pos"]).copy(),
            "pillars_xy": np.asarray(info["pillars_xy"]).copy(),
            "pillars_r": np.asarray(info["pillars_r"]).copy(),
            "hazards_static_xy": np.asarray(info["hazards_static_xy"]).copy(),
            "hazards_static_r": np.asarray(info["hazards_static_r"]).copy(),
            "dynamic_obstacles_xy": np.asarray(info["dynamic_obstacles_xy"]).copy(),
            "dynamic_obstacles_v": np.asarray(info["dynamic_obstacles_v"]).copy(),
            "dynamic_obstacles_r": np.asarray(info["dynamic_obstacles_r"]).copy(),
            "in_hazard": bool(info["in_hazard"]),
            "collided_pillar": bool(info["collided_pillar"]),
            "collided_dynamic": bool(info.get("collided_dynamic", False)),
            "wall_hit": bool(info["wall_hit"]),
            "reward": float(r),
            "costs": np.asarray(info["costs"], dtype=np.float32).copy(),
            "dist_to_goal": float(np.linalg.norm(info["goal_pos"] - info["agent_pos"])),
            "world_size": float(info["world_size"]),
            "agent_radius": float(info["agent_radius"]),
            "goal_radius": float(info["goal_radius"]),
        })

        if terminated or truncated:
            break

    summary = {
        "EpRet": ep_ret,
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpLen": len(traj),
        "Terminated": bool(traj and traj[-1]["dist_to_goal"] <= traj[-1]["goal_radius"]),
    }
    return traj, summary


def animate_trajectory(traj: List[Dict[str, Any]], save_path: str | None = None, fps: int = 30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    if len(traj) == 0:
        return

    world_size = float(traj[0]["world_size"])
    half = world_size / 2.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_title("Nav2D evaluation rollout (P3O policy)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], linewidth=1.2)

    agent_r = float(traj[0]["agent_radius"])
    goal_r = float(traj[0]["goal_radius"])

    goal_patch = Circle((0, 0), goal_r, fill=True, alpha=0.35)
    agent_patch = Circle((0, 0), agent_r, fill=True, alpha=0.9)
    ax.add_patch(goal_patch)
    ax.add_patch(agent_patch)

    pillar_patches = []
    for xy, r in zip(traj[0]["pillars_xy"], traj[0]["pillars_r"]):
        p = Circle(tuple(xy), float(r), fill=True, alpha=0.8)
        ax.add_patch(p)
        pillar_patches.append(p)

    haz_s_patches = []
    for xy, r in zip(traj[0]["hazards_static_xy"], traj[0]["hazards_static_r"]):
        p = Circle(tuple(xy), float(r), fill=True, alpha=0.5)
        ax.add_patch(p)
        haz_s_patches.append(p)

    dyn_obs_patches = []
    for xy, r in zip(traj[0]["dynamic_obstacles_xy"], traj[0]["dynamic_obstacles_r"]):
        p = Circle(tuple(xy), float(r), fill=True, alpha=0.8, facecolor="#003366")
        ax.add_patch(p)
        dyn_obs_patches.append(p)

    traj_line, = ax.plot([], [], linewidth=1.4)
    agent_quiv = ax.quiver([0], [0], [0], [0], angles='xy', scale_units='xy', scale=1.0, width=0.007)
    dh_xy0 = traj[0]["dynamic_obstacles_xy"]
    dh_v0 = traj[0]["dynamic_obstacles_v"]
    if dh_xy0.shape[0] > 0:
        dyn_quiv = ax.quiver(dh_xy0[:, 0], dh_xy0[:, 1], dh_v0[:, 0], dh_v0[:, 1], angles='xy', scale_units='xy', scale=1.0, width=0.004)
    else:
        dyn_quiv = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1.0, width=0.004)
    text = ax.text(-half + 0.2, half - 0.3, "", fontsize=9, va="top",
                   bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))
    goal_text = ax.text(0, 0, "GOAL", fontsize=8, ha="center", va="center")

    xs: List[float] = []
    ys: List[float] = []

    coll_marker, = ax.plot([], [], marker="x", linestyle="", markersize=8)

    cum_r = 0.0
    cum_c0 = 0.0
    last_episode = traj[0].get("episode", 0) if traj else 0

    def init():
        xs.clear()
        ys.clear()
        traj_line.set_data([], [])
        coll_marker.set_data([], [])
        return traj_line, agent_patch, goal_patch, agent_quiv, dyn_quiv, text, coll_marker

    def update(frame: int):
        nonlocal cum_r, cum_c0, dyn_quiv, last_episode
        d = traj[frame]
        if "episode" in d and d["episode"] != last_episode:
            last_episode = d["episode"]
            cum_r = 0.0
            cum_c0 = 0.0
            xs.clear()
            ys.clear()
            traj_line.set_data([], [])
        if frame == 0:
            xs.clear()
            ys.clear()
            traj_line.set_data([], [])
        ap = d["agent_pos"]
        av = d["agent_vel"]
        gp = d["goal_pos"]
        xs.append(float(ap[0]))
        ys.append(float(ap[1]))
        traj_line.set_data(xs, ys)

        agent_patch.center = tuple(ap)
        goal_patch.center = tuple(gp)

        agent_quiv.set_offsets([ap])
        agent_quiv.set_UVC([av[0]], [av[1]])

        dh_xy = d["dynamic_obstacles_xy"]
        dh_v = d["dynamic_obstacles_v"]
        n = min(dh_xy.shape[0], dh_v.shape[0])
        if n > 0:
            dh_xy = dh_xy[:n]
            dh_v = dh_v[:n]
            if dyn_quiv.get_offsets().shape[0] != n:
                dyn_quiv.remove()
                dyn_quiv = ax.quiver(dh_xy[:, 0], dh_xy[:, 1], dh_v[:, 0], dh_v[:, 1], angles='xy', scale_units='xy', scale=1.0, width=0.004)
            else:
                dyn_quiv.set_offsets(dh_xy)
                dyn_quiv.set_UVC(dh_v[:, 0], dh_v[:, 1])
        else:
            dyn_quiv.set_offsets(np.zeros((0, 2)))
            dyn_quiv.set_UVC([], [])

        for p, xy in zip(dyn_obs_patches, dh_xy):
            p.center = tuple(xy)

        cum_r += float(d["reward"])
        if d["costs"].shape[0] > 0:
            cum_c0 += float(d["costs"][0])

        coll = bool(d["in_hazard"] or d["collided_pillar"] or d.get("collided_dynamic", False))
        if coll:
            coll_marker.set_data([ap[0]], [ap[1]])
        else:
            coll_marker.set_data([], [])

        dist = d["dist_to_goal"]
        reached = dist <= d["goal_radius"]
        ep_text = f"ep={d['episode']}  " if "episode" in d else ""
        text.set_text(
            f"{ep_text}t={d['t']}  dist={dist:.2f}\n"
            f"ret={cum_r:.2f}  c0={cum_c0:.1f}\n"
            f"in_hazard={d['in_hazard']}  hit_pillar={d['collided_pillar']}\n"
            f"hit_dynamic={d['collided_dynamic']}  wall_hit={d['wall_hit']}"
        )
        goal_text.set_position(tuple(gp))
        goal_text.set_color("green" if reached else "black")

        return traj_line, agent_patch, goal_patch, agent_quiv, dyn_quiv, text, coll_marker, goal_text

    anim = FuncAnimation(fig, update, frames=len(traj), init_func=init, interval=1000 // fps, blit=False)

    if save_path is not None:
        save_path = str(save_path)
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            anim.save(save_path + ".gif", writer="pillow", fps=fps)
    else:
        plt.show()

    plt.close(fig)


def _concat_trajectories(trajs: List[List[Dict[str, Any]]], pad_frames: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep_idx, traj in enumerate(trajs):
        for step in traj:
            d = dict(step)
            d["episode"] = int(ep_idx)
            out.append(d)
        if pad_frames > 0 and traj:
            pad = dict(traj[-1])
            pad["episode"] = int(ep_idx)
            pad["reward"] = 0.0
            pad["costs"] = np.zeros_like(traj[-1]["costs"], dtype=np.float32)
            for _ in range(pad_frames):
                out.append(dict(pad))
    return out


def eval_worker(args: Tuple[str, str, int, int, str]) -> Dict[str, Any]:
    config_path, checkpoint_path, seed, max_steps, device = args
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    env, agent = build_env_and_agent(cfg, checkpoint_path, device)
    _, summ = rollout_episode(env, agent, max_steps=max_steps, seed=seed, deterministic=True)
    summ["seed"] = int(seed)
    return summ


def main():
    args = parse_args()
    cfg = load_config(args.config)

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.checkpoint).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel > 1 and args.device != "cpu":
        print("[Eval] Parallel eval uses CPU workers; switching device to cpu.")
        device = "cpu"
    else:
        device = args.device

    env_cfg = Nav2DConfig(**cfg["env"])
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env_cfg.max_steps)

    seeds = [args.seed + i for i in range(int(args.num_episodes))]

    if args.parallel > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.parallel)) as pool:
            summaries = pool.map(eval_worker, [(args.config, args.checkpoint, s, max_steps, device) for s in seeds])
    else:
        summaries = [eval_worker((args.config, args.checkpoint, s, max_steps, device)) for s in seeds]

    ep_rets = np.asarray([s["EpRet"] for s in summaries], dtype=np.float32)
    ep_costs = np.asarray([s.get("EpCost0", 0.0) for s in summaries], dtype=np.float32)
    ep_lens = np.asarray([s["EpLen"] for s in summaries], dtype=np.float32)
    term_rate = float(np.mean([s.get("Terminated", False) for s in summaries]))

    print(f"[Eval] episodes={len(summaries)} return_mean={ep_rets.mean():.2f} return_std={ep_rets.std():.2f} cost_mean={ep_costs.mean():.2f} len_mean={ep_lens.mean():.1f} term_rate={term_rate:.2f}")

    out_summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_episodes": len(summaries),
        "seeds": seeds,
        "return_mean": float(ep_rets.mean()),
        "return_std": float(ep_rets.std()),
        "cost_mean": float(ep_costs.mean()),
        "len_mean": float(ep_lens.mean()),
        "terminated_rate": term_rate,
        "episodes": summaries,
    }
    with open(save_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

    if args.save_gif or args.save_mp4 or args.save_mp4_all:
        env, agent = build_env_and_agent(cfg, args.checkpoint, device)
        if args.save_gif or args.save_mp4:
            traj, _ = rollout_episode(env, agent, max_steps=max_steps, seed=args.seed, deterministic=True)
            if args.save_gif:
                out = save_dir / "eval_rollout.gif"
                animate_trajectory(traj, save_path=str(out), fps=30)
                print(f"  [Saved] {out}")
            if args.save_mp4:
                out = save_dir / "eval_rollout.mp4"
                animate_trajectory(traj, save_path=str(out), fps=30)
                print(f"  [Saved] {out}")
        if args.save_mp4_all:
            trajs = []
            for s in seeds:
                traj, _ = rollout_episode(env, agent, max_steps=max_steps, seed=s, deterministic=True)
                trajs.append(traj)
            merged = _concat_trajectories(trajs, pad_frames=10)
            out = save_dir / "eval_rollout_all.mp4"
            animate_trajectory(merged, save_path=str(out), fps=30)
            print(f"  [Saved] {out}")


if __name__ == "__main__":
    main()
