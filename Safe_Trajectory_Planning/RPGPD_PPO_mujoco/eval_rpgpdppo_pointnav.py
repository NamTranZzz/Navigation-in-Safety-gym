"""
eval_rpgpdppo_pointnav.py

Evaluate a saved checkpoint on Safety-Gymnasium point navigation for RPG-PD PPO.

Usage:
  python eval_rpgpdppo_pointnav.py --config config_pointnav_RPGPD.json --checkpoint runs_pointnav_rpgpd/ckpt_epoch_001.pt
  python eval_rpgpdppo_pointnav.py --config config_pointnav_RPGPD.json --checkpoint runs_pointnav_rpgpd/ckpt_epoch_001.pt --num_episodes 10 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure local repo code is imported before any site-packages copy.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mujoco_env import MujocoPointNavEnv, MujocoPointNavConfig
from cmdp_wrapper import RewardCostWrapper, CMDPConfig
from rpgpd_ppo_agent import RPGPDPPOAgent, RPGPDPPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_pointnav_RPGPD.json", help="Path to JSON config")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda or mps")
    p.add_argument("--seed", type=int, default=0, help="Base seed for evaluation episodes")
    p.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes")
    p.add_argument("--max_steps", type=int, default=None, help="Override eval max steps")
    p.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--save_dir", type=str, default=None, help="Directory to save eval outputs")
    p.add_argument("--save_gif", action="store_true", help="Save one evaluation rollout as GIF")
    p.add_argument("--save_mp4", action="store_true", help="Save one evaluation rollout as MP4")
    p.add_argument("--save_mp4_all", action="store_true", help="Save all evaluation rollouts into a single MP4")
    p.add_argument("--save_mp4_each", action="store_true", help="Save one MP4 per evaluation episode")
    p.add_argument("--fps", type=int, default=30, help="FPS for saved GIF/MP4")
    p.add_argument("--camera_name", type=str, default=None, help="MuJoCo camera name (e.g., topdown)")
    p.add_argument("--camera_id", type=int, default=None, help="MuJoCo camera id (overrides name if set)")
    p.add_argument("--render_width", type=int, default=None, help="Render width for video frames")
    p.add_argument("--render_height", type=int, default=None, help="Render height for video frames")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    # PyTorch 2.6 defaults to weights_only=True; we need full checkpoint dicts here.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_env_and_agent(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    device: str,
    render_mode: str | None = None,
    env_kwargs: Dict[str, Any] | None = None,
) -> Tuple[RewardCostWrapper, RPGPDPPOAgent]:
    env_cfg = MujocoPointNavConfig(**cfg["env"])
    if env_kwargs:
        merged = dict(env_cfg.env_kwargs or {})
        merged.update(env_kwargs)
        env_cfg.env_kwargs = merged
    env = MujocoPointNavEnv(env_cfg, render_mode=render_mode)

    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )
    wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)

    algo_dict = cfg["algo"]
    algo_cfg = RPGPDPPOConfig(
        gamma=float(algo_dict["gamma"]),
        lam=float(algo_dict["lam"]),
        clip_ratio=float(algo_dict["clip_ratio"]),
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
        dual_lr=float(algo_dict.get("dual_lr", 0.5)),
        dual_tau=float(algo_dict.get("dual_tau", 0.0)),
        lambda_init=float(algo_dict.get("lambda_init", 0.0)),
        lambda_max=float(algo_dict.get("lambda_max", 1000.0)),
        dual_scale_one_minus_gamma=bool(algo_dict.get("dual_scale_one_minus_gamma", True)),
    )

    agent = RPGPDPPOAgent(
        obs_dim=wrapped.obs_dim(),
        act_dim=wrapped.act_dim(),
        num_costs=len(cmdp_cfg.cost_limits),
        cfg=algo_cfg,
        device=device,
    )

    ckpt = load_checkpoint(checkpoint_path, device)
    agent.pi.load_state_dict(ckpt["pi"])
    agent.v_r.load_state_dict(ckpt["v_r"])
    agent.v_c.load_state_dict(ckpt["v_c"])
    if "lambda" in ckpt:
        agent.lam = np.asarray(ckpt["lambda"], dtype=np.float32)
    agent.pi.eval()
    agent.v_r.eval()
    agent.v_c.eval()

    return wrapped, agent


@torch.no_grad()
def rollout_episode(
    env: RewardCostWrapper,
    agent: RPGPDPPOAgent,
    max_steps: int,
    seed: int | None = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)

    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0
    terminated = False

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(act)
        ep_ret += float(r)
        ep_cost += np.asarray(info["costs"], dtype=np.float32)
        ep_len += 1
        if terminated or truncated:
            break

    summary = {
        "EpRet": ep_ret,
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpLen": ep_len,
        "Terminated": bool(terminated),
    }
    return summary


@torch.no_grad()
def rollout_frames(
    env: RewardCostWrapper,
    agent: RPGPDPPOAgent,
    max_steps: int,
    seed: int | None,
    fps: int,
    render_kwargs: Dict[str, Any] | None = None,
) -> List[np.ndarray]:
    obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []

    frame = env.render(**(render_kwargs or {}))
    if frame is not None:
        frames.append(np.asarray(frame))

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(act)
        frame = env.render(**(render_kwargs or {}))
        if frame is not None:
            frames.append(np.asarray(frame))
        if terminated or truncated:
            break

    if not frames:
        raise RuntimeError("No frames captured. Ensure render_mode='rgb_array' and env supports rendering.")
    return frames


def _resize_frames(frames: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    from PIL import Image

    resized = []
    for f in frames:
        img = Image.fromarray(f)
        img = img.resize((width, height), resample=Image.LANCZOS)
        resized.append(np.asarray(img))
    return resized


def save_frames(frames: List[np.ndarray], save_path: Path, fps: int, width: int | None = None, height: int | None = None) -> None:
    import imageio.v2 as imageio

    if width is not None and height is not None:
        frames = _resize_frames(frames, width, height)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), frames, fps=fps)


def _concat_frames(all_frames: List[List[np.ndarray]], pad_frames: int = 10) -> List[np.ndarray]:
    merged: List[np.ndarray] = []
    for frames in all_frames:
        merged.extend(frames)
        if frames and pad_frames > 0:
            merged.extend([frames[-1]] * pad_frames)
    return merged


def eval_worker(args: Tuple[str, str, int, int, str]) -> Dict[str, Any]:
    config_path, checkpoint_path, seed, max_steps, device = args
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    env, agent = build_env_and_agent(cfg, checkpoint_path, device)
    summ = rollout_episode(env, agent, max_steps=max_steps, seed=seed, deterministic=True)
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

    env_cfg = MujocoPointNavConfig(**cfg["env"])
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

    print(
        f"[Eval] episodes={len(summaries)} return_mean={ep_rets.mean():.2f} "
        f"return_std={ep_rets.std():.2f} cost_mean={ep_costs.mean():.2f} "
        f"len_mean={ep_lens.mean():.1f} term_rate={term_rate:.2f}"
    )

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

    if args.save_gif or args.save_mp4 or args.save_mp4_all or args.save_mp4_each:
        env_kwargs: Dict[str, Any] = {}
        if args.camera_id is not None:
            env_kwargs["camera_id"] = int(args.camera_id)
        elif args.camera_name:
            env_kwargs["camera_name"] = str(args.camera_name)
        render_kwargs: Dict[str, Any] = {}
        if args.render_width is not None:
            render_kwargs["width"] = int(args.render_width)
        if args.render_height is not None:
            render_kwargs["height"] = int(args.render_height)
        env, agent = build_env_and_agent(cfg, args.checkpoint, device, render_mode="rgb_array", env_kwargs=env_kwargs)
        ckpt_stem = Path(args.checkpoint).stem
        if args.save_gif or args.save_mp4:
            frames = rollout_frames(env, agent, max_steps=max_steps, seed=args.seed, fps=args.fps, render_kwargs=render_kwargs)
            if args.save_gif:
                out = save_dir / f"eval_rollout_{ckpt_stem}.gif"
                save_frames(frames, out, fps=args.fps, width=args.render_width, height=args.render_height)
                print(f"  [Saved] {out}")
            if args.save_mp4:
                out = save_dir / f"eval_rollout_{ckpt_stem}.mp4"
                save_frames(frames, out, fps=args.fps, width=args.render_width, height=args.render_height)
                print(f"  [Saved] {out}")
        if args.save_mp4_all:
            all_frames = []
            for s in seeds:
                frames = rollout_frames(env, agent, max_steps=max_steps, seed=s, fps=args.fps, render_kwargs=render_kwargs)
                all_frames.append(frames)
            merged = _concat_frames(all_frames, pad_frames=10)
            out = save_dir / f"eval_rollout_all_{ckpt_stem}.mp4"
            save_frames(merged, out, fps=args.fps, width=args.render_width, height=args.render_height)
            print(f"  [Saved] {out}")
        if args.save_mp4_each:
            for s in seeds:
                frames = rollout_frames(env, agent, max_steps=max_steps, seed=s, fps=args.fps, render_kwargs=render_kwargs)
                out = save_dir / f"eval_rollout_{ckpt_stem}_seed_{s}.mp4"
                save_frames(frames, out, fps=args.fps, width=args.render_width, height=args.render_height)
                print(f"  [Saved] {out}")


if __name__ == "__main__":
    main()
