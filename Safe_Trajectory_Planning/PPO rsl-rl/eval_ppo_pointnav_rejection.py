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
    p.add_argument("--seed", type=int, default=None, help="Base seed for evaluation episodes")
    p.add_argument(
        "--random_eval_seeds",
        action="store_true",
        help="Use random per-episode seeds (training-like behavior)",
    )
    p.add_argument(
        "--fixed_eval_seed_range",
        type=str,
        default=None,
        help='Use a fixed deterministic eval seed list, e.g. "1-120" or "1,2,3".',
    )
    p.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes")
    p.add_argument("--max_steps", type=int, default=None, help="Override eval max steps")
    p.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    p.add_argument(
        "--max_attempts",
        type=int,
        default=10000,
        help="Max sampled episodes while collecting accepted ones (cost <= cost limit)",
    )
    p.add_argument("--save_dir", type=str, default=None, help="Directory to save eval outputs")
    p.add_argument("--save_gif", action="store_true", help="Save one evaluation rollout as GIF")
    p.add_argument("--save_mp4", action="store_true", help="Save one evaluation rollout as MP4")
    p.add_argument("--save_mp4_all", action="store_true", help="Save all evaluation rollouts into a single MP4")
    p.add_argument("--save_mp4_each", action="store_true", help="Save one MP4 per evaluation episode")
    p.add_argument("--fps", type=int, default=30, help="FPS for saved GIF/MP4")
    p.add_argument("--camera_name", type=str, default=None, help="MuJoCo camera name (e.g., topdown)")
    p.add_argument("--camera_id", type=int, default=None, help="MuJoCo camera id (overrides name if set)")
    p.add_argument("--camera_topdown", action="store_true", help="Auto-select a top-down MuJoCo camera")
    p.add_argument("--render_width", type=int, default=None, help="Render width for video frames")
    p.add_argument("--render_height", type=int, default=None, help="Render height for video frames")
    p.add_argument("--mp4_macro_block_size", type=int, default=1, help="FFmpeg macro block size for MP4 (1 keeps exact size)")
    p.add_argument("--mp4_crf", type=int, default=18, help="FFmpeg CRF for MP4 quality (lower is better, typical 17-23)")
    p.add_argument("--mp4_preset", type=str, default="slow", help="FFmpeg preset for MP4 (slower usually gives better quality)")
    return p.parse_args()


def parse_seed_spec(spec: str) -> list[int]:
    s = str(spec).strip()
    if not s:
        raise ValueError("empty seed spec")
    if "-" in s and "," not in s:
        lo_s, hi_s = s.split("-", 1)
        lo = int(lo_s.strip())
        hi = int(hi_s.strip())
        if hi < lo:
            raise ValueError(f"invalid seed range: {spec}")
        return [int(x) for x in range(lo, hi + 1)]
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError(f"invalid seed list: {spec}")
    return out


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
    gamma: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
    render_kwargs: Dict[str, Any] | None = None,
) -> List[np.ndarray]:
    obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    ep_ret = 0.0
    disc_cost0 = 0.0
    disc = 1.0

    def _overlay_metrics(frame: np.ndarray, ret_val: float, dcost_val: float) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
                scale = 255.0 if maxv <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        draw = ImageDraw.Draw(img)
        h, w = img.height, img.width
        font_size = max(24, int(h * 0.04))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
        line1 = f"Reward (undisc): {ret_val:.2f}"
        line2 = f"Cost0 (disc): {dcost_val:.2f}"
        x = max(12, int(0.012 * w))
        y = max(22, int(0.04 * h))
        pad = max(8, int(font_size * 0.35))
        line_gap = max(4, int(font_size * 0.2))
        txt1_w, txt1_h = draw.textbbox((0, 0), line1, font=font)[2:]
        txt2_w, txt2_h = draw.textbbox((0, 0), line2, font=font)[2:]
        box_w = max(txt1_w, txt2_w) + 2 * pad
        box_h = txt1_h + txt2_h + line_gap + 2 * pad
        draw.rectangle((x, y, x + box_w, y + box_h), fill=(0, 0, 0))
        draw.text((x + pad, y + pad), line1, fill=(255, 255, 255), font=font)
        draw.text((x + pad, y + pad + txt1_h + line_gap), line2, fill=(255, 255, 255), font=font)
        return np.asarray(img)

    frame = env.render(**(render_kwargs or {}))
    if frame is not None:
        arr = np.asarray(frame)
        if expected_width is not None and expected_height is not None:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if w != int(expected_width) or h != int(expected_height):
                print(
                    f"[Video] Rendered frame is {w}x{h}, not requested {expected_width}x{expected_height}. "
                    "This indicates renderer-side resolution limits."
                )
        frames.append(_overlay_metrics(arr, ep_ret, disc_cost0))

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(act)
        ep_ret += float(r)
        costs = np.asarray(info.get("costs", [0.0]), dtype=np.float32)
        c0 = float(costs[0]) if len(costs) > 0 else 0.0
        disc_cost0 += disc * c0
        disc *= gamma
        frame = env.render(**(render_kwargs or {}))
        if frame is not None:
            frames.append(_overlay_metrics(np.asarray(frame), ep_ret, disc_cost0))
        if terminated or truncated:
            break

    if not frames:
        raise RuntimeError("No frames captured. Ensure render_mode='rgb_array' and env supports rendering.")
    return frames


@torch.no_grad()
def rollout_episode_with_frames(
    env: RewardCostWrapper,
    agent: RPGPDPPOAgent,
    max_steps: int,
    seed: int | None,
    deterministic: bool,
    gamma: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
    render_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    obs, info = env.reset(seed=seed)

    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0
    terminated = False
    disc_cost0 = 0.0
    disc = 1.0
    frames: List[np.ndarray] = []

    def _overlay_metrics(frame: np.ndarray, ret_val: float, dcost_val: float) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
                scale = 255.0 if maxv <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        draw = ImageDraw.Draw(img)
        h, w = img.height, img.width
        font_size = max(24, int(h * 0.04))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
        line1 = f"Reward (undisc): {ret_val:.2f}"
        line2 = f"Cost0 (disc): {dcost_val:.2f}"
        x = max(12, int(0.012 * w))
        y = max(22, int(0.04 * h))
        pad = max(8, int(font_size * 0.35))
        line_gap = max(4, int(font_size * 0.2))
        txt1_w, txt1_h = draw.textbbox((0, 0), line1, font=font)[2:]
        txt2_w, txt2_h = draw.textbbox((0, 0), line2, font=font)[2:]
        box_w = max(txt1_w, txt2_w) + 2 * pad
        box_h = txt1_h + txt2_h + line_gap + 2 * pad
        draw.rectangle((x, y, x + box_w, y + box_h), fill=(0, 0, 0))
        draw.text((x + pad, y + pad), line1, fill=(255, 255, 255), font=font)
        draw.text((x + pad, y + pad + txt1_h + line_gap), line2, fill=(255, 255, 255), font=font)
        return np.asarray(img)

    frame = env.render(**(render_kwargs or {}))
    if frame is not None:
        arr = np.asarray(frame)
        if expected_width is not None and expected_height is not None:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if w != int(expected_width) or h != int(expected_height):
                print(
                    f"[Video] Rendered frame is {w}x{h}, not requested {expected_width}x{expected_height}. "
                    "This indicates renderer-side resolution limits."
                )
        frames.append(_overlay_metrics(arr, ep_ret, disc_cost0))

    for _ in range(max_steps):
        act, _, _, _ = agent.act(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(act)
        ep_ret += float(r)
        costs = np.asarray(info["costs"], dtype=np.float32)
        ep_cost += costs
        c0 = float(costs[0]) if len(costs) > 0 else 0.0
        disc_cost0 += disc * c0
        disc *= gamma
        ep_len += 1
        frame = env.render(**(render_kwargs or {}))
        if frame is not None:
            frames.append(_overlay_metrics(np.asarray(frame), ep_ret, disc_cost0))
        if terminated or truncated:
            break

    if not frames:
        raise RuntimeError("No frames captured. Ensure render_mode='rgb_array' and env supports rendering.")

    summary = {
        "EpRet": ep_ret,
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpLen": ep_len,
        "Terminated": bool(terminated),
    }
    return summary, frames


def _resize_frames(frames: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    from PIL import Image

    resized = []
    for f in frames:
        img = Image.fromarray(f)
        img = img.resize((width, height), resample=Image.LANCZOS)
        resized.append(np.asarray(img))
    return resized


def save_frames(
    frames: List[np.ndarray],
    save_path: Path,
    fps: int,
    width: int | None = None,
    height: int | None = None,
    mp4_macro_block_size: int = 1,
    mp4_crf: int = 18,
    mp4_preset: str = "slow",
) -> None:
    import imageio.v2 as imageio

    if width is not None and height is not None and frames:
        src_h, src_w = int(frames[0].shape[0]), int(frames[0].shape[1])
        if src_w != int(width) or src_h != int(height):
            if save_path.suffix.lower() == ".mp4":
                print(
                    f"[Video] Keeping native frame size {src_w}x{src_h} for MP4 to avoid quality loss from upscaling "
                    f"to {width}x{height}."
                )
            else:
                frames = _resize_frames(frames, width, height)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".mp4":
        # Keep exact render size (e.g., true 1920x1080) and use higher-quality H.264 settings.
        with imageio.get_writer(
            str(save_path),
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=int(mp4_macro_block_size),
            output_params=["-crf", str(int(mp4_crf)), "-preset", str(mp4_preset)],
        ) as writer:
            for f in frames:
                writer.append_data(f)
    else:
        imageio.mimsave(str(save_path), frames, fps=fps)


def _concat_frames(all_frames: List[List[np.ndarray]], pad_frames: int = 10) -> List[np.ndarray]:
    merged: List[np.ndarray] = []
    for frames in all_frames:
        merged.extend(frames)
        if frames and pad_frames > 0:
            merged.extend([frames[-1]] * pad_frames)
    return merged


def _find_mj_model(obj: Any, max_depth: int = 6) -> Any | None:
    queue: List[Any] = [obj]
    seen = set()
    depth = 0
    while queue and depth <= max_depth:
        nxt: List[Any] = []
        for cur in queue:
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            model = getattr(cur, "model", None)
            if model is not None and hasattr(model, "ncam"):
                return model
            sim = getattr(cur, "sim", None)
            if sim is not None and getattr(sim, "model", None) is not None:
                return sim.model
            for attr in ("env", "_env", "unwrapped", "task", "_task", "world", "_world", "builder", "_builder"):
                if hasattr(cur, attr):
                    nxt.append(getattr(cur, attr))
        queue = nxt
        depth += 1
    return None


def _camera_name(model: Any, cam_id: int) -> str:
    try:
        adr = int(model.name_camadr[cam_id])
        raw = model.names[adr:]
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        name = bytes(raw).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        return str(name)
    except Exception:
        return ""


def _pick_topdown_camera_id(env: RewardCostWrapper) -> int | None:
    model = _find_mj_model(env)
    if model is None:
        return None
    ncam = int(getattr(model, "ncam", 0))
    if ncam <= 0:
        return None

    # Prefer cameras explicitly named as top-down; otherwise choose the highest camera by z-position.
    preferred = []
    for cam_id in range(ncam):
        name = _camera_name(model, cam_id).lower()
        if any(k in name for k in ("top", "down", "overhead", "bird", "global")):
            preferred.append(cam_id)
    if preferred:
        return int(max(preferred, key=lambda i: float(model.cam_pos[i][2])))
    return int(max(range(ncam), key=lambda i: float(model.cam_pos[i][2])))


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

    checkpoint_dir = Path(args.checkpoint).resolve().parent
    save_dir = Path(args.save_dir) if args.save_dir else checkpoint_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel > 1 and args.device != "cpu":
        print("[Eval] Parallel eval uses CPU workers; switching device to cpu.")
        device = "cpu"
    else:
        device = args.device

    env_cfg = MujocoPointNavConfig(**cfg["env"])
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env_cfg.max_steps)

    base_seed = 0 if args.seed is None else int(args.seed)

    if args.parallel > 1:
        print("[Eval] Rejection evaluation currently uses sequential sampling; ignoring --parallel.")

    cost_limits = cfg.get("cmdp", {}).get("cost_limits", [float("inf")])
    cost_limit0 = float(cost_limits[0]) if len(cost_limits) > 0 else float("inf")
    target_n = int(args.num_episodes)
    max_attempts = int(args.max_attempts)

    accepted_summaries: List[Dict[str, Any]] = []
    accepted_seeds: List[int] = []
    accepted_frames: List[List[np.ndarray]] = []
    rejected = 0
    attempts = 0
    need_video = args.save_gif or args.save_mp4 or args.save_mp4_all or args.save_mp4_each

    env_kwargs: Dict[str, Any] = {}
    render_kwargs: Dict[str, Any] = {}
    if need_video:
        if args.camera_id is not None:
            env_kwargs["camera_id"] = int(args.camera_id)
        elif args.camera_name:
            env_kwargs["camera_name"] = str(args.camera_name)
        if args.render_width is not None:
            render_kwargs["width"] = int(args.render_width)
            env_kwargs["width"] = int(args.render_width)
        if args.render_height is not None:
            render_kwargs["height"] = int(args.render_height)
            env_kwargs["height"] = int(args.render_height)
        env, agent = build_env_and_agent(cfg, args.checkpoint, device, render_mode="rgb_array", env_kwargs=env_kwargs)
        if args.camera_topdown:
            topdown_id = _pick_topdown_camera_id(env)
            if topdown_id is None:
                print("[Video] Could not auto-select top-down camera; using default render camera.")
            else:
                render_kwargs["camera_id"] = int(topdown_id)
                print(f"[Video] Using auto top-down camera_id={topdown_id}")
    else:
        env, agent = build_env_and_agent(cfg, args.checkpoint, device)

    if args.fixed_eval_seed_range is not None:
        fixed_seeds = parse_seed_spec(args.fixed_eval_seed_range)
        print(
            f"[Seed] fixed_eval_seed_range enabled: candidate seeds={len(fixed_seeds)} "
            f"(first={fixed_seeds[0]}, last={fixed_seeds[-1]}).",
        )
        idx = 0

        def _next_seed() -> int:
            nonlocal idx
            s = int(fixed_seeds[idx % len(fixed_seeds)])
            idx += 1
            return s
    elif args.random_eval_seeds:
        if args.seed is None:
            base_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"[Seed] random_eval_seeds enabled: base_seed={base_seed}")
        rng = np.random.default_rng(base_seed)

        def _next_seed() -> int:
            return int(rng.integers(0, 2**31 - 1))
    else:
        offset = 0

        def _next_seed() -> int:
            nonlocal offset
            s = int(base_seed + offset)
            offset += 1
            return s

    while len(accepted_summaries) < target_n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Reached max_attempts={max_attempts} before collecting {target_n} accepted episodes "
                f"(cost_limit0={cost_limit0}). Accepted={len(accepted_summaries)}, rejected={rejected}."
            )
        s = _next_seed()
        attempts += 1
        if need_video:
            summ, frames = rollout_episode_with_frames(
                env,
                agent,
                max_steps=max_steps,
                seed=s,
                deterministic=True,
                gamma=float(cfg["algo"]["gamma"]),
                expected_width=args.render_width,
                expected_height=args.render_height,
                render_kwargs=render_kwargs,
            )
        else:
            summ = rollout_episode(env, agent, max_steps=max_steps, seed=s, deterministic=True)
        c0 = float(summ.get("EpCost0", 0.0))
        if c0 <= cost_limit0:
            accepted_summaries.append(summ)
            accepted_seeds.append(int(s))
            if need_video:
                accepted_frames.append(frames)
        else:
            rejected += 1

    summaries = accepted_summaries
    seeds = accepted_seeds

    ep_rets = np.asarray([s["EpRet"] for s in summaries], dtype=np.float32)
    ep_costs = np.asarray([s.get("EpCost0", 0.0) for s in summaries], dtype=np.float32)
    ep_lens = np.asarray([s["EpLen"] for s in summaries], dtype=np.float32)
    term_rate = float(np.mean([s.get("Terminated", False) for s in summaries]))

    print(
        f"[Eval-Rejection] accepted={len(summaries)}/{attempts} rejected={rejected} "
        f"cost_limit0={cost_limit0:.2f} return_mean={ep_rets.mean():.2f} "
        f"return_std={ep_rets.std():.2f} cost_mean={ep_costs.mean():.2f} "
        f"len_mean={ep_lens.mean():.1f} term_rate={term_rate:.2f}"
    )

    out_summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_episodes": len(summaries),
        "num_attempts": int(attempts),
        "num_rejected": int(rejected),
        "acceptance_rate": float(len(summaries) / attempts if attempts > 0 else 0.0),
        "cost_limit0": float(cost_limit0),
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

    if need_video:
        ckpt_stem = Path(args.checkpoint).stem
        if args.save_gif or args.save_mp4:
            frames = accepted_frames[0] if len(accepted_frames) > 0 else []
            if args.save_gif:
                out = checkpoint_dir / f"eval_rollout_{ckpt_stem}.gif"
                save_frames(
                    frames,
                    out,
                    fps=args.fps,
                    width=args.render_width,
                    height=args.render_height,
                    mp4_macro_block_size=args.mp4_macro_block_size,
                    mp4_crf=args.mp4_crf,
                    mp4_preset=args.mp4_preset,
                )
                print(f"  [Saved] {out}")
            if args.save_mp4:
                out = checkpoint_dir / f"eval_rollout_{ckpt_stem}.mp4"
                save_frames(
                    frames,
                    out,
                    fps=args.fps,
                    width=args.render_width,
                    height=args.render_height,
                    mp4_macro_block_size=args.mp4_macro_block_size,
                    mp4_crf=args.mp4_crf,
                    mp4_preset=args.mp4_preset,
                )
                print(f"  [Saved] {out}")
        if args.save_mp4_all:
            merged = _concat_frames(accepted_frames, pad_frames=10)
            out = checkpoint_dir / f"eval_rollout_all_{ckpt_stem}.mp4"
            save_frames(
                merged,
                out,
                fps=args.fps,
                width=args.render_width,
                height=args.render_height,
                mp4_macro_block_size=args.mp4_macro_block_size,
                mp4_crf=args.mp4_crf,
                mp4_preset=args.mp4_preset,
            )
            print(f"  [Saved] {out}")
        if args.save_mp4_each:
            for s, frames in zip(seeds, accepted_frames):
                out = checkpoint_dir / f"eval_rollout_{ckpt_stem}_seed_{s}.mp4"
                save_frames(
                    frames,
                    out,
                    fps=args.fps,
                    width=args.render_width,
                    height=args.render_height,
                    mp4_macro_block_size=args.mp4_macro_block_size,
                    mp4_crf=args.mp4_crf,
                    mp4_preset=args.mp4_preset,
                )
                print(f"  [Saved] {out}")


if __name__ == "__main__":
    main()
