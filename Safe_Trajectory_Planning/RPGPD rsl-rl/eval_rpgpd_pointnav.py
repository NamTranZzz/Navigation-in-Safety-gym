from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tensordict import TensorDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rsl-rl RPGPD checkpoint on Safety-Gymnasium point navigation.")
    parser.add_argument("--config", type=str, default="config_pointnav_RPGPD_DynObs.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to rsl-rl checkpoint (e.g., model_149.pt).")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions (default is deterministic).")
    parser.add_argument("--save_mp4", type=str, default=None, help="Optional MP4 output path for the first episode.")
    parser.add_argument("--save_mp4_all", type=str, default=None, help="Optional MP4 output path for all episodes.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera_name", type=str, default=None, help="MuJoCo camera name.")
    parser.add_argument("--camera_id", type=int, default=None, help="MuJoCo camera id (overrides camera_name).")
    parser.add_argument("--camera_topdown", action="store_true", help="Auto-select a top-down camera if available.")
    parser.add_argument("--render_width", type=int, default=None, help="Render width for video.")
    parser.add_argument("--render_height", type=int, default=None, help="Render height for video.")
    parser.add_argument("--mp4_macro_block_size", type=int, default=1, help="MP4 macro block size (1 keeps exact size).")
    parser.add_argument("--mp4_crf", type=int, default=18, help="MP4 quality CRF (lower is better).")
    parser.add_argument("--mp4_preset", type=str, default="slow", help="MP4 encoder preset.")
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
            print("[Warn] MPS is not available. Falling back to cpu.")
            return "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available. Falling back to cpu.")
        return "cpu"
    return dev


def make_policy(
    config: Dict[str, Any],
    checkpoint: str,
    device: str,
    seed: int,
):
    from cmdp_wrapper import CMDPConfig, RewardCostWrapper
    from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv
    from rpgpd_agent import RslRPGPDConfig, SafetyGymVecEnv, build_train_cfg

    try:
        from rsl_rl.runners import OnPolicyRunner
    except Exception as exc:
        raise ImportError("rsl-rl is required for evaluation. Install `rsl-rl-lib`.") from exc

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
        cost_limits=tuple(cmdp_cfg.get("cost_limits", [25.0])),
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=tuple(cmdp_cfg.get("cost_scales", [1.0])),
    )

    def env_factory() -> RewardCostWrapper:
        env = MujocoPointNavEnv(mujoco_cfg, render_mode=None)
        return RewardCostWrapper(env, cfg=reward_cost_cfg)

    vec_env = SafetyGymVecEnv(
        env_factory=env_factory,
        num_envs=1,
        device=device,
        seed=seed,
        gamma=float(algo_cfg.get("gamma", 0.99)),
    )

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
        hidden_dims=tuple(algo_cfg.get("hidden_dims", [256, 256, 256])),
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


def rollout_episode(
    env: Any,
    policy: Any,
    device: str,
    max_steps: int,
    seed: int,
    stochastic: bool,
    save_frames: bool,
    render_kwargs: Dict[str, Any] | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
):
    obs_np, info = env.reset(seed=seed)
    ep_ret = 0.0
    ep_cost = np.zeros((len(info.get("cost_limits", [0.0])),), dtype=np.float32)
    frames = []
    if save_frames:
        frame = env.render(**(render_kwargs or {}))
        if frame is not None:
            f0 = _to_uint8_frame(frame)
            if expected_width is not None and expected_height is not None:
                h, w = int(f0.shape[0]), int(f0.shape[1])
                if w != int(expected_width) or h != int(expected_height):
                    print(
                        f"[Video] Rendered source is {w}x{h}, requested {expected_width}x{expected_height}. "
                        "Backend likely ignores size args."
                    )
                else:
                    print(f"[Video] Rendered source size: {w}x{h}")
            else:
                print(f"[Video] Rendered source size: {int(f0.shape[1])}x{int(f0.shape[0])}")
            frames.append(f0)

    for t in range(max_steps):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)
        obs_td = TensorDict({"policy": obs}, batch_size=[1], device=device)
        with torch.no_grad():
            act = policy(obs_td, stochastic_output=bool(stochastic))
        act_np = act.squeeze(0).detach().cpu().numpy()

        obs_np, rew, terminated, truncated, info = env.step(act_np)
        ep_ret += float(rew)
        costs = np.asarray(info.get("costs", [0.0]), dtype=np.float32).reshape(-1)
        if costs.shape[0] == ep_cost.shape[0]:
            ep_cost += costs

        if save_frames:
            frame = env.render(**(render_kwargs or {}))
            if frame is not None:
                frames.append(_to_uint8_frame(frame))

        if bool(terminated or truncated):
            return {
                "EpRet": ep_ret,
                "EpLen": t + 1,
                "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
                "EpCostVec": ep_cost.copy(),
                "Frames": frames,
            }

    return {
        "EpRet": ep_ret,
        "EpLen": max_steps,
        "EpCost0": float(ep_cost[0]) if len(ep_cost) > 0 else 0.0,
        "EpCostVec": ep_cost.copy(),
        "Frames": frames,
    }


def _resize_frames(frames: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    if not frames:
        return frames
    from PIL import Image

    out: list[np.ndarray] = []
    for fr in frames:
        img = Image.fromarray(np.asarray(fr))
        if img.width != int(width) or img.height != int(height):
            img = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        out.append(np.asarray(img))
    return out


def _to_uint8_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if maxv <= 1.0 + 1e-6:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_mp4(
    path: str,
    frames: list[np.ndarray],
    fps: int,
    width: int | None = None,
    height: int | None = None,
    mp4_macro_block_size: int = 1,
    mp4_crf: int = 18,
    mp4_preset: str = "slow",
) -> None:
    if len(frames) == 0:
        print("[Warn] No frames available for MP4 export.")
        return
    import imageio.v2 as imageio

    if width is not None and height is not None and frames:
        src_h, src_w = int(frames[0].shape[0]), int(frames[0].shape[1])
        if src_w != int(width) or src_h != int(height):
            print(
                f"[Video] Keeping native frame size {src_w}x{src_h} for MP4 to avoid quality loss from upscaling "
                f"to {width}x{height}."
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(out),
        fps=int(fps),
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=int(mp4_macro_block_size),
        output_params=["-crf", str(int(mp4_crf)), "-preset", str(mp4_preset)],
    ) as writer:
        for fr in frames:
            writer.append_data(_to_uint8_frame(fr))
    print(f"[Eval] Saved MP4: {out}")


def concat_episodes(all_episode_frames: list[list[np.ndarray]], pad_frames: int = 10) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for idx, frs in enumerate(all_episode_frames):
        if len(frs) == 0:
            continue
        merged.extend(frs)
        if idx < len(all_episode_frames) - 1 and pad_frames > 0:
            black = np.zeros_like(frs[-1])
            for _ in range(int(pad_frames)):
                merged.append(black)
    return merged


def resolve_output_path(output_arg: str, checkpoint_path: str, default_name: str) -> str:
    ckpt_dir = Path(checkpoint_path).resolve().parent
    p = Path(output_arg).expanduser() if output_arg else Path(default_name)
    if p.is_absolute():
        return str(p)
    # Keep all relative outputs in checkpoint folder to simplify result management.
    return str(ckpt_dir / p.name)


def _find_mj_model(obj: Any, max_depth: int = 6) -> Any | None:
    cur = obj
    for _ in range(max_depth):
        if cur is None:
            return None
        model = getattr(cur, "model", None)
        if model is not None and hasattr(model, "ncam"):
            return model
        cur = getattr(cur, "env", None) or getattr(cur, "_env", None) or getattr(cur, "unwrapped", None)
    return None


def _camera_name(model: Any, cam_id: int) -> str:
    try:
        cam_obj = model.cam(cam_id)
        return str(cam_obj.name)
    except Exception:
        try:
            start = int(model.name_camadr[cam_id])
            name_bytes = model.names[start:]
            if isinstance(name_bytes, bytes):
                return name_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            return str(name_bytes)
        except Exception:
            return ""


def _pick_topdown_camera_id(env: Any) -> int | None:
    model = _find_mj_model(env)
    if model is None:
        return None
    keys = ("top", "down", "overhead", "bird")
    ncam = int(getattr(model, "ncam", 0))
    for cid in range(ncam):
        name = _camera_name(model, cid).lower()
        if any(k in name for k in keys):
            return int(cid)
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    seed = int(args.seed)
    set_seed(seed)

    policy = make_policy(
        config=cfg,
        checkpoint=args.checkpoint,
        device=device,
        seed=seed,
    )

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
    reward_cost_cfg = CMDPConfig(
        cost_limits=tuple(cmdp_cfg.get("cost_limits", [25.0])),
        reward_scale=float(cmdp_cfg.get("reward_scale", 1.0)),
        cost_scales=tuple(cmdp_cfg.get("cost_scales", [1.0])),
    )

    render_mode = "rgb_array" if (args.save_mp4 or args.save_mp4_all) else None
    env_kwargs = dict(mujoco_cfg.env_kwargs or {})
    if args.camera_name is not None:
        env_kwargs["camera_name"] = str(args.camera_name)
    if args.camera_id is not None:
        env_kwargs["camera_id"] = int(args.camera_id)
    if args.render_width is not None:
        env_kwargs["width"] = int(args.render_width)
    if args.render_height is not None:
        env_kwargs["height"] = int(args.render_height)
    mujoco_cfg.env_kwargs = env_kwargs

    eval_env = RewardCostWrapper(MujocoPointNavEnv(mujoco_cfg, render_mode=render_mode), cfg=reward_cost_cfg)
    render_kwargs: Dict[str, Any] = {}
    if args.render_width is not None:
        render_kwargs["width"] = int(args.render_width)
    if args.render_height is not None:
        render_kwargs["height"] = int(args.render_height)
    if args.camera_id is not None:
        render_kwargs["camera_id"] = int(args.camera_id)
    elif args.camera_name is not None:
        render_kwargs["camera_name"] = str(args.camera_name)
    elif args.camera_topdown:
        top_id = _pick_topdown_camera_id(eval_env)
        if top_id is not None:
            render_kwargs["camera_id"] = int(top_id)
            print(f"[Eval] Using top-down camera_id={top_id}")

    summaries = []
    first_frames = []
    all_frames: list[list[np.ndarray]] = []
    for i in range(int(args.num_episodes)):
        ep_seed = seed + i
        out = rollout_episode(
            env=eval_env,
            policy=policy,
            device=device,
            max_steps=max_steps,
            seed=ep_seed,
            stochastic=bool(args.stochastic),
            save_frames=bool(args.save_mp4_all or (args.save_mp4 and i == 0)),
            render_kwargs=render_kwargs,
            expected_width=args.render_width,
            expected_height=args.render_height,
        )
        summaries.append(out)
        if i == 0:
            first_frames = out["Frames"]
        if args.save_mp4_all:
            all_frames.append(out["Frames"])
        print(
            f"[Eval] ep={i:03d} seed={ep_seed} ret={out['EpRet']:.3f} "
            f"len={out['EpLen']} cost0={out['EpCost0']:.3f}"
        )

    eval_env.close()

    rets = np.asarray([s["EpRet"] for s in summaries], dtype=np.float32)
    lens = np.asarray([s["EpLen"] for s in summaries], dtype=np.float32)
    c0 = np.asarray([s["EpCost0"] for s in summaries], dtype=np.float32)
    print(
        "[Eval] summary "
        f"episodes={len(summaries)} "
        f"return_mean={rets.mean():.3f} return_std={rets.std():.3f} "
        f"len_mean={lens.mean():.2f} "
        f"cost0_mean={c0.mean():.3f} cost0_std={c0.std():.3f}"
    )

    if args.save_mp4:
        out_first = resolve_output_path(args.save_mp4, args.checkpoint, "eval_first_episode.mp4")
        save_mp4(
            out_first,
            first_frames,
            fps=args.fps,
            width=args.render_width,
            height=args.render_height,
            mp4_macro_block_size=args.mp4_macro_block_size,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )
    if args.save_mp4_all:
        out_all = resolve_output_path(args.save_mp4_all, args.checkpoint, "eval_all_episodes.mp4")
        merged = concat_episodes(all_frames, pad_frames=max(1, int(args.fps // 3)))
        save_mp4(
            out_all,
            merged,
            fps=args.fps,
            width=args.render_width,
            height=args.render_height,
            mp4_macro_block_size=args.mp4_macro_block_size,
            mp4_crf=args.mp4_crf,
            mp4_preset=args.mp4_preset,
        )


if __name__ == "__main__":
    main()
