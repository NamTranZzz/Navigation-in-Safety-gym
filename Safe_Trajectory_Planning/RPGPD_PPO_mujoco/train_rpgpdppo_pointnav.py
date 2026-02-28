"""
train_rpgpdppo_pointnav.py

Train RPG-PD integrated with PPO on MuJoCo Safety-Gymnasium point navigation.

Usage:
  python train_rpgpdppo_pointnav.py --config config_pointnav_RPGPD.json --device cpu --epochs 50
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
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

HISTORY_FILE = "train_history.json"


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
    p.add_argument(
        "--reset_retries",
        type=int,
        default=8,
        help="Retries per worker when env reset/model build fails with transient texture decode errors.",
    )
    p.add_argument(
        "--reset_retry_backoff",
        type=float,
        default=0.2,
        help="Base backoff seconds for reset retries (exponential).",
    )
    p.add_argument(
        "--reset_lock_path",
        type=str,
        default="/tmp/safety_gymnasium_texture_reset.lock",
        help="Cross-process lock file used to serialize env build/reset in parallel rollout.",
    )
    p.add_argument(
        "--serialize_env_reset",
        dest="serialize_env_reset",
        action="store_true",
        help="Serialize MuJoCo env build/reset across parallel workers to avoid texture decode races.",
    )
    p.add_argument(
        "--no_serialize_env_reset",
        dest="serialize_env_reset",
        action="store_false",
        help="Disable reset serialization lock.",
    )
    p.set_defaults(serialize_env_reset=True)
    # Accepted for CLI compatibility (not used in this script)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--live", action="store_true", help="Render environment live during sequential rollout collection")
    p.add_argument(
        "--fixed_rollout_seeds",
        action="store_true",
        help="Disable per-rollout random seeding and use deterministic seed schedule",
    )
    p.add_argument(
        "--fixed_rollout_seed_range",
        type=str,
        default=None,
        help='Use the same rollout seed list every epoch, e.g. "1-120" or "1,2,3,4".',
    )

    # outputs
    p.add_argument("--save_dir", type=str, default="runs_pointnav_rpgpd")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def expand_rollout_seeds(seed_list: list[int], rollout_total: int) -> list[int]:
    if rollout_total <= 0:
        return []
    if len(seed_list) == 0:
        raise ValueError("fixed_rollout_seed_range produced an empty seed list")
    return [int(seed_list[i % len(seed_list)]) for i in range(rollout_total)]


def _next_obs_for_value(next_obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    # VecEnv-style APIs may return reset obs on done; use terminal_observation for V(s_{t+1}) when available.
    candidate = info.get("terminal_observation", next_obs)
    try:
        return np.asarray(candidate, dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray(next_obs, dtype=np.float32).reshape(-1)


def load_checkpoint(path: Path, map_location: str) -> Dict[str, Any]:
    """
    Load checkpoint dict across PyTorch versions.
    PyTorch 2.6 defaults torch.load(..., weights_only=True), which breaks
    checkpoints containing numpy objects (e.g., lambda vectors).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch may not support the weights_only argument.
        return torch.load(path, map_location=map_location)


def _is_texture_decode_error(exc: Exception) -> bool:
    msg = str(exc)
    return ("PNG file load error" in msg) and ("ADLER32" in msg)


@contextlib.contextmanager
def _reset_lock(lock_path: str):
    # Best-effort cross-process file lock for Unix-like systems.
    try:
        import fcntl  # type: ignore
    except Exception:
        yield
        return
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_metric_plot(
    save_dir: Path,
    values: list[float],
    name: str,
    ylabel: str,
    epochs: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if len(values) == 0:
        return
    if epochs is not None and len(epochs) == len(values):
        x = np.asarray(epochs, dtype=np.int32)
    else:
        x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, values, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Training {ylabel} per Epoch")
    ax.grid(True, alpha=0.3)
    out = save_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_cost_plot(
    save_dir: Path,
    undiscounted_costs: list[float],
    discounted_costs: list[float],
    cost_limit: float,
    name: str,
    epochs: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if len(undiscounted_costs) == 0 and len(discounted_costs) == 0:
        return

    n = max(len(undiscounted_costs), len(discounted_costs))
    if epochs is not None and len(epochs) >= n:
        x = np.asarray(epochs[:n], dtype=np.int32)
    else:
        x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(undiscounted_costs) > 0:
        ax.plot(
            x[: len(undiscounted_costs)],
            undiscounted_costs,
            linewidth=1.6,
            color="tab:blue",
            label="Undiscounted Cost",
        )
    if len(discounted_costs) > 0:
        ax.plot(
            x[: len(discounted_costs)],
            discounted_costs,
            linewidth=1.6,
            color="tab:red",
            label="Discounted Cost",
        )
    ax.axhline(float(cost_limit), linestyle="--", linewidth=1.4, color="tab:red", alpha=0.85, label="Cost Limit")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cost")
    ax.set_title("Training Cost per Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = save_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_return_cost_plot(
    save_dir: Path,
    returns: list[float],
    returns_std: list[float],
    undiscounted_costs: list[float],
    undiscounted_costs_std: list[float],
    discounted_costs: list[float],
    discounted_costs_std: list[float],
    dual_values: list[float],
    cost_limit: float,
    name: str,
    epochs: list[int] | None = None,
) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n = max(len(returns), len(undiscounted_costs), len(discounted_costs), len(dual_values))
    if n == 0:
        return
    if epochs is not None and len(epochs) >= n:
        x = np.asarray(epochs[:n], dtype=np.int32)
    else:
        x = np.arange(1, n + 1)

    fig, (ax_ret, ax_cost, ax_dual) = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.08},
    )

    def plot_mean_std_gradient(
        ax: Any,
        x_vals: np.ndarray,
        means: list[float],
        stds: list[float],
        color: str,
        mean_label: str,
        std_label: str,
        linewidth: float = 1.8,
    ) -> None:
        if len(means) == 0:
            return
        n_local = min(len(means), len(x_vals))
        if n_local == 0:
            return
        mean_arr = np.asarray(means[:n_local], dtype=np.float32)
        ax.plot(x_vals[:n_local], mean_arr, linewidth=linewidth, color=color, label=mean_label)
        if len(stds) == 0:
            return
        std_arr = np.asarray(stds[:n_local], dtype=np.float32)
        if std_arr.shape[0] < n_local:
            std_arr = np.pad(std_arr, (0, n_local - std_arr.shape[0]), mode="constant", constant_values=0.0)
        std_arr = np.abs(std_arr)
        max_std = float(np.max(std_arr)) if std_arr.size > 0 else 0.0
        if max_std <= 1e-12:
            return
        lower = mean_arr - std_arr
        upper = mean_arr + std_arr
        rgba = mcolors.to_rgba(color)
        for i in range(n_local - 1):
            std_mid = float(0.5 * (std_arr[i] + std_arr[i + 1]))
            alpha = 0.05 + 0.30 * min(1.0, std_mid / max_std)
            label = std_label if i == 0 else None
            ax.fill_between(
                x_vals[i : i + 2],
                lower[i : i + 2],
                upper[i : i + 2],
                color=rgba,
                alpha=alpha,
                linewidth=0.0,
                label=label,
            )

    if len(returns) > 0:
        plot_mean_std_gradient(
            ax=ax_ret,
            x_vals=x,
            means=returns,
            stds=returns_std,
            color="green",
            mean_label="Return Mean",
            std_label="Return ±1 std",
            linewidth=1.8,
        )
    if len(undiscounted_costs) > 0:
        plot_mean_std_gradient(
            ax=ax_cost,
            x_vals=x,
            means=undiscounted_costs,
            stds=undiscounted_costs_std,
            color="tab:blue",
            mean_label="Undiscounted Cost Mean",
            std_label="Undiscounted Cost ±1 std",
            linewidth=1.5,
        )
    if len(discounted_costs) > 0:
        plot_mean_std_gradient(
            ax=ax_cost,
            x_vals=x,
            means=discounted_costs,
            stds=discounted_costs_std,
            color="tab:red",
            mean_label="Discounted Cost Mean",
            std_label="Discounted Cost ±1 std",
            linewidth=1.5,
        )
    ax_cost.axhline(
        float(cost_limit),
        linestyle="--",
        linewidth=1.3,
        color="tab:red",
        alpha=0.85,
        label="Cost Limit",
    )

    ax_ret.set_ylabel("return", color="green")
    ax_ret.tick_params(axis="y", labelcolor="green")
    ax_ret.set_title("Training Return and Cost per Epoch")
    ax_ret.grid(True, alpha=0.3)
    ax_ret.legend(loc="best")

    ax_cost.set_xlabel("epoch")
    ax_cost.set_ylabel("cost")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(loc="best")

    if len(dual_values) > 0:
        ax_dual.plot(
            x[: len(dual_values)],
            dual_values,
            linewidth=1.5,
            color="tab:purple",
            label="Dual Variable (lambda)",
        )
    ax_dual.set_xlabel("epoch")
    ax_dual.set_ylabel("lambda")
    ax_dual.grid(True, alpha=0.3)
    ax_dual.legend(loc="best")

    if len(x) > 0:
        ax_dual.set_xlim(float(x[0]), float(x[-1]))

    out = save_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _history_path(save_dir: Path) -> Path:
    return save_dir / HISTORY_FILE


def load_history(save_dir: Path, upto_epoch: int | None = None) -> Dict[str, list]:
    out = {
        "epochs": [],
        "ret_hist": [],
        "ret_std_hist": [],
        "c0_hist": [],
        "c0_std_hist": [],
        "jc0_hist": [],
        "jc0_std_hist": [],
        "lam0_hist": [],
    }
    p = _history_path(save_dir)
    if not p.exists():
        return out
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        epochs = [int(x) for x in raw.get("epochs", [])]
        ret_hist = [float(x) for x in raw.get("ret_hist", [])]
        c0_hist = [float(x) for x in raw.get("c0_hist", [])]
        jc0_hist = [float(x) for x in raw.get("jc0_hist", [])]
        n = min(len(epochs), len(ret_hist), len(c0_hist), len(jc0_hist))
        epochs, ret_hist, c0_hist, jc0_hist = epochs[:n], ret_hist[:n], c0_hist[:n], jc0_hist[:n]
        ret_std_hist = [float(x) for x in raw.get("ret_std_hist", [])]
        if len(ret_std_hist) >= n:
            ret_std_hist = ret_std_hist[:n]
        else:
            ret_std_hist = ret_std_hist + [0.0] * (n - len(ret_std_hist))
        c0_std_hist = [float(x) for x in raw.get("c0_std_hist", [])]
        if len(c0_std_hist) >= n:
            c0_std_hist = c0_std_hist[:n]
        else:
            c0_std_hist = c0_std_hist + [0.0] * (n - len(c0_std_hist))
        jc0_std_hist = [float(x) for x in raw.get("jc0_std_hist", [])]
        if len(jc0_std_hist) >= n:
            jc0_std_hist = jc0_std_hist[:n]
        else:
            jc0_std_hist = jc0_std_hist + [0.0] * (n - len(jc0_std_hist))
        lam0_hist = [float(x) for x in raw.get("lam0_hist", [])]
        if len(lam0_hist) >= n:
            lam0_hist = lam0_hist[:n]
        else:
            lam0_hist = lam0_hist + [0.0] * (n - len(lam0_hist))
        if upto_epoch is not None:
            keep_n = 0
            for i, e in enumerate(epochs):
                if int(e) <= int(upto_epoch):
                    keep_n = i + 1
                else:
                    break
            epochs, ret_hist, ret_std_hist, c0_hist, c0_std_hist, jc0_hist, jc0_std_hist, lam0_hist = (
                epochs[:keep_n],
                ret_hist[:keep_n],
                ret_std_hist[:keep_n],
                c0_hist[:keep_n],
                c0_std_hist[:keep_n],
                jc0_hist[:keep_n],
                jc0_std_hist[:keep_n],
                lam0_hist[:keep_n],
            )
        out = {
            "epochs": epochs,
            "ret_hist": ret_hist,
            "ret_std_hist": ret_std_hist,
            "c0_hist": c0_hist,
            "c0_std_hist": c0_std_hist,
            "jc0_hist": jc0_hist,
            "jc0_std_hist": jc0_std_hist,
            "lam0_hist": lam0_hist,
        }
    except Exception as e:
        print(f"[History] Failed to load {p}: {e}")
    return out


def save_history(
    save_dir: Path,
    epochs: list[int],
    ret_hist: list[float],
    ret_std_hist: list[float],
    c0_hist: list[float],
    c0_std_hist: list[float],
    jc0_hist: list[float],
    jc0_std_hist: list[float],
    lam0_hist: list[float],
) -> None:
    n = min(
        len(epochs),
        len(ret_hist),
        len(ret_std_hist),
        len(c0_hist),
        len(c0_std_hist),
        len(jc0_hist),
        len(jc0_std_hist),
        len(lam0_hist),
    )
    payload = {
        "epochs": [int(x) for x in epochs[:n]],
        "ret_hist": [float(x) for x in ret_hist[:n]],
        "ret_std_hist": [float(x) for x in ret_std_hist[:n]],
        "c0_hist": [float(x) for x in c0_hist[:n]],
        "c0_std_hist": [float(x) for x in c0_std_hist[:n]],
        "jc0_hist": [float(x) for x in jc0_hist[:n]],
        "jc0_std_hist": [float(x) for x in jc0_std_hist[:n]],
        "lam0_hist": [float(x) for x in lam0_hist[:n]],
    }
    p = _history_path(save_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def save_checkpoint(
    save_dir: Path,
    ep: int,
    agent: RPGPDPPOAgent,
    cfg: RPGPDPPOConfig,
    obs_dim: int,
    act_dim: int,
    num_costs: int,
    history: Dict[str, list] | None = None,
) -> Path:
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
    if history is not None:
        ckpt["history"] = history
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

    cmdp_cfg = CMDPConfig(
        cost_limits=tuple(cfg["cmdp"]["cost_limits"]),
        reward_scale=float(cfg["cmdp"].get("reward_scale", 1.0)),
        cost_scales=tuple(cfg["cmdp"].get("cost_scales", [1.0])),
    )

    wrapped: RewardCostWrapper | None = None
    obs = None
    info = None
    max_retries = max(0, int(args.get("reset_retries", 8)))
    retry_backoff = max(0.0, float(args.get("reset_retry_backoff", 0.2)))
    serialize_reset = bool(args.get("serialize_env_reset", True))
    lock_path = str(args.get("reset_lock_path", "/tmp/safety_gymnasium_texture_reset.lock"))
    for attempt in range(max_retries + 1):
        try:
            ctx = _reset_lock(lock_path) if serialize_reset else contextlib.nullcontext()
            with ctx:
                env_cfg = MujocoPointNavConfig(**cfg["env"])
                env = MujocoPointNavEnv(env_cfg, render_mode=None)
                wrapped = RewardCostWrapper(env, cfg=cmdp_cfg)
                obs, info = wrapped.reset(seed=args["seed"])
            break
        except Exception as exc:
            if wrapped is not None:
                try:
                    wrapped.close()
                except Exception:
                    pass
                wrapped = None
            if _is_texture_decode_error(exc) and attempt < max_retries:
                sleep_s = retry_backoff * (2.0**attempt) + random.uniform(0.0, retry_backoff)
                time.sleep(sleep_s)
                continue
            raise

    assert wrapped is not None and obs is not None and info is not None

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

    max_steps = int(args["max_steps"])
    ep_ret = 0.0
    ep_cost = np.zeros((len(info["cost_limits"]),), dtype=np.float32)
    ep_len = 0
    last_val_r = 0.0
    last_val_c = np.zeros((len(info["cost_limits"]),), dtype=np.float32)

    traj = {
        "obs": [],
        "act": [],
        "logp": [],
        "rew": [],
        "costs": [],
        "val_r": [],
        "val_c": [],
        "done": [],
        "terminated": [],
        "truncated": [],
    }

    for t in range(max_steps):
        act, logp, v_r, v_c = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, info = wrapped.step(act)
        costs = np.asarray(info["costs"], dtype=np.float32)
        # Keep termination cause explicit:
        # - terminated: true MDP terminal (goal/failure), no bootstrap
        # - truncated: time-limit/cap only, bootstrap
        time_limit_reached = bool(truncated or (t + 1 >= max_steps))
        terminated_step = bool(terminated)
        truncated_step = bool(time_limit_reached and (not terminated_step))
        # done is only for loop control/logging.
        done = bool(terminated or time_limit_reached)

        traj["obs"].append(obs)
        traj["act"].append(act)
        traj["logp"].append(float(logp))
        traj["rew"].append(float(rew))
        traj["costs"].append(costs)
        traj["val_r"].append(float(v_r))
        traj["val_c"].append(v_c)
        traj["done"].append(done)
        traj["terminated"].append(terminated_step)
        traj["truncated"].append(truncated_step)

        ep_ret += float(rew)
        ep_cost += costs
        ep_len += 1
        obs = next_obs

        if done:
            # Bootstrap mask rule:
            # - true terminal -> mask=0 -> last value is 0
            # - truncation/time-limit -> mask=1 -> bootstrap from V(next_obs_for_value)
            if bool(terminated):
                last_val_r = 0.0
                last_val_c = np.zeros_like(ep_cost, dtype=np.float32)
            else:
                obs_for_value = _next_obs_for_value(next_obs, info)
                last_val_r, last_val_c = agent.value(obs_for_value)
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
        "terminated": np.asarray(traj["terminated"], dtype=np.float32),
        "truncated": np.asarray(traj["truncated"], dtype=np.float32),
    }

    return {
        "traj": traj_np,
        "ep_ret": float(ep_ret),
        "ep_cost": ep_cost.astype(np.float32),
        "ep_len": int(ep_len),
        "last_val_r": float(last_val_r),
        "last_val_c": np.asarray(last_val_c, dtype=np.float32),
        "seed": int(args["seed"]),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    base_save_dir = Path(args.save_dir)
    if args.resume_checkpoint:
        save_dir = Path(args.resume_checkpoint).resolve().parent
    else:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        save_dir = base_save_dir / run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Save] run_dir={save_dir}")

    random_seed_each_rollout = not args.fixed_rollout_seeds
    fixed_rollout_seeds: list[int] | None = None
    if args.fixed_rollout_seed_range is not None:
        fixed_rollout_seeds = parse_seed_spec(args.fixed_rollout_seed_range)
        # Explicit fixed seed list has highest priority.
        random_seed_each_rollout = False
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
        pi_lr_mu=float(algo["pi_lr_mu"]) if algo.get("pi_lr_mu") is not None else None,
        pi_lr_std=float(algo["pi_lr_std"]) if algo.get("pi_lr_std") is not None else None,
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
        ckpt = load_checkpoint(Path(args.init_checkpoint).resolve(), map_location=args.device)
        agent.pi.load_state_dict(ckpt.get("pi", ckpt.get("pi_hat")))
        agent.v_r.load_state_dict(ckpt["v_r"])
        agent.v_c.load_state_dict(ckpt["v_c"])
        agent.lam = np.asarray(ckpt.get("lambda", ckpt.get("lambda_hat", agent.lam)), dtype=np.float32)
        print(f"[Init] Loaded weights from {args.init_checkpoint}")
    elif args.resume_checkpoint:
        ckpt = load_checkpoint(Path(args.resume_checkpoint).resolve(), map_location=args.device)
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
    if fixed_rollout_seeds is not None:
        rollout_total_cfg = int(algo_cfg.num_roll_out) if algo_cfg.num_roll_out is not None else 1
        if len(fixed_rollout_seeds) < rollout_total_cfg:
            print(
                f"[Seed] fixed_rollout_seed_range has {len(fixed_rollout_seeds)} seed(s) and num_roll_out={rollout_total_cfg}; "
                "seeds will repeat cyclically each epoch.",
            )
        print(
            f"[Seed] fixed_rollout_seed_range enabled: using same {len(fixed_rollout_seeds)} rollout seeds every epoch "
            f"(first={fixed_rollout_seeds[0]}, last={fixed_rollout_seeds[-1]}).",
        )
    if args.live and algo_cfg.num_roll_out is not None and int(args.rollout_parallel) > 1:
        print("[Live] Live rendering is only supported in sequential rollout. Disabling rollout_parallel.")
        args.rollout_parallel = 1
    if int(args.rollout_parallel) > 1 and int(algo_cfg.num_roll_out) <= 1:
        print("[Rollout] rollout_parallel>1 with a single episode per epoch has no benefit; using sequential collection.")
        args.rollout_parallel = 1

    epoch_hist: list[int] = []
    ret_hist: list[float] = []
    ret_std_hist: list[float] = []
    c0_hist: list[float] = []
    c0_std_hist: list[float] = []
    jc0_hist: list[float] = []
    jc0_std_hist: list[float] = []
    lam0_hist: list[float] = []
    if args.resume_checkpoint:
        resume_epoch = max(0, start_epoch - 1)
        hist = load_history(save_dir, upto_epoch=resume_epoch)
        epoch_hist = hist["epochs"]
        ret_hist = hist["ret_hist"]
        ret_std_hist = hist["ret_std_hist"]
        c0_hist = hist["c0_hist"]
        c0_std_hist = hist["c0_std_hist"]
        jc0_hist = hist["jc0_hist"]
        jc0_std_hist = hist["jc0_std_hist"]
        lam0_hist = hist["lam0_hist"]
        if len(epoch_hist) == 0 and isinstance(ckpt.get("history"), dict):
            h = ckpt["history"]
            epoch_hist = [int(x) for x in h.get("epochs", []) if int(x) <= resume_epoch]
            ret_hist = [float(x) for x in h.get("ret_hist", [])][: len(epoch_hist)]
            raw_ret_std_hist = [float(x) for x in h.get("ret_std_hist", [])]
            ret_std_hist = raw_ret_std_hist[: len(epoch_hist)]
            if len(ret_std_hist) < len(epoch_hist):
                ret_std_hist += [0.0] * (len(epoch_hist) - len(ret_std_hist))
            c0_hist = [float(x) for x in h.get("c0_hist", [])][: len(epoch_hist)]
            raw_c0_std_hist = [float(x) for x in h.get("c0_std_hist", [])]
            c0_std_hist = raw_c0_std_hist[: len(epoch_hist)]
            if len(c0_std_hist) < len(epoch_hist):
                c0_std_hist += [0.0] * (len(epoch_hist) - len(c0_std_hist))
            jc0_hist = [float(x) for x in h.get("jc0_hist", [])][: len(epoch_hist)]
            raw_jc0_std_hist = [float(x) for x in h.get("jc0_std_hist", [])]
            jc0_std_hist = raw_jc0_std_hist[: len(epoch_hist)]
            if len(jc0_std_hist) < len(epoch_hist):
                jc0_std_hist += [0.0] * (len(epoch_hist) - len(jc0_std_hist))
            raw_lam0_hist = [float(x) for x in h.get("lam0_hist", [])]
            lam0_hist = raw_lam0_hist[: len(epoch_hist)]
            if len(lam0_hist) < len(epoch_hist):
                lam0_hist += [0.0] * (len(epoch_hist) - len(lam0_hist))
        if len(epoch_hist) > 0:
            print(f"[History] Restored {len(epoch_hist)} epochs for plotting (through epoch {epoch_hist[-1]}).")
        else:
            print("[History] No prior history file found; plots will start from resumed epoch.")

    for ep in range(start_epoch, epochs + 1):
        epoch_t0 = time.perf_counter()
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
            if fixed_rollout_seeds is not None:
                seeds = expand_rollout_seeds(fixed_rollout_seeds, rollout_total)
            elif random_seed_each_rollout:
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
                    "reset_retries": int(args.reset_retries),
                    "reset_retry_backoff": float(args.reset_retry_backoff),
                    "serialize_env_reset": bool(args.serialize_env_reset),
                    "reset_lock_path": str(args.reset_lock_path),
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
                        done=bool(traj["done"][i]),
                        terminated=bool(traj["terminated"][i]),
                        truncated=bool(traj["truncated"][i]),
                    )
                last_val_r = float(summ["last_val_r"])
                last_val_c = np.asarray(summ["last_val_c"], dtype=np.float32)
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
            seq_rollout_seeds: list[int] | None = None
            rollout_total = int(algo_cfg.num_roll_out) if algo_cfg.num_roll_out is not None else 1
            if fixed_rollout_seeds is not None:
                seq_rollout_seeds = expand_rollout_seeds(fixed_rollout_seeds, rollout_total)
            collect = trainer.collect_epoch(
                seed=seed + ep,
                live=args.live,
                random_seed_each_rollout=random_seed_each_rollout,
                rollout_seeds=seq_rollout_seeds,
            )
        upd = trainer.train_epoch()
        epoch_hist.append(int(ep))
        ret_hist.append(float(collect.get("EpRetMean", 0.0)))
        ret_std_hist.append(float(collect.get("EpRetStd", 0.0)))
        c0_hist.append(float(collect.get("EpCost0Mean", 0.0)))
        c0_std_hist.append(float(collect.get("EpCost0Std", 0.0)))
        lam0 = float(agent.lam[0]) if len(agent.lam) > 0 else 0.0
        lam0_hist.append(lam0)
        jc0_mean = float(upd.get("Jc_mean_0", upd.get("Jc_0", 0.0)))
        jc0_std = float(upd.get("Jc_std_0", 0.0))
        jc0_hist.append(jc0_mean)
        jc0_std_hist.append(jc0_std)
        save_history(
            save_dir,
            epoch_hist,
            ret_hist,
            ret_std_hist,
            c0_hist,
            c0_std_hist,
            jc0_hist,
            jc0_std_hist,
            lam0_hist,
        )
        jr_mean = float(upd.get("Jr_mean", 0.0))
        jr_std = float(upd.get("Jr_std", 0.0))
        viol0 = float(upd.get("viol_0", 0.0))
        grad_log_std_mean_abs = float(upd.get("grad_log_std_mean_abs", 0.0))
        grad_std_mean_abs = float(upd.get("grad_std_mean_abs", 0.0))
        grad_mu_mean_abs = float(upd.get("grad_mu_mean_abs", 0.0))
        clipfrac = float(upd.get("clipfrac", 0.0))
        ratio_mean = float(upd.get("ratio_mean", 1.0))
        ratio_min = float(upd.get("ratio_min", 1.0))
        ratio_max = float(upd.get("ratio_max", 1.0))
        pi_stop = int(upd.get("pi_early_stop_iter", 0))
        epoch_sec = time.perf_counter() - epoch_t0
        with torch.no_grad():
            policy_std = torch.exp(agent.pi.log_std.detach()).clamp(1e-4, 10.0).cpu().numpy()
        std_mean = float(np.mean(policy_std))
        std_min = float(np.min(policy_std))
        std_max = float(np.max(policy_std))

        print(
            f"Epoch {ep:03d} | Ret {collect['EpRetMean']:.2f} (std {collect.get('EpRetStd', 0.0):.3f}) | "
            f"Jr {jr_mean:.3f} (std {jr_std:.3f}) | "
            f"C0 {collect.get('EpCost0Mean', 0.0):.2f} (std {collect.get('EpCost0Std', 0.0):.3f}) | "
            f"Jc0 {jc0_mean:.3f} (std {jc0_std:.3f}) | viol0 {viol0:.3f} | "
            f"KL {upd.get('kl', 0.0):.4f} | λ0 {lam0:.3f} | "
            f"pi_std mean/min/max {std_mean:.4f}/{std_min:.4f}/{std_max:.4f} | "
            f"|dL/dmu| {grad_mu_mean_abs:.3e} | |dL/dlogstd| {grad_log_std_mean_abs:.3e} | "
            f"|dL/dstd| {grad_std_mean_abs:.3e} | "
            f"clipfrac {clipfrac:.3f} ratio {ratio_mean:.3f}[{ratio_min:.3f},{ratio_max:.3f}] | "
            f"pi_stop {pi_stop} | "
            f"time {epoch_sec:.2f}s"
        )

        if args.checkpoint_every > 0 and (ep % args.checkpoint_every == 0):
            ckpt_path = save_checkpoint(
                save_dir,
                ep,
                agent,
                algo_cfg,
                wrapped.obs_dim(),
                wrapped.act_dim(),
                len(cmdp_cfg.cost_limits),
                history={
                    "epochs": epoch_hist,
                    "ret_hist": ret_hist,
                    "ret_std_hist": ret_std_hist,
                    "c0_hist": c0_hist,
                    "c0_std_hist": c0_std_hist,
                    "jc0_hist": jc0_hist,
                    "jc0_std_hist": jc0_std_hist,
                    "lam0_hist": lam0_hist,
                },
            )
            print(f"  [Saved] {ckpt_path}")

        cost_limit0 = float(cmdp_cfg.cost_limits[0]) if len(cmdp_cfg.cost_limits) > 0 else 0.0
        save_return_cost_plot(
            save_dir=save_dir,
            returns=ret_hist,
            returns_std=ret_std_hist,
            undiscounted_costs=c0_hist,
            undiscounted_costs_std=c0_std_hist,
            discounted_costs=jc0_hist,
            discounted_costs_std=jc0_std_hist,
            dual_values=lam0_hist,
            cost_limit=cost_limit0,
            name="returns_costs_dual_latest",
            epochs=epoch_hist,
        )


if __name__ == "__main__":
    main()
