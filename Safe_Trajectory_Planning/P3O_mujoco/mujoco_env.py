"""
mujoco_env.py

Wrapper for Safety-Gymnasium point navigation (MuJoCo).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import safety_gymnasium as safety_gymnasium
except Exception as exc:  # pragma: no cover - import-time dependency check
    raise ImportError(
        "safety_gymnasium is required. Install with: pip install safety-gymnasium mujoco gymnasium"
    ) from exc


@dataclass
class MujocoPointNavConfig:
    env_id: str = "SafetyPointGoal1-v0"
    max_steps: int = 1000
    action_scale: float = 1.0
    color_overrides: Dict[str, Tuple[float, float, float, float]] | None = None
    env_config_overrides: Dict[str, Any] | None = None
    env_kwargs: Dict[str, Any] | None = None


class MujocoPointNavEnv:
    """
    Minimal env wrapper with P3O-friendly API:
      obs, info = env.reset(seed)
      obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, cfg: Optional[MujocoPointNavConfig] = None, render_mode: Optional[str] = None):
        self.cfg = cfg or MujocoPointNavConfig()
        self.render_mode = render_mode
        make_kwargs: Dict[str, Any] = dict(self.cfg.env_kwargs or {})
        if self.cfg.env_config_overrides:
            make_kwargs["config"] = dict(self.cfg.env_config_overrides)
        self._env = safety_gymnasium.make(self.cfg.env_id, render_mode=render_mode, **make_kwargs)
        self._t = 0

        self._obs_dim = self._infer_obs_dim(self._env.observation_space)
        self._act_dim = self._infer_act_dim(self._env.action_space)

    def _infer_obs_dim(self, space) -> int:
        shape = getattr(space, "shape", None)
        if shape is not None:
            return int(np.prod(shape))
        spaces = getattr(space, "spaces", None)
        if spaces is not None:
            return int(sum(np.prod(s.shape) for s in spaces.values()))
        n = getattr(space, "n", None)
        if n is not None:
            return int(n)
        raise ValueError("Unsupported observation space")

    def _infer_act_dim(self, space) -> int:
        shape = getattr(space, "shape", None)
        if shape is not None:
            return int(np.prod(shape))
        n = getattr(space, "n", None)
        if n is not None:
            return int(n)
        raise ValueError("Unsupported action space")

    @staticmethod
    def _flatten_obs(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if "observation" in obs:
                obs = obs["observation"]
            elif "obs" in obs:
                obs = obs["obs"]
            else:
                parts = [np.asarray(v, dtype=np.float32).ravel() for k, v in sorted(obs.items())]
                obs = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0) * float(self.cfg.action_scale)
        space = self._env.action_space
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            low = np.asarray(low, dtype=np.float32).reshape(-1)
            high = np.asarray(high, dtype=np.float32).reshape(-1)
            if low.shape == action.shape:
                scaled = low + (action + 1.0) * 0.5 * (high - low)
                return np.clip(scaled, low, high)
        return action

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        obs, info = self._env.reset(seed=seed)
        self._apply_geom_colors()
        info = dict(info)
        info["t"] = 0
        info["max_steps"] = int(self.cfg.max_steps)
        return self._flatten_obs(obs), info

    def step(self, action: np.ndarray):
        self._t += 1
        act_env = self._scale_action(action)
        out = self._env.step(act_env)
        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            cost = info.get("cost", 0.0)
        else:
            raise ValueError(f"Unexpected env.step output length: {len(out)}")

        truncated = bool(truncated or (self._t >= int(self.cfg.max_steps)))
        info = dict(info)
        info["t"] = int(self._t)
        info["max_steps"] = int(self.cfg.max_steps)
        info["reward_env"] = float(reward)
        info["cost_env"] = np.asarray(cost, dtype=np.float32).copy()
        return self._flatten_obs(obs), float(reward), cost, bool(terminated), truncated, info

    def render(self, *args, **kwargs):
        try:
            return self._env.render(*args, **kwargs)
        except TypeError:
            unwrapped = getattr(self._env, "unwrapped", None)
            target = unwrapped if unwrapped is not None else self._env
            try:
                return target.render(*args, **kwargs)
            except TypeError:
                try:
                    sig = inspect.signature(target.render)
                    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    return target.render(*args, **filtered)
                except Exception:
                    return target.render()

    def close(self):
        return self._env.close()

    def obs_dim(self) -> int:
        return int(self._obs_dim)

    def act_dim(self) -> int:
        return int(self._act_dim)

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    def _apply_geom_colors(self) -> None:
        overrides = self.cfg.color_overrides
        if not overrides:
            return
        try:
            model = self._env.unwrapped.model
        except Exception:
            return

        names = []
        try:
            for i in range(model.ngeom):
                try:
                    names.append(model.geom(i).name)
                except Exception:
                    name = model.names[model.name_geomadr[i] :].split(b"\x00", 1)[0].decode()
                    names.append(name)
        except Exception:
            return

        for i, name in enumerate(names):
            for key, rgba in overrides.items():
                if key in name:
                    model.geom_rgba[i] = np.asarray(rgba, dtype=np.float32)
                    break
