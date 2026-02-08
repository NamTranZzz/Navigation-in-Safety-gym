
"""
cmdp_wrapper.py

Wraps a dynamics-only env (Nav2DEnv) and computes:
- reward
- one or more costs (multi-constraint CMDP)

This file also includes default reward/costs for Navigation:
- Reward: velocity projected toward goal direction
- Cost: sum of distances to pillars + dynamic obstacles when within a threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


RewardFn = Callable[[Dict[str, Any]], float]
CostFn = Callable[[Dict[str, Any]], float]


def reward_progress_to_goal(info: Dict[str, Any]) -> float:
    """R = dot( (goal-pos)/||goal-pos||, agent_vel ) + 10 * time_remaining (on success)"""
    # Previous reward was progress to goal:
    # prev = np.asarray(info["prev_agent_pos"], dtype=np.float32)
    # cur = np.asarray(info["agent_pos"], dtype=np.float32)
    # goal = np.asarray(info["goal_pos"], dtype=np.float32)
    # d_prev = float(np.linalg.norm(goal - prev))
    # d_cur = float(np.linalg.norm(goal - cur))
    # return d_prev - d_cur
    cur = np.asarray(info["agent_pos"], dtype=np.float32)
    goal = np.asarray(info["goal_pos"], dtype=np.float32)
    vel = np.asarray(info["agent_vel"], dtype=np.float32)
    to_goal = goal - cur
    norm = float(np.linalg.norm(to_goal))
    if norm < 1e-8:
        return 0.0
    direction = to_goal / norm
    base = float(np.dot(direction, vel))
    if norm <= float(info.get("goal_radius", 0.0)):
        max_steps = int(info.get("max_steps", 0))
        t = int(info.get("t", 0))
        time_remain = max(0, max_steps - t)
        return base +  0.5 * float(time_remain)
    return base


DEFAULT_DISTANCE_THRESHOLD = 1.0


def cost_distance_to_obstacles(info: Dict[str, Any], threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> float:
    """Sum distance to nearby pillars + dynamic obstacles (distance < threshold)."""
    agent = np.asarray(info["agent_pos"], dtype=np.float32)
    pillars = np.asarray(info.get("pillars_xy", []), dtype=np.float32).reshape(-1, 2)
    dyn_obs = np.asarray(info.get("dynamic_obstacles_xy", []), dtype=np.float32).reshape(-1, 2)
    if pillars.size == 0 and dyn_obs.size == 0:
        return 0.0
    objs = np.concatenate([pillars, dyn_obs], axis=0) if (pillars.size or dyn_obs.size) else np.zeros((0, 2), dtype=np.float32)
    dists = np.linalg.norm(objs - agent[None, :], axis=1)
    return float(np.sum(dists[dists < float(threshold)]))


@dataclass
class CMDPConfig:
    cost_limits: Tuple[float, ...] = (20.0,)
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    # optional shaping
    reward_scale: float = 1.0
    cost_scales: Tuple[float, ...] = (1.0,)


class RewardCostWrapper:
    """
    Minimal wrapper interface:
      obs, info = wrapped.reset(seed)
      obs, reward, terminated, truncated, info = wrapped.step(action)

    It writes:
      info["reward"]
      info["costs"]            np.ndarray shape (m,)
      info["cost_limits"]      np.ndarray shape (m,)
    """

    def __init__(
        self,
        env: Any,
        cfg: Optional[CMDPConfig] = None,
        reward_fn: RewardFn = reward_progress_to_goal,
        cost_fns: Sequence[CostFn] = (cost_distance_to_obstacles,),
    ):
        self.env = env
        self.cfg = cfg or CMDPConfig()
        self.reward_fn = reward_fn
        self.cost_fns = list(cost_fns)

        if len(self.cfg.cost_limits) != len(self.cost_fns):
            raise ValueError(f"cost_limits length ({len(self.cfg.cost_limits)}) must match cost_fns ({len(self.cost_fns)})")

        if len(self.cfg.cost_scales) != len(self.cost_fns):
            raise ValueError(f"cost_scales length ({len(self.cfg.cost_scales)}) must match cost_fns ({len(self.cost_fns)})")

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        # Default: reward/cost not computed on reset
        info = dict(info)
        info["reward"] = 0.0
        info["costs"] = np.zeros((len(self.cost_fns),), dtype=np.float32)
        info["cost_limits"] = np.asarray(self.cfg.cost_limits, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray):
        obs, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        r = float(self.reward_fn(info)) * float(self.cfg.reward_scale)
        costs = []
        for fn, scale in zip(self.cost_fns, self.cfg.cost_scales):
            costs.append(float(fn(info)) * float(scale))

        info["reward"] = r
        info["costs"] = np.asarray(costs, dtype=np.float32)
        info["cost_limits"] = np.asarray(self.cfg.cost_limits, dtype=np.float32)
        return obs, r, terminated, truncated, info

    # Convenience passthroughs
    def obs_dim(self) -> int:
        return int(self.env.obs_dim())

    def act_dim(self) -> int:
        return int(self.env.act_dim())

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def get_env_config(self) -> Dict[str, Any]:
        return self.env.get_config()
