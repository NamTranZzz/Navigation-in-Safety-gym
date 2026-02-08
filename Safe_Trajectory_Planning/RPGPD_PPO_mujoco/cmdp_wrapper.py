"""
cmdp_wrapper.py

Wraps a Safety-Gymnasium env and exposes reward + multi-cost CMDP signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CMDPConfig:
    cost_limits: Tuple[float, ...] = (25.0,)
    reward_scale: float = 1.0
    cost_scales: Tuple[float, ...] = (1.0,)


RewardFn = Callable[[Dict[str, Any]], float]
CostFn = Callable[[Dict[str, Any]], float]


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
        reward_fn: Optional[RewardFn] = None,
        cost_fns: Optional[Sequence[CostFn]] = None,
    ):
        self.env = env
        self.cfg = cfg or CMDPConfig()
        self.reward_fn = reward_fn
        self.cost_fns = list(cost_fns) if cost_fns is not None else None

        if len(self.cfg.cost_limits) != len(self.cfg.cost_scales):
            raise ValueError(
                f"cost_limits length ({len(self.cfg.cost_limits)}) must match cost_scales ({len(self.cfg.cost_scales)})"
            )
        if self.cost_fns is not None and len(self.cfg.cost_limits) != len(self.cost_fns):
            raise ValueError(
                f"cost_limits length ({len(self.cfg.cost_limits)}) must match cost_fns ({len(self.cost_fns)})"
            )

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        info = dict(info)
        info["reward"] = 0.0
        info["costs"] = np.zeros((len(self.cfg.cost_limits),), dtype=np.float32)
        info["cost_limits"] = np.asarray(self.cfg.cost_limits, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        if self.reward_fn is None:
            r = float(reward) * float(self.cfg.reward_scale)
        else:
            r = float(self.reward_fn(info)) * float(self.cfg.reward_scale)

        if self.cost_fns is None:
            cost_arr = np.asarray(cost, dtype=np.float32).reshape(-1)
            if cost_arr.size == 0:
                cost_arr = np.zeros((len(self.cfg.cost_scales),), dtype=np.float32)
            if cost_arr.size == 1 and len(self.cfg.cost_scales) > 1:
                cost_arr = np.full((len(self.cfg.cost_scales),), float(cost_arr[0]), dtype=np.float32)
            if cost_arr.size != len(self.cfg.cost_scales):
                raise ValueError(
                    f"cost array size ({cost_arr.size}) must match cost_scales ({len(self.cfg.cost_scales)})"
                )
            costs = cost_arr * np.asarray(self.cfg.cost_scales, dtype=np.float32)
        else:
            costs_list = []
            for fn, scale in zip(self.cost_fns, self.cfg.cost_scales):
                costs_list.append(float(fn(info)) * float(scale))
            costs = np.asarray(costs_list, dtype=np.float32)

        info["reward"] = r
        info["costs"] = costs
        info["cost_limits"] = np.asarray(self.cfg.cost_limits, dtype=np.float32)
        return obs, r, bool(terminated), bool(truncated), info

    # Convenience passthroughs
    def obs_dim(self) -> int:
        return int(self.env.obs_dim())

    def act_dim(self) -> int:
        return int(self.env.act_dim())

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def get_env_config(self) -> Dict[str, Any]:
        return self.env.get_config()
