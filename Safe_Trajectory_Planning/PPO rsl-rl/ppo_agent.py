from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from tensordict import TensorDict


@dataclass
class RslPPOConfig:
    seed: int = 0
    max_iterations: int = 300
    save_interval: int = 25
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    schedule: str = "adaptive"
    hidden_dims: tuple[int, int, int] = (256, 256, 256)
    activation: str = "elu"
    init_noise_std: float = 1.0


def build_train_cfg(cfg: RslPPOConfig, experiment_name: str, run_name: str) -> Dict[str, Any]:
    return {
        "seed": int(cfg.seed),
        "num_steps_per_env": int(cfg.num_steps_per_env),
        "save_interval": int(cfg.save_interval),
        "run_name": str(run_name),
        "experiment_name": str(experiment_name),
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": int(cfg.num_learning_epochs),
            "num_mini_batches": int(cfg.num_mini_batches),
            "clip_param": float(cfg.clip_param),
            "gamma": float(cfg.gamma),
            "lam": float(cfg.lam),
            "entropy_coef": float(cfg.entropy_coef),
            "value_loss_coef": float(cfg.value_loss_coef),
            "learning_rate": float(cfg.learning_rate),
            "max_grad_norm": float(cfg.max_grad_norm),
            "desired_kl": float(cfg.desired_kl),
            "schedule": str(cfg.schedule),
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(cfg.hidden_dims),
            "activation": str(cfg.activation),
            "obs_normalization": True,
            "stochastic": True,
            "init_noise_std": float(cfg.init_noise_std),
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(cfg.hidden_dims),
            "activation": str(cfg.activation),
            "obs_normalization": True,
            "stochastic": False,
        },
    }


class SafetyGymVecEnv:
    """Minimal vectorized env adapter for rsl-rl OnPolicyRunner."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        num_envs: int,
        device: str,
        seed: int,
        gamma: float = 0.99,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.envs = [env_factory() for _ in range(self.num_envs)]
        self._base_seed = int(seed)
        self._next_seed = int(seed)
        self.gamma = float(gamma)

        first = self.envs[0]
        self.num_obs = int(first.obs_dim())
        self.num_actions = int(first.act_dim())
        self.num_privileged_obs = 0
        self.max_episode_length = int(getattr(first.cfg, "max_steps", 1000))
        self.cfg = {"env_name": "SafetyGymVecEnv", "max_episode_length": self.max_episode_length}

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_ret_unshaped_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_ret_unshaped_disc_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_disc_factor_buf = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_cost0_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_collision_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_success_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.reset()

    def _reset_one(self, i: int, seed: int | None = None) -> np.ndarray:
        obs, _ = self.envs[i].reset(seed=seed)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    @staticmethod
    def _to_scalar(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    @staticmethod
    def _extract_success(info: Dict[str, Any]) -> float:
        for key in ("success", "is_success", "goal_met", "goal_achieved", "task_success"):
            if key in info:
                v = info[key]
                try:
                    if isinstance(v, (bool, np.bool_)):
                        return float(v)
                    return 1.0 if float(np.asarray(v).reshape(-1)[0]) > 0.5 else 0.0
                except Exception:
                    return 0.0
        return 0.0

    @staticmethod
    def _extract_collision_increment(info: Dict[str, Any], fallback_cost0: float) -> float:
        for key in ("collision", "collisions", "num_collisions", "contact", "contacts"):
            if key in info:
                try:
                    v = float(np.asarray(info[key]).reshape(-1)[0])
                    return max(0.0, v)
                except Exception:
                    pass
        # In Safety-Gym style tasks, cost0 is often collision count-like.
        return max(0.0, float(fallback_cost0))

    def get_observations(self):
        return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        obs = []
        for i in range(self.num_envs):
            obs.append(self._reset_one(i, seed=self._base_seed + i))
        self._obs_buf = torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        self.episode_length_buf.zero_()
        self._episode_return_buf.zero_()
        self._episode_ret_unshaped_buf.zero_()
        self._episode_ret_unshaped_disc_buf.zero_()
        self._episode_disc_factor_buf.fill_(1.0)
        self._episode_cost0_buf.zero_()
        self._episode_collision_buf.zero_()
        self._episode_success_buf.zero_()
        return self._obs_buf, None

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        next_obs: List[np.ndarray] = []
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        done_returns: List[float] = []
        done_lengths: List[int] = []
        done_ret_unshaped: List[float] = []
        done_ret_unshaped_disc: List[float] = []
        done_cost0: List[float] = []
        done_success: List[float] = []
        done_collision: List[float] = []

        for i in range(self.num_envs):
            o2, r, terminated, truncated, info = self.envs[i].step(actions_np[i])
            done = bool(terminated or truncated)
            costs = np.asarray(info.get("costs", [0.0]), dtype=np.float32).reshape(-1)
            c0 = float(costs[0]) if costs.size > 0 else 0.0
            r_unshaped = self._to_scalar(info.get("reward_unshaped", r), default=float(r))

            rewards[i] = float(r)
            self._episode_return_buf[i] += float(r)
            self._episode_ret_unshaped_buf[i] += float(r_unshaped)
            self._episode_ret_unshaped_disc_buf[i] += self._episode_disc_factor_buf[i] * float(r_unshaped)
            self._episode_disc_factor_buf[i] *= float(self.gamma)
            self._episode_cost0_buf[i] += float(c0)
            self._episode_collision_buf[i] += float(self._extract_collision_increment(info, fallback_cost0=c0))
            self.episode_length_buf[i] += 1

            if done:
                dones[i] = True
                if bool(truncated):
                    time_outs[i] = 1.0
                done_returns.append(float(self._episode_return_buf[i].item()))
                done_lengths.append(int(self.episode_length_buf[i].item()))
                done_ret_unshaped.append(float(self._episode_ret_unshaped_buf[i].item()))
                done_ret_unshaped_disc.append(float(self._episode_ret_unshaped_disc_buf[i].item()))
                done_cost0.append(float(self._episode_cost0_buf[i].item()))
                succ = float(self._extract_success(info))
                done_success.append(succ)
                done_collision.append(float(self._episode_collision_buf[i].item()))

                self._episode_return_buf[i] = 0.0
                self._episode_ret_unshaped_buf[i] = 0.0
                self._episode_ret_unshaped_disc_buf[i] = 0.0
                self._episode_disc_factor_buf[i] = 1.0
                self._episode_cost0_buf[i] = 0.0
                self._episode_collision_buf[i] = 0.0
                self._episode_success_buf[i] = 0.0
                self.episode_length_buf[i] = 0
                self._next_seed += 1
                o2 = self._reset_one(i, seed=self._next_seed)

            next_obs.append(np.asarray(o2, dtype=np.float32).reshape(-1))

        self._obs_buf = torch.as_tensor(np.stack(next_obs, axis=0), dtype=torch.float32, device=self.device)

        extras: Dict[str, Any] = {"time_outs": time_outs}
        if done_returns:
            extras["episode"] = {
                "r": torch.as_tensor(done_returns, dtype=torch.float32, device=self.device),
                "l": torch.as_tensor(done_lengths, dtype=torch.float32, device=self.device),
                "EpRetUnshaped": torch.as_tensor(done_ret_unshaped, dtype=torch.float32, device=self.device),
                "EpRetUnshapedDiscounted": torch.as_tensor(
                    done_ret_unshaped_disc, dtype=torch.float32, device=self.device
                ),
                "EpCost0": torch.as_tensor(done_cost0, dtype=torch.float32, device=self.device),
                "success_rate": torch.as_tensor(done_success, dtype=torch.float32, device=self.device),
                "collision_count": torch.as_tensor(done_collision, dtype=torch.float32, device=self.device),
            }
        obs_td = TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs], device=self.device)
        return obs_td, rewards, dones, extras

    def close(self):
        for env in self.envs:
            env.close()
