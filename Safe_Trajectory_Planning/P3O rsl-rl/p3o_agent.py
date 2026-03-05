from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

try:
    from tensordict import TensorDictBase
except Exception:  # pragma: no cover
    TensorDictBase = TensorDict


@dataclass
class RslP3OConfig:
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

    # P3O-specific
    kappa: float = 20.0
    cost_value_loss_coef: float = 1.0
    value_learning_rate: float = 1e-3
    normalize_reward_advantage: bool = True
    normalize_cost_advantages: bool = True


def build_train_cfg(cfg: RslP3OConfig, experiment_name: str, run_name: str) -> Dict[str, Any]:
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
        self._episode_cost_buf = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)
        self._episode_cost_disc_buf = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)
        self._episode_collision_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_success_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)

        self._done_costs: List[np.ndarray] = []
        self._done_costs_discounted: List[np.ndarray] = []
        self.reset()

    @property
    def num_costs(self) -> int:
        return int(len(self.envs[0].cfg.cost_limits))

    def pop_recent_cost_stats(self) -> Dict[str, np.ndarray]:
        if len(self._done_costs) == 0:
            zeros = np.zeros((self.num_costs,), dtype=np.float32)
            return {
                "undiscounted_mean": zeros,
                "undiscounted_std": zeros,
                "discounted_mean": zeros,
                "discounted_std": zeros,
                "num_episodes": np.asarray(0, dtype=np.int32),
            }
        c = np.stack(self._done_costs, axis=0).astype(np.float32)
        cd = np.stack(self._done_costs_discounted, axis=0).astype(np.float32)
        out = {
            "undiscounted_mean": np.mean(c, axis=0).astype(np.float32),
            "undiscounted_std": np.std(c, axis=0).astype(np.float32),
            "discounted_mean": np.mean(cd, axis=0).astype(np.float32),
            "discounted_std": np.std(cd, axis=0).astype(np.float32),
            "num_episodes": np.asarray(c.shape[0], dtype=np.int32),
        }
        self._done_costs.clear()
        self._done_costs_discounted.clear()
        return out

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
        self._episode_cost_buf.zero_()
        self._episode_cost_disc_buf.zero_()
        self._episode_collision_buf.zero_()
        self._episode_success_buf.zero_()
        self._done_costs.clear()
        self._done_costs_discounted.clear()
        return self._obs_buf, None

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        next_obs: List[np.ndarray] = []
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        step_costs = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)

        done_returns: List[float] = []
        done_lengths: List[int] = []
        done_ret_unshaped: List[float] = []
        done_ret_unshaped_disc: List[float] = []
        done_cost0: List[float] = []
        done_cost0_disc: List[float] = []
        done_success: List[float] = []
        done_collision: List[float] = []

        for i in range(self.num_envs):
            o2, r, terminated, truncated, info = self.envs[i].step(actions_np[i])
            done = bool(terminated or truncated)
            costs = np.asarray(info.get("costs", np.zeros((self.num_costs,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if costs.size == 0:
                costs = np.zeros((self.num_costs,), dtype=np.float32)
            if costs.size == 1 and self.num_costs > 1:
                costs = np.full((self.num_costs,), float(costs[0]), dtype=np.float32)
            costs = costs[: self.num_costs]
            c0 = float(costs[0]) if costs.size > 0 else 0.0
            r_unshaped = self._to_scalar(info.get("reward_unshaped", r), default=float(r))

            # Training uses unshaped reward (P3O reward/cost separation).
            rewards[i] = float(r_unshaped)
            step_costs[i] = torch.as_tensor(costs, dtype=torch.float32, device=self.device)

            self._episode_return_buf[i] += float(r_unshaped)
            self._episode_ret_unshaped_buf[i] += float(r_unshaped)
            self._episode_ret_unshaped_disc_buf[i] += self._episode_disc_factor_buf[i] * float(r_unshaped)
            self._episode_cost_buf[i] += torch.as_tensor(costs, dtype=torch.float32, device=self.device)
            self._episode_cost_disc_buf[i] += self._episode_disc_factor_buf[i] * torch.as_tensor(
                costs,
                dtype=torch.float32,
                device=self.device,
            )
            self._episode_disc_factor_buf[i] *= float(self.gamma)
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
                done_cost0.append(float(self._episode_cost_buf[i, 0].item()))
                done_cost0_disc.append(float(self._episode_cost_disc_buf[i, 0].item()))
                self._done_costs.append(self._episode_cost_buf[i].detach().cpu().numpy().astype(np.float32))
                self._done_costs_discounted.append(self._episode_cost_disc_buf[i].detach().cpu().numpy().astype(np.float32))
                succ = float(self._extract_success(info))
                done_success.append(succ)
                done_collision.append(float(self._episode_collision_buf[i].item()))

                self._episode_return_buf[i] = 0.0
                self._episode_ret_unshaped_buf[i] = 0.0
                self._episode_ret_unshaped_disc_buf[i] = 0.0
                self._episode_disc_factor_buf[i] = 1.0
                self._episode_cost_buf[i].zero_()
                self._episode_cost_disc_buf[i].zero_()
                self._episode_collision_buf[i] = 0.0
                self._episode_success_buf[i] = 0.0
                self.episode_length_buf[i] = 0
                self._next_seed += 1
                o2 = self._reset_one(i, seed=self._next_seed)

            next_obs.append(np.asarray(o2, dtype=np.float32).reshape(-1))

        self._obs_buf = torch.as_tensor(np.stack(next_obs, axis=0), dtype=torch.float32, device=self.device)

        extras: Dict[str, Any] = {
            "time_outs": time_outs,
            "costs": step_costs,
        }
        if done_returns:
            extras["episode"] = {
                "EpRetUnshaped": torch.as_tensor(done_ret_unshaped, dtype=torch.float32, device=self.device),
                "EpRetUnshapedDiscounted": torch.as_tensor(
                    done_ret_unshaped_disc,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "EpCost0": torch.as_tensor(done_cost0, dtype=torch.float32, device=self.device),
                "EpCost0Discounted": torch.as_tensor(done_cost0_disc, dtype=torch.float32, device=self.device),
                "success_rate": torch.as_tensor(done_success, dtype=torch.float32, device=self.device),
                "collision_count": torch.as_tensor(done_collision, dtype=torch.float32, device=self.device),
            }
        obs_td = TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs], device=self.device)
        return obs_td, rewards, dones, extras

    def close(self):
        for env in self.envs:
            env.close()


def _activation(name: str) -> type[nn.Module]:
    act = str(name).lower()
    if act == "relu":
        return nn.ReLU
    if act == "tanh":
        return nn.Tanh
    if act == "gelu":
        return nn.GELU
    if act == "silu":
        return nn.SiLU
    return nn.ELU


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], activation: str):
        super().__init__()
        dims = [int(input_dim), *(int(h) for h in hidden_dims)]
        act_cls = _activation(activation)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_obs_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, TensorDictBase):
        return x["policy"]
    if isinstance(x, dict) and "policy" in x:
        return x["policy"]
    return x


def _reshape_batch(x: Any, batch_size: int) -> Any:
    _ = batch_size
    return x


def _flatten_batch(x: Any) -> Any:
    if isinstance(x, TensorDictBase):
        return x.reshape(-1)
    return x.reshape(-1, *x.shape[2:])


def _flatten_vec(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x
    return x.reshape(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE on [T, N, ...] tensors."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * next_values - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


class P3OTrainerHook:
    def __init__(
        self,
        runner: Any,
        vec_env: SafetyGymVecEnv,
        cfg: RslP3OConfig,
        cost_limits: np.ndarray,
        device: str,
    ):
        self.runner = runner
        self.vec_env = vec_env
        self.cfg = cfg
        self.device = torch.device(device)
        self.cost_limits = torch.as_tensor(np.asarray(cost_limits, dtype=np.float32), device=self.device)

        self.alg = runner.alg
        self.storage = self.alg.storage

        self.reward_critic = ValueMLP(
            input_dim=int(vec_env.num_obs),
            output_dim=1,
            hidden_dims=tuple(cfg.hidden_dims),
            activation=str(cfg.activation),
        ).to(self.device)
        self.cost_critic = ValueMLP(
            input_dim=int(vec_env.num_obs),
            output_dim=int(self.cost_limits.numel()),
            hidden_dims=tuple(cfg.hidden_dims),
            activation=str(cfg.activation),
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.alg.actor.parameters(), lr=float(cfg.learning_rate))
        self.value_optimizer = torch.optim.Adam(
            list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
            lr=float(cfg.value_learning_rate),
        )

        self.rollout_costs: List[torch.Tensor] = []
        self.last_disc_cost = torch.zeros_like(self.cost_limits)

        self._orig_process_env_step = self.alg.process_env_step

    def install(self) -> None:
        self.alg.process_env_step = self._wrap_process_env_step
        self.alg.update = self.update

    def _wrap_process_env_step(self, *args, **kwargs):
        infos = kwargs.get("infos", None)
        if infos is None and len(args) > 0 and isinstance(args[-1], dict):
            infos = args[-1]
        if isinstance(infos, dict) and "costs" in infos:
            c = infos["costs"]
            if not isinstance(c, torch.Tensor):
                c = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            else:
                c = c.to(self.device)
            if c.dim() == 1:
                c = c.unsqueeze(-1)
            self.rollout_costs.append(c.detach())
        return self._orig_process_env_step(*args, **kwargs)

    def _compute_jc_shift(self) -> torch.Tensor:
        stats = self.vec_env.pop_recent_cost_stats()
        ep_count = int(stats.get("num_episodes", 0))
        if ep_count > 0:
            jc = torch.as_tensor(np.asarray(stats["discounted_mean"], dtype=np.float32), device=self.device)
            self.last_disc_cost = jc
        else:
            jc = self.last_disc_cost
        return (1.0 - float(self.cfg.gamma)) * (jc - self.cost_limits)

    def update(self) -> Dict[str, float]:
        st = self.storage
        cfg = self.cfg

        obs_seq = st.observations
        actions_seq = st.actions
        old_logp_seq = getattr(st, "actions_log_prob", None)
        if old_logp_seq is None:
            old_logp_seq = getattr(st, "old_actions_log_prob")

        rewards_seq = st.rewards.to(self.device)
        dones_seq = st.dones.float().to(self.device)
        if rewards_seq.dim() == 3 and rewards_seq.shape[-1] == 1:
            rewards_seq = rewards_seq.squeeze(-1)
        if dones_seq.dim() == 3 and dones_seq.shape[-1] == 1:
            dones_seq = dones_seq.squeeze(-1)

        T = int(rewards_seq.shape[0])
        N = int(rewards_seq.shape[1])
        B = T * N

        if len(self.rollout_costs) >= T:
            costs_seq = torch.stack(self.rollout_costs[:T], dim=0).to(self.device)
        else:
            m = int(self.cost_limits.numel())
            costs_seq = torch.zeros((T, N, m), dtype=torch.float32, device=self.device)
            if len(self.rollout_costs) > 0:
                got = torch.stack(self.rollout_costs, dim=0).to(self.device)
                costs_seq[: got.shape[0]] = got
        obs_seq = _reshape_batch(obs_seq, T)
        obs_flat = _flatten_batch(obs_seq)
        obs_flat_policy = _to_obs_tensor(obs_flat).to(self.device)

        actions_flat = _flatten_batch(actions_seq).to(self.device)
        old_logp_flat = _flatten_vec(_flatten_batch(old_logp_seq).to(self.device))

        with torch.no_grad():
            v_r = self.reward_critic(_to_obs_tensor(obs_seq).to(self.device)).squeeze(-1)
            v_c = self.cost_critic(_to_obs_tensor(obs_seq).to(self.device))
            next_obs = self.runner.env.get_observations()
            next_obs_policy = _to_obs_tensor(next_obs).to(self.device)
            last_v_r = self.reward_critic(next_obs_policy).squeeze(-1)
            last_v_c = self.cost_critic(next_obs_policy)

            adv_r, ret_r = compute_gae(
                rewards=rewards_seq,
                values=v_r,
                dones=dones_seq,
                last_values=last_v_r,
                gamma=float(cfg.gamma),
                lam=float(cfg.lam),
            )
            adv_c, ret_c = compute_gae(
                rewards=costs_seq,
                values=v_c,
                dones=dones_seq.unsqueeze(-1),
                last_values=last_v_c,
                gamma=float(cfg.gamma),
                lam=float(cfg.lam),
            )

            adv_r = adv_r.reshape(B)
            ret_r = ret_r.reshape(B)
            adv_c = adv_c.reshape(B, -1)
            ret_c = ret_c.reshape(B, -1)

            if cfg.normalize_reward_advantage:
                adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            if cfg.normalize_cost_advantages:
                adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (adv_c.std(dim=0, keepdim=True) + 1e-8)

        shift = self._compute_jc_shift()

        num_mini_batches = max(1, int(cfg.num_mini_batches))
        mini_batch_size = max(1, B // num_mini_batches)

        total_loss_pi = 0.0
        total_surrogate_r = 0.0
        total_penalty = 0.0
        total_entropy = 0.0
        total_value = 0.0
        total_value_r = 0.0
        total_value_c = 0.0
        total_kl = 0.0
        total_penalty_min = float("inf")
        total_penalty_max = float("-inf")
        updates = 0

        early_stop = False
        for _ in range(int(cfg.num_learning_epochs)):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                if idx.numel() == 0:
                    continue

                obs_mb = obs_flat[idx]
                obs_mb_policy = obs_flat_policy[idx]
                act_mb = actions_flat[idx]
                old_logp_mb = old_logp_flat[idx]
                adv_r_mb = adv_r[idx]
                adv_c_mb = adv_c[idx]

                self.alg.actor(obs_mb, stochastic_output=True)
                logp = self.alg.actor.get_output_log_prob(act_mb).reshape(-1)
                ratio = torch.exp(logp - old_logp_mb)
                ratio_clipped = torch.clamp(ratio, 1.0 - float(cfg.clip_param), 1.0 + float(cfg.clip_param))

                surr_r = torch.min(ratio * adv_r_mb, ratio_clipped * adv_r_mb).mean()
                cost_obj = torch.max(
                    ratio.unsqueeze(-1) * adv_c_mb,
                    ratio_clipped.unsqueeze(-1) * adv_c_mb,
                ).mean(dim=0)
                lc_vec = cost_obj + shift
                penalty = torch.relu(lc_vec).sum()

                entropy = 0.0
                if hasattr(self.alg.actor, "output_entropy"):
                    entropy = self.alg.actor.output_entropy.mean()

                loss_pi = -surr_r + float(cfg.kappa) * penalty - float(cfg.entropy_coef) * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.alg.actor.parameters(), float(cfg.max_grad_norm))
                self.actor_optimizer.step()

                ret_r_mb = ret_r[idx]
                ret_c_mb = ret_c[idx]
                v_r_mb = self.reward_critic(obs_mb_policy).squeeze(-1)
                v_c_mb = self.cost_critic(obs_mb_policy)
                loss_v_r = (v_r_mb - ret_r_mb).pow(2).mean()
                loss_v_c = (v_c_mb - ret_c_mb).pow(2).mean()
                loss_v = float(cfg.value_loss_coef) * loss_v_r + float(cfg.cost_value_loss_coef) * loss_v_c

                self.value_optimizer.zero_grad(set_to_none=True)
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
                    float(cfg.max_grad_norm),
                )
                self.value_optimizer.step()

                with torch.no_grad():
                    kl = (old_logp_mb - logp).mean()

                updates += 1
                total_loss_pi += float(loss_pi.item())
                total_surrogate_r += float(surr_r.item())
                penalty_val = float(penalty.item())
                total_penalty += penalty_val
                total_penalty_min = min(total_penalty_min, penalty_val)
                total_penalty_max = max(total_penalty_max, penalty_val)
                total_entropy += float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)
                total_value += float(loss_v.item())
                total_value_r += float(loss_v_r.item())
                total_value_c += float(loss_v_c.item())
                total_kl += float(kl.item())

                if float(cfg.desired_kl) > 0.0 and float(kl.item()) > float(cfg.desired_kl):
                    early_stop = True
                    break
            if early_stop:
                break

        self.rollout_costs.clear()

        denom = max(1, updates)
        total_penalty_mean = float(total_penalty / denom)
        if updates == 0:
            total_penalty_min = 0.0
            total_penalty_max = 0.0
        out: Dict[str, float] = {
            "surrogate": float(total_surrogate_r / denom),
            "total_penalty_mean": total_penalty_mean,
            "total_penalty_min": float(total_penalty_min),
            "total_penalty_max": float(total_penalty_max),
            "policy_loss": float(total_loss_pi / denom),
            "value": float(total_value / denom),
            "value_loss_reward": float(total_value_r / denom),
            "value_loss_cost": float(total_value_c / denom),
            "entropy": float(total_entropy / denom),
            "kl_distance": float(total_kl / denom),
            "kappa": float(cfg.kappa),
        }
        for i in range(int(self.cost_limits.numel())):
            out[f"cost_shift_{i}"] = float(shift[i].item())
            out[f"Jc_disc_{i}"] = float(self.last_disc_cost[i].item())
        # Match rsl-rl PPO lifecycle: clear rollout storage after each update cycle.
        self.storage.clear()
        return out
