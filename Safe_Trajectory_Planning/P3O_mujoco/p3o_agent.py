
"""
p3o_agent.py

Penalized Proximal Policy Optimization (P3O) implementation for multi-constraint CMDPs.

Key equations:
  L_P3O(θ) = L_R^CLIP(θ) + κ * Σ_i relu(L_Ci^CLIP(θ))

  L_R^CLIP(θ) = E[ -min( r(θ) A_R , clip(r(θ)) A_R ) ]

  L_Ci^CLIP(θ) = E[ max( r(θ) A_Ci , clip(r(θ)) A_Ci ) ] + (1-γ)(J_Ci(π_k) - d_i)

This file depends only on numpy + torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def combined_shape(length: int, shape):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, int(shape))
    return (length, *shape)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Compute discounted cumulative sums of vectors."""
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = float(x[t]) + discount * running
        out[t] = running
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes=(255, 255), activation=nn.Tanh):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, int(h)))
            layers.append(activation())
            last = int(h)
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

        # Orthogonal-ish init often helps, but keep default for simplicity.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TanhGaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing to keep actions in (-1, 1).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(255,255)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)
        # state-independent log_std
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, obs: torch.Tensor):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).clamp(1e-4, 10.0)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _tanh_correction(u: torch.Tensor) -> torch.Tensor:
        # log|det d(tanh(u))/du| = sum log(1 - tanh(u)^2)
        # Add epsilon for numerical stability.
        return torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        u = dist.rsample()  # reparameterized
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(dim=-1)
        logp = logp_u - self._tanh_correction(u)
        return a, logp

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Compute log π(a|s) for already-squashed act in (-1,1).
        Invert tanh via atanh.
        """
        # clamp act for atanh stability
        a = act.clamp(-1 + 1e-6, 1 - 1e-6)
        u = 0.5 * torch.log((1 + a) / (1 - a))  # atanh
        dist = self._dist(obs)
        logp_u = dist.log_prob(u).sum(dim=-1)
        logp = logp_u - self._tanh_correction(u)
        return logp

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(obs)
        return torch.tanh(mu)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int = 1, hidden_sizes=(255,255)):
        super().__init__()
        self.v_net = MLP(obs_dim, out_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)


@dataclass
class P3OConfig:
    # Paper defaults (Table 2)
    gamma: float = 0.99
    lam: float = 0.97
    clip_ratio: float = 0.2
    kappa: float = 20.0
    target_kl: float = 1e-2

    # Optimization
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    max_grad_norm: float = 0.5

    # Data collection
    steps_per_epoch: int = 30_000   # Navigation buffer size in paper (Table 3)
    max_ep_len: int = 1000          # rollout length T in paper (Table 3)
    num_roll_out: Optional[int] = None  # if set, collect this many episodes per epoch

    # Nets
    hidden_sizes: Tuple[int, int] = (255, 255)

    # Optional entropy bonus
    entropy_coef: float = 0.0

    # Advantage normalization
    normalize_advantages: bool = True


class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, num_costs: int, size: int, gamma: float, lam: float):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros((size, num_costs), dtype=np.float32)

        self.val_r_buf = np.zeros(size, dtype=np.float32)
        self.val_c_buf = np.zeros((size, num_costs), dtype=np.float32)

        self.adv_r_buf = np.zeros(size, dtype=np.float32)
        self.ret_r_buf = np.zeros(size, dtype=np.float32)

        self.adv_c_buf = np.zeros((size, num_costs), dtype=np.float32)
        self.ret_c_buf = np.zeros((size, num_costs), dtype=np.float32)

        self.done_buf = np.zeros(size, dtype=np.float32)

        self.gamma = float(gamma)
        self.lam = float(lam)
        self.max_size = int(size)
        self.ptr = 0
        self.path_start_idx = 0

        # For J_C estimation (per-episode discounted sums)
        self._ep_cost_returns: List[np.ndarray] = []  # list of shape (num_costs,)

    def store(self, obs, act, logp, rew, costs, val_r, val_c, done):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp

        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = costs

        self.val_r_buf[self.ptr] = val_r
        self.val_c_buf[self.ptr] = val_c

        self.done_buf[self.ptr] = float(done)
        self.ptr += 1

    def finish_path(self, last_val_r: float, last_val_c: np.ndarray):
        """
        Call at the end of a trajectory (done or epoch cutoff).
        Computes GAE advantages and returns for reward + each cost.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        last_val_c = np.atleast_1d(np.asarray(last_val_c, dtype=np.float32))

        # Reward
        rews = np.append(self.rew_buf[path_slice], last_val_r)
        vals = np.append(self.val_r_buf[path_slice], last_val_r)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_r_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_r_buf[path_slice] = self.adv_r_buf[path_slice] + self.val_r_buf[path_slice]

        # Costs (multi)
        for i in range(self.cost_buf.shape[1]):
            c = np.append(self.cost_buf[path_slice, i], float(last_val_c[i]))
            v = np.append(self.val_c_buf[path_slice, i], float(last_val_c[i]))
            d = c[:-1] + self.gamma * v[1:] - v[:-1]
            self.adv_c_buf[path_slice, i] = discount_cumsum(d, self.gamma * self.lam)
            self.ret_c_buf[path_slice, i] = self.adv_c_buf[path_slice, i] + self.val_c_buf[path_slice, i]

        # Episode discounted cost return estimate from trajectory start
        # J_C ≈ Σ γ^t c_t
        traj_costs = self.cost_buf[path_slice]  # (T, m)
        disc = (self.gamma ** np.arange(traj_costs.shape[0], dtype=np.float32)).reshape(-1, 1)
        ep_Jc = (traj_costs * disc).sum(axis=0)
        self._ep_cost_returns.append(ep_Jc.astype(np.float32))

        self.path_start_idx = self.ptr

    def get_Jc_estimate(self) -> np.ndarray:
        if len(self._ep_cost_returns) == 0:
            return np.zeros((self.cost_buf.shape[1],), dtype=np.float32)
        return np.mean(np.stack(self._ep_cost_returns, axis=0), axis=0).astype(np.float32)

    def get(self):
        """
        Return a dict of torch tensors ready for training.
        Advantage normalization is handled outside.
        """
        assert self.ptr > 0, "Buffer is empty."
        end = int(self.ptr)
        data = dict(
            obs=torch.as_tensor(self.obs_buf[:end], dtype=torch.float32),
            act=torch.as_tensor(self.act_buf[:end], dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf[:end], dtype=torch.float32),
            adv_r=torch.as_tensor(self.adv_r_buf[:end], dtype=torch.float32),
            ret_r=torch.as_tensor(self.ret_r_buf[:end], dtype=torch.float32),
            adv_c=torch.as_tensor(self.adv_c_buf[:end], dtype=torch.float32),
            ret_c=torch.as_tensor(self.ret_c_buf[:end], dtype=torch.float32),
        )
        return data


class P3OAgent:
    def __init__(self, obs_dim: int, act_dim: int, num_costs: int, cfg: Optional[P3OConfig] = None, device: str = "cpu"):
        self.cfg = cfg or P3OConfig()
        self.device = torch.device(device)

        self.pi = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.v_r = ValueNet(obs_dim, out_dim=1, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.v_c = ValueNet(obs_dim, out_dim=num_costs, hidden_sizes=self.cfg.hidden_sizes).to(self.device)

        self.pi_opt = optim.Adam(self.pi.parameters(), lr=self.cfg.pi_lr)
        self.vr_opt = optim.Adam(self.v_r.parameters(), lr=self.cfg.vf_lr)
        self.vc_opt = optim.Adam(self.v_c.parameters(), lr=self.cfg.vf_lr)

        self.num_costs = int(num_costs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        if deterministic:
            a = self.pi.deterministic(obs_t)
            logp = self.pi.log_prob(obs_t, a)
        else:
            a, logp = self.pi.sample(obs_t)
        v_r = self.v_r(obs_t)
        v_c = self.v_c(obs_t)  # shape (1, m) or (1,) when m=1
        v_c_np = v_c.squeeze(0).cpu().numpy().astype(np.float32)
        v_c_np = np.atleast_1d(v_c_np)
        return a.squeeze(0).cpu().numpy().astype(np.float32), float(logp.item()), float(v_r.item()), v_c_np

    def update(self, data: Dict[str, torch.Tensor], cost_limits: np.ndarray, Jc_est: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        device = self.device

        obs = data["obs"].to(device)
        act = data["act"].to(device)
        logp_old = data["logp"].to(device)

        adv_r = data["adv_r"].to(device)
        ret_r = data["ret_r"].to(device)

        adv_c = data["adv_c"].to(device)  # (N, m)
        ret_c = data["ret_c"].to(device)

        # Advantage normalization (paper notes it helps fixed kappa)
        if cfg.normalize_advantages:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            # per-cost normalization
            adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (adv_c.std(dim=0, keepdim=True) + 1e-8)

        # Shift constants per constraint: (1-gamma)(J_C - d)
        cost_limits_t = torch.as_tensor(cost_limits, dtype=torch.float32, device=device)
        Jc_t = torch.as_tensor(Jc_est, dtype=torch.float32, device=device)
        shift = (1.0 - cfg.gamma) * (Jc_t - cost_limits_t)  # (m,)

        # -------- policy update --------
        pi_info = {}
        for i in range(cfg.train_pi_iters):
            logp = self.pi.log_prob(obs, act)
            ratio = torch.exp(logp - logp_old)

            clip_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)

            # Reward clipped objective (loss form)
            surr1_r = ratio * adv_r
            surr2_r = clip_ratio * adv_r
            loss_r = -(torch.min(surr1_r, surr2_r)).mean()

            # Costs clipped objectives (loss form)
            # L_Ci = E[ max(ratio*A_Ci, clip_ratio*A_Ci) ] + shift_i
            surr1_c = ratio.unsqueeze(-1) * adv_c
            surr2_c = clip_ratio.unsqueeze(-1) * adv_c
            adv_c_term = torch.max(surr1_c, surr2_c).mean(dim=0)  # (m,)
            lc_vec = adv_c_term + shift  # (m,)

            penalty = torch.relu(lc_vec).sum()

            # Entropy bonus (optional)
            # Approx entropy via distribution entropy on pre-tanh normal (rough).
            # If you want a more accurate tanh entropy, we can add it later.
            entropy_bonus = 0.0
            if cfg.entropy_coef != 0.0:
                dist = self.pi._dist(obs)
                entropy_bonus = -cfg.entropy_coef * dist.entropy().sum(dim=-1).mean()

            loss_pi = loss_r + cfg.kappa * penalty + entropy_bonus

            self.pi_opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.max_grad_norm)
            self.pi_opt.step()

            # Approx KL for early stopping
            with torch.no_grad():
                kl = (logp_old - self.pi.log_prob(obs, act)).mean().item()
            if kl > cfg.target_kl:
                pi_info["pi_early_stop_iter"] = i + 1
                break

        # -------- value updates --------
        # Reward critic
        for _ in range(cfg.train_v_iters):
            v = self.v_r(obs)
            loss_v = ((v - ret_r) ** 2).mean()
            self.vr_opt.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.v_r.parameters(), cfg.max_grad_norm)
            self.vr_opt.step()

        # Cost critics (vector output)
        for _ in range(cfg.train_v_iters):
            v = self.v_c(obs)  # (N, m)
            loss_vc = ((v - ret_c) ** 2).mean()
            self.vc_opt.zero_grad(set_to_none=True)
            loss_vc.backward()
            torch.nn.utils.clip_grad_norm_(self.v_c.parameters(), cfg.max_grad_norm)
            self.vc_opt.step()

        # Logging info
        with torch.no_grad():
            logp = self.pi.log_prob(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)

            surr1_r = ratio * adv_r
            surr2_r = clip_ratio * adv_r
            loss_r = -(torch.min(surr1_r, surr2_r)).mean().item()

            surr1_c = ratio.unsqueeze(-1) * adv_c
            surr2_c = clip_ratio.unsqueeze(-1) * adv_c
            adv_c_term = torch.max(surr1_c, surr2_c).mean(dim=0)
            lc_vec = adv_c_term + shift
            penalty_vec = torch.relu(lc_vec)
            penalty_sum = penalty_vec.sum().item()

            entropy = self.pi._dist(obs).entropy().sum(dim=-1).mean().item()
            kl = (logp_old - logp).mean().item()

        out = dict(
            loss_r=float(loss_r),
            penalty_sum=float(penalty_sum),
            kl=float(kl),
            entropy=float(entropy),
            Jc_0=float(Jc_est[0]) if len(Jc_est) > 0 else 0.0,
            adv_c_term_sum=float(adv_c_term.sum().item()),
            constraint_violation_sum=float(shift.sum().item()),
        )
        for j in range(self.num_costs):
            out[f"Jc_{j}"] = float(Jc_est[j])
            out[f"lc_{j}"] = float(lc_vec[j].item())
            out[f"penalty_{j}"] = float(penalty_vec[j].item())
            out[f"adv_c_term_{j}"] = float(adv_c_term[j].item())
            out[f"constraint_violation_{j}"] = float(shift[j].item())
        out.update(pi_info)
        return out


class P3OTrainer:
    def __init__(self, env: Any, agent: P3OAgent, cost_limits: np.ndarray, cfg: Optional[P3OConfig] = None):
        self.env = env
        self.agent = agent
        self.cfg = cfg or agent.cfg
        self.cost_limits = np.asarray(cost_limits, dtype=np.float32)

        if self.cfg.num_roll_out is not None:
            buf_size = int(self.cfg.num_roll_out) * int(self.cfg.max_ep_len)
        else:
            buf_size = int(self.cfg.steps_per_epoch)

        self.buf = RolloutBuffer(
            obs_dim=env.obs_dim(),
            act_dim=env.act_dim(),
            num_costs=len(cost_limits),
            size=buf_size,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam
        )

    def collect_epoch(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Collect either steps_per_epoch steps or num_roll_out episodes (if configured).
        Returns basic episode stats accumulated during collection.
        """
        cfg = self.cfg
        ep_rets: List[float] = []
        ep_costs: List[np.ndarray] = []
        ep_lens: List[int] = []

        if cfg.num_roll_out is not None:
            obs, info = self.env.reset(seed=seed)
            for _ in range(int(cfg.num_roll_out)):
                ep_ret = 0.0
                ep_cost = np.zeros((len(self.cost_limits),), dtype=np.float32)
                ep_len = 0

                for _t in range(cfg.max_ep_len):
                    act, logp, v_r, v_c = self.agent.act(obs, deterministic=False)
                    next_obs, rew, terminated, truncated, info = self.env.step(act)
                    costs = np.asarray(info["costs"], dtype=np.float32)

                    done = bool(terminated or truncated or (ep_len + 1 >= cfg.max_ep_len))
                    self.buf.store(obs, act, logp, rew, costs, v_r, v_c, done)

                    ep_ret += float(rew)
                    ep_cost += costs
                    ep_len += 1
                    obs = next_obs

                    if done:
                        last_val_r = 0.0
                        last_val_c = np.zeros_like(ep_cost, dtype=np.float32)
                        self.buf.finish_path(last_val_r=last_val_r, last_val_c=last_val_c)
                        ep_rets.append(ep_ret)
                        ep_costs.append(ep_cost.copy())
                        ep_lens.append(ep_len)
                        obs, info = self.env.reset(seed=None)
                        break
        else:
            obs, info = self.env.reset(seed=seed)
            ep_ret = 0.0
            ep_cost = np.zeros((len(self.cost_limits),), dtype=np.float32)
            ep_len = 0

            for t in range(cfg.steps_per_epoch):
                act, logp, v_r, v_c = self.agent.act(obs, deterministic=False)
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                costs = np.asarray(info["costs"], dtype=np.float32)

                done = bool(terminated or truncated or (ep_len + 1 >= cfg.max_ep_len))

                self.buf.store(obs, act, logp, rew, costs, v_r, v_c, done)

                ep_ret += float(rew)
                ep_cost += costs
                ep_len += 1

                obs = next_obs

                if done or (t == cfg.steps_per_epoch - 1):
                    # Bootstrap values if trajectory cut off due to epoch end
                    if done:
                        last_val_r = 0.0
                        last_val_c = np.zeros_like(ep_cost, dtype=np.float32)
                    else:
                        # partial trajectory (epoch cutoff)
                        _, _, last_val_r, last_val_c = self.agent.act(obs, deterministic=True)

                    self.buf.finish_path(last_val_r=last_val_r, last_val_c=last_val_c)

                    ep_rets.append(ep_ret)
                    ep_costs.append(ep_cost.copy())
                    ep_lens.append(ep_len)

                    obs, info = self.env.reset(seed=None)
                    ep_ret = 0.0
                    ep_cost[:] = 0.0
                    ep_len = 0

        stats = {
            "EpRetMean": float(np.mean(ep_rets)) if ep_rets else 0.0,
            "EpLenMean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        }
        if ep_costs:
            C = np.stack(ep_costs, axis=0)
            for i in range(C.shape[1]):
                stats[f"EpCost{i}Mean"] = float(np.mean(C[:, i]))
        return stats

    def train_epoch(self) -> Dict[str, float]:
        data = self.buf.get()
        Jc_est = self.buf.get_Jc_estimate()
        log = self.agent.update(data, cost_limits=self.cost_limits, Jc_est=Jc_est)

        # Reset buffer pointers for next epoch
        self.buf.ptr = 0
        self.buf.path_start_idx = 0
        self.buf._ep_cost_returns = []

        return log
