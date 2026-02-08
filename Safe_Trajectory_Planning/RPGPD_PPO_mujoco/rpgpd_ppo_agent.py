"""
rpgpd_ppo_agent.py

Regularized Policy Gradient Primal-Dual (RPG-PD) integrated with PPO-style policy updates.

Single-loop algorithm (one rollout batch per epoch):
  1) Rollout with current policy π_k
  2) Primal update: PPO on the Lagrangian advantage
        A_L(s,a) = A_r(s,a) - λ^T A_c(s,a)
     with optional policy regularizer H(π) implemented as an entropy bonus.
  3) Dual update: projected ascent with optional shrinkage (dual regularizer)
        λ <- Proj_{[0, λ_max]}( (1 - ηλ τ) λ + ηλ * scale * (J_c - d) )

Multi-constraint support: costs are vectors c(s,a) in R^m with limits d in R^m.

Notes on "regularizer" terms:
- The paper's τ H(π) is implemented as an entropy bonus in PPO with coefficient `entropy_coef`.
- The paper's (τ/2)||λ||^2 is implemented as a shrinkage term in dual update with coefficient `dual_tau`.
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
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, int(h)))
            layers.append(activation())
            last = int(h)
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing to keep actions in (-1, 1)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, obs: torch.Tensor):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).clamp(1e-4, 10.0)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _tanh_correction(u: torch.Tensor) -> torch.Tensor:
        # log|det d(tanh(u))/du| = sum log(1 - tanh(u)^2)
        return torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        u = dist.rsample()
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(dim=-1)
        logp = logp_u - self._tanh_correction(u)
        return a, logp

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, obs_dim: int, out_dim: int = 1, hidden_sizes=(256, 256), squeeze_output: bool = True):
        super().__init__()
        self.squeeze_output = bool(squeeze_output)
        self.v_net = MLP(obs_dim, out_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self.v_net(obs)
        if self.squeeze_output:
            return v.squeeze(-1)
        return v


@dataclass
class RPGPDPPOConfig:
    # PPO / GAE
    gamma: float = 0.99
    lam: float = 0.97
    clip_ratio: float = 0.2
    target_kl: float = 1e-2

    # Optimization
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    max_grad_norm: float = 0.5

    # Data collection
    steps_per_epoch: int = 30_000
    max_ep_len: int = 1000
    num_roll_out: Optional[int] = None  # if set, collect this many episodes per epoch

    # Nets
    hidden_sizes: Tuple[int, int] = (256, 256)

    # Policy regularizer (H(pi)) approx via entropy bonus
    entropy_coef: float = 0.0

    # Advantage normalization
    normalize_advantages: bool = True

    # Dual update
    dual_lr: float = 0.5          # η_λ
    dual_tau: float = 0.0         # τ for (τ/2)||λ||^2 => shrinkage
    lambda_init: float = 0.0
    lambda_max: float = 1000.0
    dual_scale_one_minus_gamma: bool = True


class RolloutBuffer:
    """Stores one rollout batch."""
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

        # Per-episode discounted cost sums for dual update
        self._ep_cost_returns: List[np.ndarray] = []
        # Per-episode discounted reward returns (for logging)
        self._ep_ret_returns: List[float] = []

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self._ep_cost_returns = []
        self._ep_ret_returns = []

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
        path_slice = slice(self.path_start_idx, self.ptr)
        last_val_c = np.atleast_1d(np.asarray(last_val_c, dtype=np.float32))

        # Reward GAE
        rews = np.append(self.rew_buf[path_slice], last_val_r)
        vals = np.append(self.val_r_buf[path_slice], last_val_r)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_r_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_r_buf[path_slice] = self.adv_r_buf[path_slice] + self.val_r_buf[path_slice]

        # Cost GAE for each cost dimension
        for i in range(self.cost_buf.shape[1]):
            c = np.append(self.cost_buf[path_slice, i], float(last_val_c[i]))
            v = np.append(self.val_c_buf[path_slice, i], float(last_val_c[i]))
            d = c[:-1] + self.gamma * v[1:] - v[:-1]
            self.adv_c_buf[path_slice, i] = discount_cumsum(d, self.gamma * self.lam)
            self.ret_c_buf[path_slice, i] = self.adv_c_buf[path_slice, i] + self.val_c_buf[path_slice, i]

        # Episode discounted cost sum (for dual update signal)
        traj_costs = self.cost_buf[path_slice]  # (T, m)
        disc = (self.gamma ** np.arange(traj_costs.shape[0], dtype=np.float32)).reshape(-1, 1)
        ep_Jc = (traj_costs * disc).sum(axis=0)
        self._ep_cost_returns.append(ep_Jc.astype(np.float32))

        # Episode discounted reward sum (for logging)
        traj_rews = self.rew_buf[path_slice]
        disc_r = (self.gamma ** np.arange(traj_rews.shape[0], dtype=np.float32))
        ep_Jr = float((traj_rews * disc_r).sum())
        self._ep_ret_returns.append(ep_Jr)

        self.path_start_idx = self.ptr

    def get_Jc_estimate(self) -> np.ndarray:
        if len(self._ep_cost_returns) == 0:
            return np.zeros((self.cost_buf.shape[1],), dtype=np.float32)
        return np.mean(np.stack(self._ep_cost_returns, axis=0), axis=0).astype(np.float32)

    def get_Jc_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._ep_cost_returns) == 0:
            zeros = np.zeros((self.cost_buf.shape[1],), dtype=np.float32)
            return zeros, zeros
        stacked = np.stack(self._ep_cost_returns, axis=0).astype(np.float32)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        return mean.astype(np.float32), std.astype(np.float32)

    def get_Jr_stats(self) -> Tuple[float, float]:
        if len(self._ep_ret_returns) == 0:
            return 0.0, 0.0
        arr = np.asarray(self._ep_ret_returns, dtype=np.float32)
        return float(np.mean(arr)), float(np.std(arr))

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.ptr > 0, "Buffer is empty."
        end = int(self.ptr)
        return dict(
            obs=torch.as_tensor(self.obs_buf[:end], dtype=torch.float32),
            act=torch.as_tensor(self.act_buf[:end], dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf[:end], dtype=torch.float32),
            adv_r=torch.as_tensor(self.adv_r_buf[:end], dtype=torch.float32),
            ret_r=torch.as_tensor(self.ret_r_buf[:end], dtype=torch.float32),
            adv_c=torch.as_tensor(self.adv_c_buf[:end], dtype=torch.float32),
            ret_c=torch.as_tensor(self.ret_c_buf[:end], dtype=torch.float32),
        )


class RPGPDPPOAgent:
    """Single-loop RPG-PD agent with PPO primal updates."""
    def __init__(self, obs_dim: int, act_dim: int, num_costs: int, cfg: Optional[RPGPDPPOConfig] = None, device: str = "cpu"):
        self.cfg = cfg or RPGPDPPOConfig()
        self.device = torch.device(device)
        self.num_costs = int(num_costs)

        self.pi = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.v_r = ValueNet(obs_dim, out_dim=1, hidden_sizes=self.cfg.hidden_sizes, squeeze_output=True).to(self.device)
        self.v_c = ValueNet(
            obs_dim,
            out_dim=num_costs,
            hidden_sizes=self.cfg.hidden_sizes,
            squeeze_output=False,
        ).to(self.device)

        self.vr_opt = optim.Adam(self.v_r.parameters(), lr=self.cfg.vf_lr)
        self.vc_opt = optim.Adam(self.v_c.parameters(), lr=self.cfg.vf_lr)

        self.lam = np.full((self.num_costs,), float(self.cfg.lambda_init), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            a = self.pi.deterministic(obs_t)
            logp = self.pi.log_prob(obs_t, a)
        else:
            a, logp = self.pi.sample(obs_t)
        v_r = self.v_r(obs_t)
        v_c = self.v_c(obs_t)
        v_c_np = np.atleast_1d(v_c.squeeze(0).cpu().numpy().astype(np.float32))
        return (
            a.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp.item()),
            float(v_r.item()),
            v_c_np,
        )

    @staticmethod
    def _project_lambda(lam: np.ndarray, lam_max: float) -> np.ndarray:
        return np.clip(lam, 0.0, float(lam_max)).astype(np.float32)

    def _dual_update(self, Jc_est: np.ndarray, cost_limits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        scale = (1.0 - cfg.gamma) if cfg.dual_scale_one_minus_gamma else 1.0
        viol = scale * (np.asarray(Jc_est, dtype=np.float32) - np.asarray(cost_limits, dtype=np.float32))
        lam_new = (1.0 - cfg.dual_lr * cfg.dual_tau) * self.lam + cfg.dual_lr * viol
        self.lam = self._project_lambda(lam_new, cfg.lambda_max)
        return self.lam.copy(), viol.astype(np.float32)

    def _compute_lagrangian_adv(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        lam_t = torch.as_tensor(self.lam, dtype=torch.float32, device=adv_c.device).view(1, -1)
        return adv_r - (adv_c * lam_t).sum(dim=-1)

    def _ppo_update(self, obs: torch.Tensor, act: torch.Tensor, logp_old: torch.Tensor, adv_L: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        self.pi.train()
        pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.pi_lr)

        early_stop_iter = 0
        for it in range(cfg.train_pi_iters):
            logp = self.pi.log_prob(obs, act)
            ratio = torch.exp(logp - logp_old)
            ratio_clip = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)

            surr1 = ratio * adv_L
            surr2 = ratio_clip * adv_L
            loss_pi = -(torch.min(surr1, surr2)).mean()

            # H(pi) regularizer via entropy bonus
            if cfg.entropy_coef != 0.0:
                dist = self.pi._dist(obs)  # pre-tanh entropy approximation
                ent = dist.entropy().sum(dim=-1).mean()
                loss_pi = loss_pi - cfg.entropy_coef * ent

            pi_opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.max_grad_norm)
            pi_opt.step()

            with torch.no_grad():
                kl = (logp_old - self.pi.log_prob(obs, act)).mean().item()
            if kl > cfg.target_kl:
                early_stop_iter = it + 1
                break

        with torch.no_grad():
            logp = self.pi.log_prob(obs, act)
            kl = (logp_old - logp).mean().item()
            entropy = self.pi._dist(obs).entropy().sum(dim=-1).mean().item()

        return {"kl": float(kl), "entropy": float(entropy), "pi_early_stop_iter": float(early_stop_iter)}

    def _update_critics(self, obs: torch.Tensor, ret_r: torch.Tensor, ret_c: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        obs = obs.to(self.device)
        ret_r = ret_r.to(self.device)
        ret_c = ret_c.to(self.device)

        loss_vr_val = 0.0
        for _ in range(cfg.train_v_iters):
            v = self.v_r(obs)
            loss_vr = ((v - ret_r) ** 2).mean()
            self.vr_opt.zero_grad(set_to_none=True)
            loss_vr.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.v_r.parameters(), cfg.max_grad_norm)
            self.vr_opt.step()
            loss_vr_val = float(loss_vr.detach().cpu().item())

        loss_vc_val = 0.0
        for _ in range(cfg.train_v_iters):
            v = self.v_c(obs)
            loss_vc = ((v - ret_c) ** 2).mean()
            self.vc_opt.zero_grad(set_to_none=True)
            loss_vc.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.v_c.parameters(), cfg.max_grad_norm)
            self.vc_opt.step()
            loss_vc_val = float(loss_vc.detach().cpu().item())

        return {"loss_vr": float(loss_vr_val), "loss_vc": float(loss_vc_val)}

    def update(self, data: Dict[str, torch.Tensor], cost_limits: np.ndarray, Jc_est: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        obs = data["obs"].to(self.device)
        act = data["act"].to(self.device)
        logp_old = data["logp"].to(self.device)
        adv_r = data["adv_r"].to(self.device)
        ret_r = data["ret_r"].to(self.device)
        adv_c = data["adv_c"].to(self.device)
        ret_c = data["ret_c"].to(self.device)

        if cfg.normalize_advantages:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (adv_c.std(dim=0, keepdim=True) + 1e-8)

        adv_L = self._compute_lagrangian_adv(adv_r, adv_c)

        pi_log = self._ppo_update(obs, act, logp_old, adv_L.detach())
        v_log = self._update_critics(obs, ret_r, ret_c)

        lam_new, viol = self._dual_update(Jc_est=Jc_est, cost_limits=cost_limits)

        out = {**pi_log, **v_log}
        for i in range(self.num_costs):
            out[f"lambda_{i}"] = float(lam_new[i])
            out[f"viol_{i}"] = float(viol[i])
            out[f"Jc_{i}"] = float(np.asarray(Jc_est, dtype=np.float32)[i])
        return out


class RPGPDPPOTrainer:
    def __init__(self, env: Any, agent: RPGPDPPOAgent, cost_limits: np.ndarray, cfg: Optional[RPGPDPPOConfig] = None):
        self.env = env
        self.agent = agent
        self.cfg = cfg or agent.cfg
        self.cost_limits = np.asarray(cost_limits, dtype=np.float32)

        rollouts_per_epoch = int(self.cfg.num_roll_out) if self.cfg.num_roll_out is not None else 1
        buf_size = int(rollouts_per_epoch) * int(self.cfg.max_ep_len)

        self.buf = RolloutBuffer(
            obs_dim=env.obs_dim(),
            act_dim=env.act_dim(),
            num_costs=len(cost_limits),
            size=buf_size,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

    def collect_epoch(
        self,
        seed: Optional[int] = None,
        live: bool = False,
        random_seed_each_rollout: bool = False,
    ) -> Dict[str, float]:
        cfg = self.cfg
        ep_rets: List[float] = []
        ep_costs: List[np.ndarray] = []
        ep_lens: List[int] = []

        first_seed = int(np.random.randint(0, 2**31 - 1)) if random_seed_each_rollout else seed
        obs, info = self.env.reset(seed=first_seed)
        ep_ret = 0.0
        ep_cost = np.zeros((len(self.cost_limits),), dtype=np.float32)
        ep_len = 0

        rollouts_remaining = int(cfg.num_roll_out) if cfg.num_roll_out is not None else 1

        while True:
            act, logp, v_r, v_c = self.agent.act(obs, deterministic=False)
            next_obs, rew, terminated, truncated, info = self.env.step(act)
            costs = np.asarray(info["costs"], dtype=np.float32)
            if live:
                self.env.render()

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

                rollouts_remaining -= 1
                if rollouts_remaining <= 0:
                    break

                next_seed = int(np.random.randint(0, 2**31 - 1)) if random_seed_each_rollout else None
                obs, info = self.env.reset(seed=next_seed)
                ep_ret = 0.0
                ep_cost[:] = 0.0
                ep_len = 0

        stats: Dict[str, float] = {
            "EpRetMean": float(np.mean(ep_rets)) if ep_rets else 0.0,
            "EpRetStd": float(np.std(ep_rets)) if ep_rets else 0.0,
            "EpLenMean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        }
        if ep_costs:
            C = np.stack(ep_costs, axis=0)
            for i in range(C.shape[1]):
                stats[f"EpCost{i}Mean"] = float(np.mean(C[:, i]))
                stats[f"EpCost{i}Std"] = float(np.std(C[:, i]))
        return stats

    def train_epoch(self) -> Dict[str, float]:
        data = self.buf.get()
        Jc_mean, Jc_std = self.buf.get_Jc_stats()
        Jr_mean, Jr_std = self.buf.get_Jr_stats()
        log = self.agent.update(data, cost_limits=self.cost_limits, Jc_est=Jc_mean)
        for i in range(len(self.cost_limits)):
            log[f"Jc_mean_{i}"] = float(Jc_mean[i])
            log[f"Jc_std_{i}"] = float(Jc_std[i])
        log["Jr_mean"] = float(Jr_mean)
        log["Jr_std"] = float(Jr_std)
        self.buf.reset()
        return log
