"""
opgpd_ppo_agent.py

Optimistic Policy Gradient Primal-Dual (OPG-PD) integrated with PPO-style clipped policy updates.

High-level idea (extragradient / optimism):
  Given current (hat) primal-dual iterate (π̂_k, λ̂_k):
    1) Rollout with π̂_k and do a PPO update on the Lagrangian to obtain a *predicted* policy π_k.
       Also do a *predicted* dual update to obtain λ_k.
    2) Rollout with π_k and do a PPO update on the Lagrangian to obtain the next hat policy π̂_{k+1}.
       Also do a *corrected* dual update to obtain λ̂_{k+1}.

We model multi-constraint CMDP with per-step costs c_i(s,a) and limits d_i.
We use a standard Lagrangian objective (reward - λ·cost) and a projected dual ascent:
  λ <- Proj_{[0, λ_max]} ( (1 - ηλ τ) λ + ηλ * scale * (J_c - d) )

Notes:
- We keep the PPO update as a practical "proximal" subroutine.
- For the correction step, we optionally add a KL-to-anchor penalty to keep the corrected policy close to π̂_k.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing to keep actions in (-1, 1)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(255, 255)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, obs: torch.Tensor):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).clamp(1e-4, 10.0)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _tanh_correction(u: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, obs_dim: int, out_dim: int = 1, hidden_sizes=(255, 255)):
        super().__init__()
        self.v_net = MLP(obs_dim, out_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)


@dataclass
class OPGPDPPOConfig:
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

    # Data collection (per rollout)
    steps_per_epoch: int = 30_000
    max_ep_len: int = 1000
    num_roll_out: Optional[int] = None  # if set, collect this many episodes per rollout

    # Nets
    hidden_sizes: Tuple[int, int] = (255, 255)

    # Optional entropy bonus
    entropy_coef: float = 0.0

    # Advantage normalization
    normalize_advantages: bool = True

    # Dual update
    dual_lr: float = 0.5
    dual_tau: float = 0.0  # shrinkage (regularization) on lambda
    lambda_init: float = 0.0
    lambda_max: float = 1000.0
    dual_scale_one_minus_gamma: bool = True

    # OPG-PD correction anchoring (optional)
    anchor_kl_coef: float = 0.0  # if >0, add beta * KL(π || π_anchor) in correction step


class RolloutBuffer:
    """Stores one rollout batch (steps_per_epoch steps OR num_roll_out episodes)."""
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

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self._ep_cost_returns = []

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

        # Cost GAE (multi)
        for i in range(self.cost_buf.shape[1]):
            c = np.append(self.cost_buf[path_slice, i], float(last_val_c[i]))
            v = np.append(self.val_c_buf[path_slice, i], float(last_val_c[i]))
            d = c[:-1] + self.gamma * v[1:] - v[:-1]
            self.adv_c_buf[path_slice, i] = discount_cumsum(d, self.gamma * self.lam)
            self.ret_c_buf[path_slice, i] = self.adv_c_buf[path_slice, i] + self.val_c_buf[path_slice, i]

        # Episode discounted cost sum from trajectory start
        traj_costs = self.cost_buf[path_slice]  # (T, m)
        disc = (self.gamma ** np.arange(traj_costs.shape[0], dtype=np.float32)).reshape(-1, 1)
        ep_Jc = (traj_costs * disc).sum(axis=0)
        self._ep_cost_returns.append(ep_Jc.astype(np.float32))

        self.path_start_idx = self.ptr

    def get_Jc_estimate(self) -> np.ndarray:
        if len(self._ep_cost_returns) == 0:
            return np.zeros((self.cost_buf.shape[1],), dtype=np.float32)
        return np.mean(np.stack(self._ep_cost_returns, axis=0), axis=0).astype(np.float32)

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


class OPGPDPPOAgent:
    """
    Holds:
      - π̂ (hat) policy: pi_hat
      - π (predicted) policy: pi_pred
      - Value nets: v_r, v_c
      - Dual variables: lambda_hat, lambda_pred
    """
    def __init__(self, obs_dim: int, act_dim: int, num_costs: int, cfg: Optional[OPGPDPPOConfig] = None, device: str = "cpu"):
        self.cfg = cfg or OPGPDPPOConfig()
        self.device = torch.device(device)
        self.num_costs = int(num_costs)

        self.pi_hat = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.pi_pred = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.pi_anchor = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes=self.cfg.hidden_sizes).to(self.device)

        self.v_r = ValueNet(obs_dim, out_dim=1, hidden_sizes=self.cfg.hidden_sizes).to(self.device)
        self.v_c = ValueNet(obs_dim, out_dim=num_costs, hidden_sizes=self.cfg.hidden_sizes).to(self.device)

        self.vr_opt = optim.Adam(self.v_r.parameters(), lr=self.cfg.vf_lr)
        self.vc_opt = optim.Adam(self.v_c.parameters(), lr=self.cfg.vf_lr)

        # dual variables (kept on CPU as numpy-like, but also as torch when needed)
        self.lambda_hat = np.full((self.num_costs,), float(self.cfg.lambda_init), dtype=np.float32)
        self.lambda_pred = np.full((self.num_costs,), float(self.cfg.lambda_init), dtype=np.float32)

        self.active_policy: str = "hat"  # "hat" or "pred"

    def set_active_policy(self, which: str) -> None:
        assert which in ("hat", "pred")
        self.active_policy = which

    def _current_pi(self) -> TanhGaussianPolicy:
        return self.pi_hat if self.active_policy == "hat" else self.pi_pred

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        pi = self._current_pi()
        if deterministic:
            a = pi.deterministic(obs_t)
            logp = pi.log_prob(obs_t, a)
        else:
            a, logp = pi.sample(obs_t)
        v_r = self.v_r(obs_t)
        v_c = self.v_c(obs_t)
        v_c_np = np.atleast_1d(v_c.squeeze(0).cpu().numpy().astype(np.float32))
        return a.squeeze(0).cpu().numpy().astype(np.float32), float(logp.item()), float(v_r.item()), v_c_np

    @staticmethod
    def _project_lambda(lam: np.ndarray, lam_max: float) -> np.ndarray:
        return np.clip(lam, 0.0, float(lam_max)).astype(np.float32)

    def _dual_update(self, lam_center: np.ndarray, Jc_est: np.ndarray, cost_limits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        λ <- Proj( (1-ητ) λ + η * scale * (Jc - d) )
        Returns (new_lambda, scaled_violation).
        """
        cfg = self.cfg
        scale = (1.0 - cfg.gamma) if cfg.dual_scale_one_minus_gamma else 1.0
        viol = scale * (np.asarray(Jc_est, dtype=np.float32) - np.asarray(cost_limits, dtype=np.float32))
        lam_new = (1.0 - cfg.dual_lr * cfg.dual_tau) * np.asarray(lam_center, dtype=np.float32) + cfg.dual_lr * viol
        lam_new = self._project_lambda(lam_new, cfg.lambda_max)
        return lam_new, viol.astype(np.float32)

    def _ppo_update(
        self,
        policy: TanhGaussianPolicy,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp_old: torch.Tensor,
        adv: torch.Tensor,
        *,
        anchor_policy: Optional[TanhGaussianPolicy] = None,
        anchor_kl_coef: float = 0.0,
    ) -> Dict[str, float]:
        """
        Full-batch PPO update on `policy` using clipped objective with `adv`.
        Optionally adds anchor KL penalty: beta * KL(N_new || N_anchor) (pre-tanh Normal).
        """
        cfg = self.cfg
        device = self.device
        policy.train()

        # Fresh optimizer each call (important because we sometimes overwrite initial weights)
        pi_opt = optim.Adam(policy.parameters(), lr=cfg.pi_lr)

        pi_info: Dict[str, float] = {}
        for it in range(cfg.train_pi_iters):
            logp = policy.log_prob(obs, act)
            ratio = torch.exp(logp - logp_old)

            clip_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)

            surr1 = ratio * adv
            surr2 = clip_ratio * adv
            loss_pi = -(torch.min(surr1, surr2)).mean()

            # Entropy bonus (rough; on pre-tanh Normal)
            if cfg.entropy_coef != 0.0:
                dist = policy._dist(obs)
                ent = dist.entropy().sum(dim=-1).mean()
                loss_pi = loss_pi - cfg.entropy_coef * ent

            # Anchor KL penalty (correction step)
            if anchor_policy is not None and anchor_kl_coef > 0.0:
                with torch.no_grad():
                    pass
                dist_new = policy._dist(obs)
                dist_anchor = anchor_policy._dist(obs)
                # KL between two Normals (per-dimension), averaged over batch
                kl_anchor = torch.distributions.kl_divergence(dist_new, dist_anchor).sum(dim=-1).mean()
                loss_pi = loss_pi + float(anchor_kl_coef) * kl_anchor
                pi_info["kl_anchor"] = float(kl_anchor.detach().cpu().item())
            else:
                pi_info["kl_anchor"] = 0.0

            pi_opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            pi_opt.step()

            # Early stop via approximate KL using action log-probs
            with torch.no_grad():
                kl = (logp_old - policy.log_prob(obs, act)).mean().item()
            if kl > cfg.target_kl:
                pi_info["pi_early_stop_iter"] = float(it + 1)
                break

        # Final stats
        with torch.no_grad():
            logp = policy.log_prob(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            surr1 = ratio * adv
            surr2 = clip_ratio * adv
            loss_pi_val = -(torch.min(surr1, surr2)).mean().item()
            entropy = policy._dist(obs).entropy().sum(dim=-1).mean().item()
            kl = (logp_old - logp).mean().item()

        out = {
            "loss_pi": float(loss_pi_val),
            "kl": float(kl),
            "entropy": float(entropy),
            "pi_early_stop_iter": float(pi_info.get("pi_early_stop_iter", 0.0)),
            "kl_anchor": float(pi_info.get("kl_anchor", 0.0)),
        }
        return out

    def _update_critics(self, obs: torch.Tensor, ret_r: torch.Tensor, ret_c: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        device = self.device

        obs = obs.to(device)
        ret_r = ret_r.to(device)
        ret_c = ret_c.to(device)

        # Reward critic
        for _ in range(cfg.train_v_iters):
            v = self.v_r(obs)
            loss_vr = ((v - ret_r) ** 2).mean()
            self.vr_opt.zero_grad(set_to_none=True)
            loss_vr.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.v_r.parameters(), cfg.max_grad_norm)
            self.vr_opt.step()

        # Cost critic
        for _ in range(cfg.train_v_iters):
            v = self.v_c(obs)
            loss_vc = ((v - ret_c) ** 2).mean()
            self.vc_opt.zero_grad(set_to_none=True)
            loss_vc.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.v_c.parameters(), cfg.max_grad_norm)
            self.vc_opt.step()

        return {"loss_vr": float(loss_vr.detach().cpu().item()), "loss_vc": float(loss_vc.detach().cpu().item())}

    def _compute_lagrangian_adv(self, adv_r: torch.Tensor, adv_c: torch.Tensor, lam: np.ndarray) -> torch.Tensor:
        lam_t = torch.as_tensor(lam, dtype=torch.float32, device=adv_c.device).view(1, -1)
        return adv_r - (adv_c * lam_t).sum(dim=-1)

    def prediction_update(self, data: Dict[str, torch.Tensor], cost_limits: np.ndarray, Jc_est: np.ndarray) -> Dict[str, float]:
        """
        Update π_pred and λ_pred using data collected by π_hat.
        """
        cfg = self.cfg
        device = self.device

        # Copy π_hat -> π_pred
        self.pi_pred.load_state_dict(self.pi_hat.state_dict())

        obs = data["obs"].to(device)
        act = data["act"].to(device)
        logp_old = data["logp"].to(device)

        adv_r = data["adv_r"].to(device)
        ret_r = data["ret_r"].to(device)
        adv_c = data["adv_c"].to(device)
        ret_c = data["ret_c"].to(device)

        if cfg.normalize_advantages:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (adv_c.std(dim=0, keepdim=True) + 1e-8)

        adv_L = self._compute_lagrangian_adv(adv_r, adv_c, self.lambda_hat)

        # PPO update on predicted policy
        pi_log = self._ppo_update(
            self.pi_pred,
            obs=obs,
            act=act,
            logp_old=logp_old,
            adv=adv_L.detach(),  # advantage treated as constant wrt theta
            anchor_policy=None,
            anchor_kl_coef=0.0,
        )

        # Critic update
        v_log = self._update_critics(obs, ret_r, ret_c)

        # Dual predicted update
        self.lambda_pred, viol = self._dual_update(self.lambda_hat, Jc_est=Jc_est, cost_limits=cost_limits)

        out = {
            **{f"lambda_hat_{i}": float(self.lambda_hat[i]) for i in range(self.num_costs)},
            **{f"lambda_pred_{i}": float(self.lambda_pred[i]) for i in range(self.num_costs)},
            **{f"viol_hat_{i}": float(viol[i]) for i in range(self.num_costs)},
            **{f"Jc_hat_{i}": float(np.asarray(Jc_est, dtype=np.float32)[i]) for i in range(self.num_costs)},
            "step": 1.0,  # prediction
        }
        out.update(pi_log)
        out.update(v_log)
        return out

    def correction_update(self, data: Dict[str, torch.Tensor], cost_limits: np.ndarray, Jc_est: np.ndarray) -> Dict[str, float]:
        """
        Update π_hat and λ_hat using data collected by π_pred and using λ_pred for the Lagrangian advantage.
        """
        cfg = self.cfg
        device = self.device

        # Save anchor (old π_hat) for optional anchoring penalty
        self.pi_anchor.load_state_dict(self.pi_hat.state_dict())
        for p in self.pi_anchor.parameters():
            p.requires_grad_(False)

        # Initialize π_hat from π_pred (improves PPO stability since logp_old is from π_pred)
        self.pi_hat.load_state_dict(self.pi_pred.state_dict())

        obs = data["obs"].to(device)
        act = data["act"].to(device)
        logp_old = data["logp"].to(device)  # from π_pred rollout

        adv_r = data["adv_r"].to(device)
        ret_r = data["ret_r"].to(device)
        adv_c = data["adv_c"].to(device)
        ret_c = data["ret_c"].to(device)

        if cfg.normalize_advantages:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (adv_c.std(dim=0, keepdim=True) + 1e-8)

        adv_L = self._compute_lagrangian_adv(adv_r, adv_c, self.lambda_pred)

        pi_log = self._ppo_update(
            self.pi_hat,
            obs=obs,
            act=act,
            logp_old=logp_old,
            adv=adv_L.detach(),
            anchor_policy=self.pi_anchor if cfg.anchor_kl_coef > 0.0 else None,
            anchor_kl_coef=cfg.anchor_kl_coef,
        )

        v_log = self._update_critics(obs, ret_r, ret_c)

        # Dual corrected update (centered at previous λ̂, using costs from π_pred rollout)
        lambda_hat_next, viol = self._dual_update(self.lambda_hat, Jc_est=Jc_est, cost_limits=cost_limits)
        self.lambda_hat = lambda_hat_next

        out = {
            **{f"lambda_hat_next_{i}": float(self.lambda_hat[i]) for i in range(self.num_costs)},
            **{f"lambda_pred_{i}": float(self.lambda_pred[i]) for i in range(self.num_costs)},
            **{f"viol_pred_{i}": float(viol[i]) for i in range(self.num_costs)},
            **{f"Jc_pred_{i}": float(np.asarray(Jc_est, dtype=np.float32)[i]) for i in range(self.num_costs)},
            "step": 2.0,  # correction
        }
        out.update(pi_log)
        out.update(v_log)
        return out


class OPGPDPPOTrainer:
    def __init__(self, env: Any, agent: OPGPDPPOAgent, cost_limits: np.ndarray, cfg: Optional[OPGPDPPOConfig] = None):
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
            lam=self.cfg.lam,
        )

    def collect_epoch(self, policy: str, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Collect one rollout batch using either π̂ ("hat") or π ("pred").
        """
        cfg = self.cfg
        self.agent.set_active_policy(policy)

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
                    if done:
                        last_val_r = 0.0
                        last_val_c = np.zeros_like(ep_cost, dtype=np.float32)
                    else:
                        # epoch cutoff bootstrapping
                        _, _, last_val_r, last_val_c = self.agent.act(obs, deterministic=True)
                    self.buf.finish_path(last_val_r=last_val_r, last_val_c=last_val_c)

                    ep_rets.append(ep_ret)
                    ep_costs.append(ep_cost.copy())
                    ep_lens.append(ep_len)

                    obs, info = self.env.reset(seed=None)
                    ep_ret = 0.0
                    ep_cost[:] = 0.0
                    ep_len = 0

        stats: Dict[str, float] = {
            "EpRetMean": float(np.mean(ep_rets)) if ep_rets else 0.0,
            "EpLenMean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        }
        if ep_costs:
            C = np.stack(ep_costs, axis=0)
            for i in range(C.shape[1]):
                stats[f"EpCost{i}Mean"] = float(np.mean(C[:, i]))
        return stats

    def train_prediction(self) -> Dict[str, float]:
        data = self.buf.get()
        Jc_est = self.buf.get_Jc_estimate()
        log = self.agent.prediction_update(data, cost_limits=self.cost_limits, Jc_est=Jc_est)
        self.buf.reset()
        return log

    def train_correction(self) -> Dict[str, float]:
        data = self.buf.get()
        Jc_est = self.buf.get_Jc_estimate()
        log = self.agent.correction_update(data, cost_limits=self.cost_limits, Jc_est=Jc_est)
        self.buf.reset()
        return log
