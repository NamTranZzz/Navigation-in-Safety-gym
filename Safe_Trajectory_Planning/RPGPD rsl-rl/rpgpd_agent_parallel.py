from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import torch
from tensordict import TensorDict

from cmdp_wrapper import CMDPConfig, RewardCostWrapper
from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv
from rpgpd_agent import RPGPDTrainerHook, RslRPGPDConfig, build_train_cfg


def _make_wrapped_env(mujoco_cfg_dict: Dict[str, Any], cmdp_cfg_dict: Dict[str, Any]) -> RewardCostWrapper:
    mujoco_cfg = MujocoPointNavConfig(**mujoco_cfg_dict)
    cmdp_cfg = CMDPConfig(**cmdp_cfg_dict)
    env = MujocoPointNavEnv(mujoco_cfg, render_mode=None)
    return RewardCostWrapper(env, cfg=cmdp_cfg)


def _worker_main(conn, mujoco_cfg_dict: Dict[str, Any], cmdp_cfg_dict: Dict[str, Any]) -> None:
    env = _make_wrapped_env(mujoco_cfg_dict, cmdp_cfg_dict)
    num_costs = int(len(env.cfg.cost_limits))
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=payload)
                conn.send((np.asarray(obs, dtype=np.float32).reshape(-1), info))
            elif cmd == "step":
                o2, r, terminated, truncated, info = env.step(np.asarray(payload, dtype=np.float32))

                # Send compact, fixed-shape payload to reduce per-step IPC cost.
                costs = np.asarray(info.get("costs", np.zeros((num_costs,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                if costs.size == 0:
                    costs = np.zeros((num_costs,), dtype=np.float32)
                elif costs.size == 1 and num_costs > 1:
                    costs = np.full((num_costs,), float(costs[0]), dtype=np.float32)
                else:
                    costs = costs[:num_costs]
                    if costs.shape[0] < num_costs:
                        pad = np.zeros((num_costs - costs.shape[0],), dtype=np.float32)
                        costs = np.concatenate([costs, pad], axis=0)

                reward_unshaped = info.get("reward_unshaped", r)
                try:
                    reward_unshaped = float(np.asarray(reward_unshaped, dtype=np.float32).reshape(-1)[0])
                except Exception:
                    reward_unshaped = float(r)

                success = 0.0
                for key in ("success", "is_success", "goal_met", "goal_achieved", "task_success"):
                    if key in info:
                        try:
                            success = 1.0 if float(np.asarray(info[key]).reshape(-1)[0]) > 0.5 else 0.0
                        except Exception:
                            success = 0.0
                        break

                collision_increment = None
                for key in ("collision", "collisions", "num_collisions", "contact", "contacts"):
                    if key in info:
                        try:
                            collision_increment = max(0.0, float(np.asarray(info[key]).reshape(-1)[0]))
                        except Exception:
                            collision_increment = None
                        break
                if collision_increment is None:
                    collision_increment = max(0.0, float(costs[0]) if num_costs > 0 else 0.0)

                conn.send(
                    (
                        np.asarray(o2, dtype=np.float32).reshape(-1),
                        reward_unshaped,
                        bool(terminated),
                        bool(truncated),
                        costs.astype(np.float32, copy=False),
                        float(success),
                        float(collision_increment),
                    )
                )
            elif cmd == "close":
                env.close()
                conn.send(True)
                break
            else:
                raise RuntimeError(f"Unknown worker command: {cmd}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


class _SubprocWorker:
    def __init__(self, ctx: mp.context.BaseContext, mujoco_cfg_dict: Dict[str, Any], cmdp_cfg_dict: Dict[str, Any]):
        parent_conn, child_conn = ctx.Pipe()
        self.conn = parent_conn
        self.proc = ctx.Process(target=_worker_main, args=(child_conn, mujoco_cfg_dict, cmdp_cfg_dict), daemon=True)
        self.proc.start()
        child_conn.close()

    def request(self, cmd: str, payload: Any = None) -> None:
        self.conn.send((cmd, payload))

    def response(self):
        return self.conn.recv()

    def close(self) -> None:
        try:
            self.request("close", None)
            _ = self.response()
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.join(timeout=1.0)
            if self.proc.is_alive():
                self.proc.terminate()


class ParallelSafetyGymVecEnv:
    """Parallel (subprocess) vectorized env adapter for rsl-rl OnPolicyRunner."""

    def __init__(
        self,
        mujoco_cfg: MujocoPointNavConfig,
        cmdp_cfg: CMDPConfig,
        num_envs: int,
        device: str,
        seed: int,
        gamma: float = 0.99,
        start_method: str = "spawn",
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self._base_seed = int(seed)
        self._next_seed = int(seed)
        self.gamma = float(gamma)

        # Probe one local env for dimensions and metadata.
        probe_env = _make_wrapped_env(asdict(mujoco_cfg), asdict(cmdp_cfg))
        self.num_obs = int(probe_env.obs_dim())
        self.num_actions = int(probe_env.act_dim())
        self.num_privileged_obs = 0
        self.max_episode_length = int(getattr(probe_env.cfg, "max_steps", 1000))
        self.cfg = {"env_name": "ParallelSafetyGymVecEnv", "max_episode_length": self.max_episode_length}
        self._num_costs = int(len(probe_env.cfg.cost_limits))
        probe_env.close()

        self._ctx = mp.get_context(str(start_method))
        self._workers: List[_SubprocWorker] = [
            _SubprocWorker(self._ctx, asdict(mujoco_cfg), asdict(cmdp_cfg)) for _ in range(self.num_envs)
        ]

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
        return self._num_costs

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

    def get_observations(self):
        return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        for i, w in enumerate(self._workers):
            w.request("reset", self._base_seed + i)
        obs = [w.response()[0] for w in self._workers]
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
        for i, w in enumerate(self._workers):
            w.request("step", actions_np[i])

        results = [w.response() for w in self._workers]
        next_obs: List[np.ndarray | None] = [None for _ in range(self.num_envs)]
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        step_costs = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)

        done_returns: List[float] = []
        done_ret_unshaped: List[float] = []
        done_ret_unshaped_disc: List[float] = []
        done_cost0: List[float] = []
        done_cost0_disc: List[float] = []
        done_success: List[float] = []
        done_collision: List[float] = []
        done_indices: List[int] = []

        for i, out in enumerate(results):
            o2, reward_unshaped, terminated, truncated, costs, success, collision_increment = out
            done = bool(terminated or truncated)
            c0 = float(costs[0]) if costs.size > 0 else 0.0

            rewards[i] = float(reward_unshaped)
            step_costs[i] = torch.as_tensor(costs, dtype=torch.float32, device=self.device)

            self._episode_return_buf[i] += float(reward_unshaped)
            self._episode_ret_unshaped_buf[i] += float(reward_unshaped)
            self._episode_ret_unshaped_disc_buf[i] += self._episode_disc_factor_buf[i] * float(reward_unshaped)
            self._episode_cost_buf[i] += torch.as_tensor(costs, dtype=torch.float32, device=self.device)
            self._episode_cost_disc_buf[i] += self._episode_disc_factor_buf[i] * torch.as_tensor(
                costs,
                dtype=torch.float32,
                device=self.device,
            )
            self._episode_disc_factor_buf[i] *= float(self.gamma)
            self._episode_collision_buf[i] += float(collision_increment)
            self.episode_length_buf[i] += 1

            if done:
                dones[i] = True
                if bool(truncated):
                    time_outs[i] = 1.0

                done_returns.append(float(self._episode_return_buf[i].item()))
                done_ret_unshaped.append(float(self._episode_ret_unshaped_buf[i].item()))
                done_ret_unshaped_disc.append(float(self._episode_ret_unshaped_disc_buf[i].item()))
                done_cost0.append(float(self._episode_cost_buf[i, 0].item()))
                done_cost0_disc.append(float(self._episode_cost_disc_buf[i, 0].item()))
                self._done_costs.append(self._episode_cost_buf[i].detach().cpu().numpy().astype(np.float32))
                self._done_costs_discounted.append(self._episode_cost_disc_buf[i].detach().cpu().numpy().astype(np.float32))
                done_success.append(float(success))
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

                done_indices.append(i)
            else:
                next_obs[i] = np.asarray(o2, dtype=np.float32).reshape(-1)

        # Batch reset requests to reduce round-trip synchronization on done workers.
        for i in done_indices:
            self._next_seed += 1
            self._workers[i].request("reset", self._next_seed)
        for i in done_indices:
            o2, _ = self._workers[i].response()
            next_obs[i] = np.asarray(o2, dtype=np.float32).reshape(-1)

        self._obs_buf = torch.as_tensor(np.stack(next_obs, axis=0), dtype=torch.float32, device=self.device)

        extras: Dict[str, Any] = {
            "time_outs": time_outs,
            "costs": step_costs,
        }
        if done_returns:
            extras["episode"] = {
                "EpRetUnshaped": torch.as_tensor(done_ret_unshaped, dtype=torch.float32, device=self.device),
                "EpRetUnshapedDiscounted": torch.as_tensor(done_ret_unshaped_disc, dtype=torch.float32, device=self.device),
                "EpCost0": torch.as_tensor(done_cost0, dtype=torch.float32, device=self.device),
                "EpCost0Discounted": torch.as_tensor(done_cost0_disc, dtype=torch.float32, device=self.device),
                "success_rate": torch.as_tensor(done_success, dtype=torch.float32, device=self.device),
                "collision_count": torch.as_tensor(done_collision, dtype=torch.float32, device=self.device),
            }
        obs_td = TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs], device=self.device)
        return obs_td, rewards, dones, extras

    def close(self):
        for w in self._workers:
            w.close()


__all__ = [
    "RslRPGPDConfig",
    "build_train_cfg",
    "RPGPDTrainerHook",
    "ParallelSafetyGymVecEnv",
]
