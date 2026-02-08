
"""
nav_env.py

A lightweight 2D navigation environment (dynamics only):
- Agent: 2D point-mass with first-order velocity tracking (can't stop instantly).
- Obstacles:
  * Pillars: impassible circles (collision projection).
  * Hazards: static virtual circles (passable, but collision flags exposed in info).
  * Dynamic obstacles: moving obstacles (impassible) with velocity + boundary bounce.

No reward/cost is computed here. Use cmdp_wrapper.py for that.

This file has *no* dependency on gym/gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


def _seeded_rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


@dataclass
class Nav2DConfig:
    # World / episode
    world_size: float = 10.0              # world is square [-world_size/2, +world_size/2]
    dt: float = 0.05
    max_steps: int = 1000                # matches paper Navigation rollout length (T=1e3)
    agent_radius: float = 0.15
    goal_radius: float = 0.35

    # Agent dynamics
    max_speed: float = 1.0
    vel_time_constant: float = 0.25      # smaller -> more responsive, still smooth

    # Obstacles
    num_pillars: int = 4
    pillar_radius_range: Tuple[float, float] = (0.30, 0.55)

    num_hazards_static: int = 8
    hazard_static_radius: float = 0.25

    num_dynamic_obstacles: int = 4
    dynamic_obstacle_radius: float = 0.22
    dynamic_obstacle_speed: float = 0.55

    # Sampling margins
    boundary_margin: float = 0.35
    min_obj_separation: float = 0.12     # extra gap between circles at spawn

    # Sensors (pseudo-radar / lidar)
    num_rays: int = 16
    sensor_range: float = 4.5

    # Observation scaling
    normalize_goal_by_world: bool = True
    normalize_vel_by_max_speed: bool = True
    lidar_returns_normalized_distance: bool = True  # [0..1], 1 means no hit in range


class Nav2DEnv:
    """
    Minimal RL-style env interface:
      obs, info = env.reset(seed=0)
      obs, terminated, truncated, info = env.step(action)

    action: np.ndarray shape (2,), interpreted as desired velocity direction in [-1, 1].
    """
    def __init__(self, cfg: Optional[Nav2DConfig] = None, render_mode: Optional[str] = None):
        self.cfg = cfg or Nav2DConfig()
        self.render_mode = render_mode  # "human" or "rgb_array" or None

        self._rng = _seeded_rng(None)
        self._t = 0
        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._agent_vel = np.zeros(2, dtype=np.float32)
        self._goal_pos = np.zeros(2, dtype=np.float32)

        # Obstacles (arrays are float32)
        self._pillars_xy = np.zeros((0, 2), dtype=np.float32)
        self._pillars_r = np.zeros((0,), dtype=np.float32)

        self._haz_static_xy = np.zeros((0, 2), dtype=np.float32)
        self._haz_static_r = np.zeros((0,), dtype=np.float32)

        self._dyn_obs_xy = np.zeros((0, 2), dtype=np.float32)
        self._dyn_obs_v = np.zeros((0, 2), dtype=np.float32)
        self._dyn_obs_r = np.zeros((0,), dtype=np.float32)

        # Debug/visualization helpers
        self._last_pillar_hit_xy: Optional[np.ndarray] = None

        # For interactive matplotlib render
        self._render_ctx: Dict[str, Any] = {}

    # -----------------------------
    # Public API
    # -----------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._rng = _seeded_rng(seed)
        self._t = 0
        self._agent_vel = np.zeros(2, dtype=np.float32)

        # Sample agent + goal first, then obstacles; verify no initial overlaps.
        for _ in range(200):
            self._agent_pos = self._sample_point(margin=self.cfg.boundary_margin)
            self._goal_pos = self._sample_point_far_from(self._agent_pos, min_dist=self.cfg.world_size * 0.45)
            self._sample_obstacles()
            if self._spawn_is_clear():
                break
        else:
            # Fallback: project out of any overlaps to avoid spawning inside obstacles.
            self._resolve_pillar_penetrations()
            self._resolve_dynamic_obstacle_collisions(self._agent_pos.copy())

        obs = self._build_obs()
        info = self._build_info(
            prev_pos=self._agent_pos.copy(),
            collided_pillar=False,
            collided_dynamic=False,
            in_hazard=False,
            wall_hit=False,
        )
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
        cfg = self.cfg
        self._t += 1

        prev_pos = self._agent_pos.copy()
        self._last_pillar_hit_xy = None

        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        # Interpret action as desired velocity direction (scaled by max_speed).
        v_des = action * cfg.max_speed

        # Smooth velocity tracking: v <- (1-a)*v + a*v_des
        a = float(np.clip(cfg.dt / max(cfg.vel_time_constant, 1e-6), 0.0, 1.0))
        self._agent_vel = (1.0 - a) * self._agent_vel + a * v_des

        # Clamp speed
        speed = float(np.linalg.norm(self._agent_vel))
        if speed > cfg.max_speed:
            self._agent_vel = self._agent_vel / (speed + 1e-8) * cfg.max_speed

        # Integrate position
        self._agent_pos = self._agent_pos + self._agent_vel * cfg.dt

        # Move dynamic obstacles
        if self._dyn_obs_xy.shape[0] > 0:
            self._dyn_obs_xy = self._dyn_obs_xy + self._dyn_obs_v * cfg.dt
            self._bounce_circles_in_bounds(self._dyn_obs_xy, self._dyn_obs_v, self._dyn_obs_r)

        # Handle boundary bounce for agent
        wall_hit = self._bounce_agent_in_bounds()

        # Handle pillar/dynamic obstacle collisions (impassible).
        collided_pillar = self._resolve_pillar_collisions(prev_pos)
        collided_dynamic = self._resolve_dynamic_obstacle_collisions(prev_pos)
        # Re-check pillars in case dynamic obstacle resolution pushed us into one.
        collided_pillar = self._resolve_pillar_collisions(self._agent_pos.copy()) or collided_pillar
        # Final safety pass: ensure no residual pillar penetration.
        collided_pillar = self._resolve_pillar_penetrations() or collided_pillar

        # Hazard flag (virtual)
        in_hazard = self._check_in_hazard()

        # Termination / truncation
        dist_to_goal = float(np.linalg.norm(self._goal_pos - self._agent_pos))
        terminated = dist_to_goal <= cfg.goal_radius
        truncated = self._t >= cfg.max_steps

        obs = self._build_obs()
        info = self._build_info(prev_pos=prev_pos,
                                collided_pillar=collided_pillar,
                                collided_dynamic=collided_dynamic,
                                in_hazard=in_hazard,
                                wall_hit=wall_hit)

        # Optional render
        if self.render_mode == "human":
            self.render(info=info)

        return obs, terminated, truncated, info

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    # -----------------------------
    # Observation
    # -----------------------------
    def _build_obs(self) -> np.ndarray:
        cfg = self.cfg

        # Agent vel
        v = self._agent_vel.copy()
        if cfg.normalize_vel_by_max_speed:
            v = v / max(cfg.max_speed, 1e-6)

        # Goal relative vector
        g = (self._goal_pos - self._agent_pos).astype(np.float32)
        if cfg.normalize_goal_by_world:
            g = g / max(cfg.world_size, 1e-6)

        # Lidar-like radars
        rays = self._ray_directions(cfg.num_rays)  # (R,2)
        # Obstacles include pillars and dynamic obstacles; hazards are static-only.
        obs_xy = np.concatenate([self._pillars_xy, self._dyn_obs_xy], axis=0) if (self._pillars_xy.size or self._dyn_obs_xy.size) else np.zeros((0,2), dtype=np.float32)
        obs_r = np.concatenate([self._pillars_r, self._dyn_obs_r], axis=0) if (self._pillars_r.size or self._dyn_obs_r.size) else np.zeros((0,), dtype=np.float32)
        haz_xy = self._haz_static_xy
        haz_r = self._haz_static_r

        lidar_obstacles = self._lidar_to_circles(rays, obs_xy, obs_r)
        lidar_haz = self._lidar_to_circles(rays, haz_xy, haz_r)

        obs = np.concatenate([v, g, lidar_obstacles, lidar_haz], axis=0).astype(np.float32)
        return obs

    def obs_dim(self) -> int:
        return 2 + 2 + self.cfg.num_rays + self.cfg.num_rays

    def act_dim(self) -> int:
        return 2

    # -----------------------------
    # Info
    # -----------------------------
    def _build_info(self, prev_pos: np.ndarray, collided_pillar: bool, collided_dynamic: bool, in_hazard: bool, wall_hit: bool) -> Dict[str, Any]:
        # Debug distances to help verify collision flags vs visualization.
        nearest_pillar_dist = None
        nearest_pillar_clearance = None
        nearest_pillar_xy = None
        if self._pillars_xy.shape[0] > 0:
            dists = np.linalg.norm(self._pillars_xy - self._agent_pos[None, :], axis=1)
            idx = int(np.argmin(dists))
            nearest_pillar_dist = float(dists[idx])
            nearest_pillar_clearance = float(dists[idx] - (self._pillars_r[idx] + self.cfg.agent_radius))
            nearest_pillar_xy = self._pillars_xy[idx].copy()

        nearest_dyn_obs_dist = None
        nearest_dyn_obs_clearance = None
        nearest_dyn_obs_xy = None
        if self._dyn_obs_xy.shape[0] > 0:
            dists = np.linalg.norm(self._dyn_obs_xy - self._agent_pos[None, :], axis=1)
            idx = int(np.argmin(dists))
            nearest_dyn_obs_dist = float(dists[idx])
            nearest_dyn_obs_clearance = float(dists[idx] - (self._dyn_obs_r[idx] + self.cfg.agent_radius))
            nearest_dyn_obs_xy = self._dyn_obs_xy[idx].copy()

        info: Dict[str, Any] = {
            "t": self._t,
            "prev_agent_pos": prev_pos.astype(np.float32),
            "agent_pos": self._agent_pos.copy(),
            "agent_vel": self._agent_vel.copy(),
            "goal_pos": self._goal_pos.copy(),
            "collided_pillar": bool(collided_pillar),
            "collided_dynamic": bool(collided_dynamic),
            "in_hazard": bool(in_hazard),
            "wall_hit": bool(wall_hit),
            # Obstacles for wrapper/visualization
            "pillars_xy": self._pillars_xy.copy(),
            "pillars_r": self._pillars_r.copy(),
            "hazards_static_xy": self._haz_static_xy.copy(),
            "hazards_static_r": self._haz_static_r.copy(),
            "dynamic_obstacles_xy": self._dyn_obs_xy.copy(),
            "dynamic_obstacles_v": self._dyn_obs_v.copy(),
            "dynamic_obstacles_r": self._dyn_obs_r.copy(),
            "world_size": float(self.cfg.world_size),
            "agent_radius": float(self.cfg.agent_radius),
            "goal_radius": float(self.cfg.goal_radius),
            "max_steps": int(self.cfg.max_steps),
            "nearest_pillar_dist": nearest_pillar_dist,
            "nearest_pillar_clearance": nearest_pillar_clearance,
            "nearest_pillar_xy": nearest_pillar_xy,
            "last_pillar_hit_xy": self._last_pillar_hit_xy.copy() if self._last_pillar_hit_xy is not None else None,
            "nearest_dyn_obs_dist": nearest_dyn_obs_dist,
            "nearest_dyn_obs_clearance": nearest_dyn_obs_clearance,
            "nearest_dyn_obs_xy": nearest_dyn_obs_xy,
        }
        return info

    # -----------------------------
    # Sampling
    # -----------------------------
    def _spawn_is_clear(self) -> bool:
        cfg = self.cfg

        def clear_of(c_xy: np.ndarray, c_r: np.ndarray, radius: float) -> bool:
            if c_xy.size == 0:
                return True
            dists = np.linalg.norm(c_xy - self._agent_pos[None, :], axis=1)
            if np.any(dists < (c_r + radius + cfg.min_obj_separation)):
                return False
            d_goal = np.linalg.norm(c_xy - self._goal_pos[None, :], axis=1)
            if np.any(d_goal < (c_r + cfg.goal_radius + cfg.min_obj_separation)):
                return False
            return True

        if not clear_of(self._pillars_xy, self._pillars_r, cfg.agent_radius):
            return False
        if not clear_of(self._haz_static_xy, self._haz_static_r, cfg.agent_radius):
            return False
        if not clear_of(self._dyn_obs_xy, self._dyn_obs_r, cfg.agent_radius):
            return False
        return True

    def _sample_point(self, margin: float) -> np.ndarray:
        half = self.cfg.world_size / 2.0
        lo, hi = -half + margin, half - margin
        p = self._rng.uniform(low=lo, high=hi, size=(2,)).astype(np.float32)
        return p

    def _sample_point_far_from(self, ref: np.ndarray, min_dist: float) -> np.ndarray:
        for _ in range(10_000):
            p = self._sample_point(margin=self.cfg.boundary_margin)
            if float(np.linalg.norm(p - ref)) >= min_dist:
                return p
        # fallback
        return self._sample_point(margin=self.cfg.boundary_margin)

    def _sample_obstacles(self) -> None:
        cfg = self.cfg
        circles_xy: List[np.ndarray] = []
        circles_r: List[float] = []

        # Helper to test overlap
        def ok(new_xy: np.ndarray, new_r: float) -> bool:
            # keep away from agent and goal
            if np.linalg.norm(new_xy - self._agent_pos) < (new_r + cfg.agent_radius + cfg.min_obj_separation):
                return False
            if np.linalg.norm(new_xy - self._goal_pos) < (new_r + cfg.goal_radius + cfg.min_obj_separation):
                return False
            # keep away from other circles
            for xy, r in zip(circles_xy, circles_r):
                if np.linalg.norm(new_xy - xy) < (new_r + r + cfg.min_obj_separation):
                    return False
            return True

        # Pillars
        pillars_xy = []
        pillars_r = []
        for _ in range(cfg.num_pillars):
            for _try in range(5000):
                r = float(self._rng.uniform(cfg.pillar_radius_range[0], cfg.pillar_radius_range[1]))
                xy = self._sample_point(margin=cfg.boundary_margin + r)
                if ok(xy, r):
                    circles_xy.append(xy); circles_r.append(r)
                    pillars_xy.append(xy); pillars_r.append(r)
                    break

        # Static hazards
        haz_s_xy = []
        haz_s_r = []
        for _ in range(cfg.num_hazards_static):
            for _try in range(5000):
                r = float(cfg.hazard_static_radius)
                xy = self._sample_point(margin=cfg.boundary_margin + r)
                if ok(xy, r):
                    circles_xy.append(xy); circles_r.append(r)
                    haz_s_xy.append(xy); haz_s_r.append(r)
                    break

        # Dynamic obstacles
        dyn_obs_xy = []
        dyn_obs_r = []
        dyn_obs_v = []
        for _ in range(cfg.num_dynamic_obstacles):
            for _try in range(5000):
                r = float(cfg.dynamic_obstacle_radius)
                xy = self._sample_point(margin=cfg.boundary_margin + r)
                if ok(xy, r):
                    circles_xy.append(xy); circles_r.append(r)
                    dyn_obs_xy.append(xy); dyn_obs_r.append(r)
                    angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
                    v = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(cfg.dynamic_obstacle_speed)
                    dyn_obs_v.append(v)
                    break

        self._pillars_xy = np.array(pillars_xy, dtype=np.float32).reshape(-1, 2)
        self._pillars_r = np.array(pillars_r, dtype=np.float32).reshape(-1)

        self._haz_static_xy = np.array(haz_s_xy, dtype=np.float32).reshape(-1, 2)
        self._haz_static_r = np.array(haz_s_r, dtype=np.float32).reshape(-1)

        self._dyn_obs_xy = np.array(dyn_obs_xy, dtype=np.float32).reshape(-1, 2)
        self._dyn_obs_r = np.array(dyn_obs_r, dtype=np.float32).reshape(-1)
        self._dyn_obs_v = np.array(dyn_obs_v, dtype=np.float32).reshape(-1, 2)

    # -----------------------------
    # Physics helpers
    # -----------------------------
    def _bounce_agent_in_bounds(self) -> bool:
        cfg = self.cfg
        half = cfg.world_size / 2.0
        r = cfg.agent_radius
        wall_hit = False
        # x
        if self._agent_pos[0] < -half + r:
            self._agent_pos[0] = -half + r
            self._agent_vel[0] *= -0.3
            wall_hit = True
        elif self._agent_pos[0] > half - r:
            self._agent_pos[0] = half - r
            self._agent_vel[0] *= -0.3
            wall_hit = True
        # y
        if self._agent_pos[1] < -half + r:
            self._agent_pos[1] = -half + r
            self._agent_vel[1] *= -0.3
            wall_hit = True
        elif self._agent_pos[1] > half - r:
            self._agent_pos[1] = half - r
            self._agent_vel[1] *= -0.3
            wall_hit = True
        return wall_hit

    def _bounce_circles_in_bounds(self, xy: np.ndarray, v: np.ndarray, r: np.ndarray) -> None:
        cfg = self.cfg
        half = cfg.world_size / 2.0
        for i in range(xy.shape[0]):
            ri = float(r[i])
            # x
            if xy[i, 0] < -half + ri:
                xy[i, 0] = -half + ri
                v[i, 0] *= -1.0
            elif xy[i, 0] > half - ri:
                xy[i, 0] = half - ri
                v[i, 0] *= -1.0
            # y
            if xy[i, 1] < -half + ri:
                xy[i, 1] = -half + ri
                v[i, 1] *= -1.0
            elif xy[i, 1] > half - ri:
                xy[i, 1] = half - ri
                v[i, 1] *= -1.0

    def _resolve_pillar_collisions(self, prev_pos: np.ndarray) -> bool:
        cfg = self.cfg
        collided = False
        if self._pillars_xy.shape[0] == 0:
            return False
        seg = self._agent_pos - prev_pos
        seg_len2 = float(np.dot(seg, seg))
        for xy, r in zip(self._pillars_xy, self._pillars_r):
            if seg_len2 > 1e-10:
                t = float(np.dot(xy - prev_pos, seg) / seg_len2)
                t = float(np.clip(t, 0.0, 1.0))
                closest = prev_pos + t * seg
                delta = closest - xy
            else:
                delta = self._agent_pos - xy
            dist = float(np.linalg.norm(delta))
            min_dist = float(cfg.agent_radius + r)
            if dist < min_dist:
                collided = True
                # Project agent to the boundary of the pillar at the closest point.
                if dist < 1e-6:
                    if seg_len2 > 1e-10:
                        n = seg / (np.sqrt(seg_len2) + 1e-8)
                    else:
                        n = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    n = delta / dist
                self._agent_pos = xy + n * min_dist
                self._last_pillar_hit_xy = self._agent_pos.copy()
                # damp velocity normal component (simple inelastic response)
                vn = float(np.dot(self._agent_vel, n))
                if vn < 0:
                    self._agent_vel = self._agent_vel - (1.2 * vn) * n
                # additional damping
                self._agent_vel *= 0.85
        return collided

    def _resolve_pillar_penetrations(self, max_iters: int = 3) -> bool:
        """
        Resolve any residual overlaps with pillars using current position.
        This prevents visual/physics artifacts if projection lands inside another pillar.
        """
        cfg = self.cfg
        if self._pillars_xy.shape[0] == 0:
            return False
        collided = False
        for _ in range(max_iters):
            moved = False
            for xy, r in zip(self._pillars_xy, self._pillars_r):
                delta = self._agent_pos - xy
                dist = float(np.linalg.norm(delta))
                min_dist = float(cfg.agent_radius + r)
                if dist < min_dist:
                    collided = True
                    if dist < 1e-6:
                        n = np.array([1.0, 0.0], dtype=np.float32)
                    else:
                        n = delta / dist
                    self._agent_pos = xy + n * min_dist
                    self._last_pillar_hit_xy = self._agent_pos.copy()
                    moved = True
            if not moved:
                break
        return collided

    def _resolve_dynamic_obstacle_collisions(self, prev_pos: np.ndarray) -> bool:
        cfg = self.cfg
        collided = False
        if self._dyn_obs_xy.shape[0] == 0:
            return False
        seg = self._agent_pos - prev_pos
        seg_len2 = float(np.dot(seg, seg))
        for xy, r in zip(self._dyn_obs_xy, self._dyn_obs_r):
            if seg_len2 > 1e-10:
                t = float(np.dot(xy - prev_pos, seg) / seg_len2)
                t = float(np.clip(t, 0.0, 1.0))
                closest = prev_pos + t * seg
                delta = closest - xy
            else:
                delta = self._agent_pos - xy
            dist = float(np.linalg.norm(delta))
            min_dist = float(cfg.agent_radius + r)
            if dist < min_dist:
                collided = True
                if dist < 1e-6:
                    if seg_len2 > 1e-10:
                        n = seg / (np.sqrt(seg_len2) + 1e-8)
                    else:
                        n = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    n = delta / dist
                self._agent_pos = xy + n * min_dist
                vn = float(np.dot(self._agent_vel, n))
                if vn < 0:
                    self._agent_vel = self._agent_vel - (1.2 * vn) * n
                self._agent_vel *= 0.85
        return collided

    def _check_in_hazard(self) -> bool:
        cfg = self.cfg
        # static hazards
        for xy, r in zip(self._haz_static_xy, self._haz_static_r):
            if float(np.linalg.norm(self._agent_pos - xy)) < float(cfg.agent_radius + r):
                return True
        return False

    # -----------------------------
    # Lidar / ray casting
    # -----------------------------
    @staticmethod
    def _ray_directions(num_rays: int) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        rays = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        return rays

    def _lidar_to_circles(self, rays: np.ndarray, circles_xy: np.ndarray, circles_r: np.ndarray) -> np.ndarray:
        """
        For each ray direction, compute distance to the closest intersection with any circle
        (expanded by agent_radius). If none within sensor_range, return sensor_range.
        Output is normalized distance in [0..1] if cfg.lidar_returns_normalized_distance else raw distance.
        """
        cfg = self.cfg
        R = rays.shape[0]
        out = np.full((R,), cfg.sensor_range, dtype=np.float32)

        if circles_xy.shape[0] == 0:
            return out / cfg.sensor_range if cfg.lidar_returns_normalized_distance else out

        p = self._agent_pos.astype(np.float32)
        for i in range(R):
            d = rays[i]
            best = cfg.sensor_range
            for c, cr in zip(circles_xy, circles_r):
                # expand circle by agent radius
                r = float(cr + cfg.agent_radius)
                # Ray-circle intersection: ||p + t d - c||^2 = r^2, t>=0
                oc = p - c
                b = 2.0 * float(np.dot(d, oc))
                cc = float(np.dot(oc, oc) - r * r)
                disc = b * b - 4.0 * cc
                if disc < 0.0:
                    continue
                sqrt_disc = float(np.sqrt(disc))
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                t = None
                if t1 >= 0.0:
                    t = t1
                elif t2 >= 0.0:
                    t = t2
                if t is None:
                    continue
                if 0.0 <= t < best:
                    best = t
            out[i] = float(np.clip(best, 0.0, cfg.sensor_range))

        return out / cfg.sensor_range if cfg.lidar_returns_normalized_distance else out

    # -----------------------------
    # Rendering (matplotlib)
    # -----------------------------
    def render(self, info: Optional[Dict[str, Any]] = None, figsize: Tuple[int, int] = (6, 6)):
        """
        Rich 2D visualization using matplotlib.

        - Pillars: solid circles.
        - Static hazards: translucent circles.
        - Dynamic obstacles: translucent circles + velocity arrows.
        - Agent: circle + velocity arrow.
        - Goal: circle.
        - Trajectory: line.

        Works best with render_mode="human" (interactive).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        cfg = self.cfg
        half = cfg.world_size / 2.0

        if info is None:
            # build minimal info for rendering
            info = self._build_info(prev_pos=self._agent_pos.copy(),
                                    collided_pillar=False, collided_dynamic=False, in_hazard=False, wall_hit=False)

        if "fig" in self._render_ctx:
            fig = self._render_ctx["fig"]
            if not plt.fignum_exists(fig.number):
                # Window was closed; reset so we can recreate on next render.
                self._render_ctx.clear()

        if "fig" not in self._render_ctx:
            plt.ion()
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal")
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_title("Nav2DEnv (dynamics-only)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # World boundary
            ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], linewidth=1.5)

            # Patches
            agent_patch = Circle((0, 0), cfg.agent_radius, fill=True, alpha=0.9)
            goal_patch = Circle((0, 0), cfg.goal_radius, fill=True, alpha=0.6)

            ax.add_patch(goal_patch)
            ax.add_patch(agent_patch)

            # Obstacles
            pillar_patches = []
            pillar_clear_patches = []
            for xy, r in zip(info["pillars_xy"], info["pillars_r"]):
                p = Circle(tuple(xy), float(r), fill=True, alpha=0.8)
                ax.add_patch(p)
                pillar_patches.append(p)
                cp = Circle(tuple(xy), float(r + cfg.agent_radius), fill=False, linestyle="--", linewidth=1.0, alpha=0.4)
                ax.add_patch(cp)
                pillar_clear_patches.append(cp)

            haz_s_patches = []
            for xy, r in zip(info["hazards_static_xy"], info["hazards_static_r"]):
                p = Circle(tuple(xy), float(r), fill=True, alpha=0.5)
                ax.add_patch(p)
                haz_s_patches.append(p)

            dyn_obs_patches = []
            for xy, r in zip(info["dynamic_obstacles_xy"], info["dynamic_obstacles_r"]):
                p = Circle(tuple(xy), float(r), fill=True, alpha=0.8, facecolor="#003366")
                ax.add_patch(p)
                dyn_obs_patches.append(p)

            # Lines / arrows / text
            traj_line, = ax.plot([], [], linewidth=1.2)
            agent_quiv = ax.quiver([0], [0], [0], [0], angles='xy', scale_units='xy', scale=1.0, width=0.007)
            dh_xy0 = info["dynamic_obstacles_xy"]
            dh_v0 = info["dynamic_obstacles_v"]
            if dh_xy0.shape[0] > 0:
                dyn_quiv = ax.quiver(dh_xy0[:, 0], dh_xy0[:, 1], dh_v0[:, 0], dh_v0[:, 1], angles='xy', scale_units='xy', scale=1.0, width=0.004)
            else:
                dyn_quiv = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1.0, width=0.004)
            text = ax.text(-half + 0.2, half - 0.3, "", fontsize=9, va="top",
                           bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))
            goal_text = ax.text(0, 0, "GOAL", fontsize=8, ha="center", va="center")
            hit_marker, = ax.plot([], [], marker="x", linestyle="", markersize=8, color="red")

            self._render_ctx.update(
                fig=fig, ax=ax,
                agent_patch=agent_patch,
                goal_patch=goal_patch,
                pillar_patches=pillar_patches,
                pillar_clear_patches=pillar_clear_patches,
                haz_s_patches=haz_s_patches,
                dyn_obs_patches=dyn_obs_patches,
                traj_x=[], traj_y=[],
                traj_line=traj_line,
                agent_quiv=agent_quiv,
                dyn_quiv=dyn_quiv,
                text=text,
                goal_text=goal_text,
                hit_marker=hit_marker,
                last_t=None,
            )

        # Update
        ctx = self._render_ctx
        ctx["goal_patch"].center = tuple(info["goal_pos"])
        ctx["agent_patch"].center = tuple(info["agent_pos"])

        t = int(info["t"])
        if ctx["last_t"] is None or t <= 1 or t < int(ctx["last_t"]):
            ctx["traj_x"].clear()
            ctx["traj_y"].clear()
        ctx["last_t"] = t
        if info["collided_pillar"] or info.get("collided_dynamic", False):
            ctx["traj_x"].append(np.nan)
            ctx["traj_y"].append(np.nan)
        ctx["traj_x"].append(float(info["agent_pos"][0]))
        ctx["traj_y"].append(float(info["agent_pos"][1]))
        ctx["traj_line"].set_data(ctx["traj_x"], ctx["traj_y"])

        # Agent arrow
        ap = info["agent_pos"]
        av = info["agent_vel"]
        ctx["agent_quiv"].set_offsets([ap])
        ctx["agent_quiv"].set_UVC([av[0]], [av[1]])

        # Dynamic obstacle arrows
        dh_xy = info["dynamic_obstacles_xy"]
        dh_v = info["dynamic_obstacles_v"]
        n = min(dh_xy.shape[0], dh_v.shape[0])
        if n > 0:
            dh_xy = dh_xy[:n]
            dh_v = dh_v[:n]
            if ctx["dyn_quiv"].get_offsets().shape[0] != n:
                ctx["dyn_quiv"].remove()
                ctx["dyn_quiv"] = ctx["ax"].quiver(dh_xy[:, 0], dh_xy[:, 1], dh_v[:, 0], dh_v[:, 1], angles='xy', scale_units='xy', scale=1.0, width=0.004)
            else:
                ctx["dyn_quiv"].set_offsets(dh_xy)
                ctx["dyn_quiv"].set_UVC(dh_v[:, 0], dh_v[:, 1])
        else:
            ctx["dyn_quiv"].set_offsets(np.zeros((0,2)))
            ctx["dyn_quiv"].set_UVC([], [])

        # Update hazard patch centers
        for p, xy in zip(ctx["dyn_obs_patches"], info["dynamic_obstacles_xy"]):
            p.center = tuple(xy)

        # Status text
        dist = float(np.linalg.norm(info["goal_pos"] - info["agent_pos"]))
        reached_goal = dist <= float(self.cfg.goal_radius)
        npd = info.get("nearest_pillar_dist", None)
        ndod = info.get("nearest_dyn_obs_dist", None)
        npc = info.get("nearest_pillar_clearance", None)
        ndoc = info.get("nearest_dyn_obs_clearance", None)
        npd_s = f"{npd:.2f}" if npd is not None else "n/a"
        ndod_s = f"{ndod:.2f}" if ndod is not None else "n/a"
        npc_s = f"{npc:.2f}" if npc is not None else "n/a"
        ndoc_s = f"{ndoc:.2f}" if ndoc is not None else "n/a"
        ctx["text"].set_text(
            f"t={info['t']}  dist={dist:.2f}\n"
            f"hit_pillar={info['collided_pillar']}  hit_dynamic={info.get('collided_dynamic', False)}\n"
            f"pillar_dist={npd_s} clr={npc_s}  dyn_dist={ndod_s} clr={ndoc_s}\n"
            f"wall_hit={info['wall_hit']}  reached_goal={reached_goal}"
        )
        ctx["goal_text"].set_position(tuple(info["goal_pos"]))
        ctx["goal_text"].set_color("green" if reached_goal else "black")

        # Collision marker
        hit_xy = info.get("last_pillar_hit_xy", None)
        if hit_xy is not None:
            ctx["hit_marker"].set_data([hit_xy[0]], [hit_xy[1]])
        else:
            ctx["hit_marker"].set_data([], [])

        # Visual collision cue: slightly increase alpha and color on collision steps
        collided = bool(info["in_hazard"] or info["collided_pillar"] or info.get("collided_dynamic", False))
        ctx["agent_patch"].set_alpha(0.9 if not collided else 1.0)
        ctx["agent_patch"].set_color("red" if collided else "C1")

        ctx["fig"].canvas.draw()
        ctx["fig"].canvas.flush_events()

        if self.render_mode == "rgb_array":
            # Render to RGB array
            import matplotlib
            canvas = ctx["fig"].canvas
            canvas.draw()
            width, height = canvas.get_width_height()
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            return image
        return None
