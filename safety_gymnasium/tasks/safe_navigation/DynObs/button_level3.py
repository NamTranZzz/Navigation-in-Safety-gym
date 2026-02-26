# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DynObs level 3."""

from __future__ import annotations

import re

import numpy as np

from safety_gymnasium.tasks.safe_navigation.DynObs.button_level2 import DynObsLevel2
from safety_gymnasium.utils.common_utils import ResamplingError


class DynObsLevel3(DynObsLevel2):
    """Corridor-style DynObs with smaller pillars and gremlins."""

    def __init__(self, config) -> None:
        config = dict(config)
        self.safe_radius = float(config.pop('safe_radius', 0.07))
        self.agent_body_radius = float(config.pop('agent_body_radius', 0.1))
        self.gremlin_spawn_tube_radius = float(config.pop('gremlin_spawn_tube_radius', 0.035))
        self.gremlin_spawn_t_min = float(config.pop('gremlin_spawn_t_min', 0.25))
        self.gremlin_spawn_t_max = float(config.pop('gremlin_spawn_t_max', 0.75))
        self.gremlin_spawn_center_bias = float(config.pop('gremlin_spawn_center_bias', 2.4))

        # Keep the Level2 corridor sampler, but with milder geometry sizes.
        config.setdefault('layout_min_goal_dist', 2.8)
        config.setdefault('layout_max_goal_dist', 5.2)
        config.setdefault('layout_sample_tries', 320)
        config.setdefault('corridor_jitter', 0.12)

        config.setdefault('placements_conf.extents', [-2.6, -2.3, 2.6, 2.3])
        config.setdefault('placements_conf.margin', 0.08)
        config.setdefault('agent.placements', [(-2.3, -2.0, -1.2, 2.0)])
        config.setdefault(
            'Goal',
            {
                'size': 0.1,
                'alpha': 1.0,
                'keepout': 0.05,
                'locations': [(0.0, 2.0)],
            },
        )
        config.setdefault(
            'Buttons',
            {
                'num': 1,
                'is_constrained': False,
                'keepout': 0.2,
                'placements': [(1.2, -2.0, 2.3, 2.0)],
            },
        )

        # Corridor1-inspired count, but with smaller obstacle radii/keepouts.
        config.setdefault(
            'Pillars',
            {
                'num': 4,
                'keepout': 0.28,
                'size': 0.2,
                'height': 0.15,
            },
        )
        config.setdefault(
            'GremlinVels',
            {
                'num': 6,
                'size': 0.05,
                'travel': 0.18,
                'keepout': 0.28,
                'omega_low': 0.8,
                'omega_high': 2.6,
            },
        )

        super().__init__(config=config)

    def calculate_cost(self) -> dict:
        """Binary cost with body-aware boundary.

        cost=1 if dist(agent_center, obstacle_center) is below:
          obstacle_body_radius + safe_radius + agent_radius
        """
        # Keep individual obstacle costs for debugging/inspection.
        cost = super().calculate_cost()

        agent_xy = np.asarray(self.agent.pos[:2], dtype=np.float64)
        agent_radius = float(self.agent_body_radius)
        violated = False

        pillar_threshold = (  # pylint: disable=no-member
            float(self.pillars.size) + float(self.safe_radius) + agent_radius
        )
        for p in self.pillars.pos:  # pylint: disable=no-member
            if np.linalg.norm(agent_xy - np.asarray(p[:2], dtype=np.float64)) < pillar_threshold:
                violated = True
                break

        gremlin_threshold = (  # pylint: disable=no-member
            float(self.gremlin_vels.size) + float(self.safe_radius) + agent_radius
        )
        if not violated:
            for g in self.gremlin_vels.pos:  # pylint: disable=no-member
                if np.linalg.norm(agent_xy - np.asarray(g[:2], dtype=np.float64)) < gremlin_threshold:
                    violated = True
                    break

        cost['cost_safe_radius'] = 1.0 if violated else 0.0
        # Override aggregate cost to match requested indicator behavior.
        cost['cost_sum'] = float(cost['cost_safe_radius'])
        return cost

    def _build_corridor_anchors(
        self,
        num: int,
        pillar_keepout: float,
        margin: float,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.ndarray:
        """Build randomized alternating anchors to induce S-like routes."""
        if num <= 0:
            raise ResamplingError('Pillars.num must be > 0 for DynObsLevel3')

        x_lo = xmin + pillar_keepout + 0.28
        x_hi = xmax - pillar_keepout - 0.28
        y_lo = ymin + pillar_keepout + 0.2
        y_hi = ymax - pillar_keepout - 0.2
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ResamplingError('Invalid extents for DynObsLevel3 anchors')

        y_mid = 0.5 * (y_lo + y_hi)
        y_half = 0.5 * (y_hi - y_lo)
        lane_amp = float(self.random_generator.uniform(0.45 * y_half, 0.82 * y_half))
        lane_amp = max(0.22, lane_amp)

        ts = np.linspace(0.1, 0.9, num)
        x_jitter = self.random_generator.uniform(-0.08, 0.08, size=(num,))
        side = 1.0 if float(self.random_generator.uniform()) < 0.5 else -1.0

        anchors = []
        for i, t in enumerate(ts):
            sign = side if (i % 2 == 0) else -side
            x = x_lo + (x_hi - x_lo) * float(np.clip(t + x_jitter[i], 0.05, 0.95))
            y = y_mid + sign * lane_amp + float(self.random_generator.uniform(-0.16, 0.16))
            anchors.append([x, np.clip(y, y_lo, y_hi)])

        anchors = np.asarray(anchors, dtype=np.float64)
        min_sep = max(0.16, 2.0 * pillar_keepout + 0.28 * margin)
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                if np.linalg.norm(anchors[i] - anchors[j]) < min_sep:
                    return super()._build_corridor_anchors(
                        num=num,
                        pillar_keepout=pillar_keepout,
                        margin=margin,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
        return anchors

    def _reposition_gremlins_near_goal(self) -> None:
        """Place gremlin anchors inside a sampled S-corridor with robust fallbacks."""
        xmin, ymin, xmax, ymax = [float(x) for x in self.placements_conf.extents]
        margin = float(self.placements_conf.margin)
        g_keepout = float(self.gremlin_vels.keepout)  # pylint: disable=no-member
        g_num = int(self.gremlin_vels.num)  # pylint: disable=no-member
        if g_num <= 0:
            return

        corridor_pts = self._corridor_polyline_from_layout()
        corridor_radius = float(self.pillars.size) + 0.24  # pylint: disable=no-member
        move_amp = float(getattr(self.gremlin_vels, 'amp_high_scale', 1.0)) * float(self.gremlin_vels.travel)  # pylint: disable=no-member
        base_radius = max(0.01, min(self.gremlin_spawn_tube_radius, corridor_radius - move_amp - g_keepout - 0.5 * margin))

        occupied_xy = []
        occupied_keepout = []
        occupied_xy.append(np.asarray(self.agent.pos[:2], dtype=np.float64))
        occupied_keepout.append(float(getattr(self.agent, 'keepout', 0.35)))
        occupied_xy.append(np.asarray(self.goal.pos[:2], dtype=np.float64))  # pylint: disable=no-member
        occupied_keepout.append(float(self.buttons.keepout))  # pylint: disable=no-member
        for pos in self.pillars.pos:  # pylint: disable=no-member
            occupied_xy.append(np.asarray(pos[:2], dtype=np.float64))
            occupied_keepout.append(float(self.pillars.keepout))  # pylint: disable=no-member

        def _apply_placements(placements: list[np.ndarray]) -> None:
            for i, xy in enumerate(placements):
                self.world_info.layout[f'gremlin_vel{i}'] = np.asarray(xy, dtype=np.float64)
                obj_key = f'gremlin_vel{i}obj'
                mocap_key = f'gremlin_vel{i}mocap'
                self.world_info.world_config_dict['free_geoms'][obj_key]['pos'][:2] = xy
                self.world_info.world_config_dict['mocaps'][mocap_key]['pos'][:2] = xy

            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)

        old_t_min = self.gremlin_spawn_t_min
        old_t_max = self.gremlin_spawn_t_max
        # Progressively relax sampling constraints to avoid reset-time failures.
        modes = (
            (base_radius, 0.15, 0.85, 1.00, 700),
            (min(corridor_radius * 0.55, max(base_radius, base_radius * 1.6)), 0.08, 0.92, 0.85, 700),
            (min(corridor_radius * 0.80, max(base_radius, base_radius * 2.4)), 0.00, 1.00, 0.70, 900),
        )
        try:
            for tube_radius, t_min, t_max, clear_scale, tries_per_obj in modes:
                self.gremlin_spawn_t_min = float(t_min)
                self.gremlin_spawn_t_max = float(t_max)

                placements: list[np.ndarray] = []
                occ_xy = [x.copy() for x in occupied_xy]
                occ_keep = [float(k) for k in occupied_keepout]
                min_clearance = max(0.0, clear_scale * margin)
                feasible = True

                for _ in range(g_num):
                    found = False
                    for _ in range(int(tries_per_obj)):
                        cand = self._sample_point_in_corridor(corridor_pts, float(tube_radius))
                        if cand is None:
                            continue
                        if not (xmin + g_keepout <= cand[0] <= xmax - g_keepout):
                            continue
                        if not (ymin + g_keepout <= cand[1] <= ymax - g_keepout):
                            continue

                        valid = True
                        for occ_p, occ_k in zip(occ_xy, occ_keep):
                            if np.linalg.norm(cand - occ_p) < (g_keepout + occ_k + min_clearance):
                                valid = False
                                break
                        if not valid:
                            continue
                        placements.append(cand)
                        occ_xy.append(cand)
                        occ_keep.append(g_keepout)
                        found = True
                        break

                    if not found:
                        feasible = False
                        break

                if feasible and len(placements) == g_num:
                    _apply_placements(placements)
                    return
        finally:
            self.gremlin_spawn_t_min = old_t_min
            self.gremlin_spawn_t_max = old_t_max

        # Last resort (still corridor-only): build a broad candidate pool, then
        # greedily pack gremlins with progressively relaxed spacing. This avoids
        # random reset crashes while preserving corridor blocking behavior.
        placements: list[np.ndarray] = []
        agent_xy = np.asarray(self.agent.pos[:2], dtype=np.float64)
        agent_clear = max(
            float(g_keepout + getattr(self.agent, 'keepout', 0.35) + margin),
            float(self.gremlin_vels.size + self.safe_radius + 0.02),  # pylint: disable=no-member
        )

        candidate_pool: list[np.ndarray] = []
        for t_min, t_max, rad, n_try in (
            (0.25, 0.75, max(0.0, 0.2 * base_radius), 600),
            (0.15, 0.85, max(0.01, 0.6 * base_radius), 600),
            (0.05, 0.95, max(0.02, max(base_radius, 0.04)), 800),
        ):
            self.gremlin_spawn_t_min = float(t_min)
            self.gremlin_spawn_t_max = float(t_max)
            for _ in range(int(n_try)):
                c_try = self._sample_point_in_corridor(corridor_pts, float(rad))
                if c_try is None:
                    continue
                c_try[0] = float(np.clip(c_try[0], xmin + g_keepout, xmax - g_keepout))
                c_try[1] = float(np.clip(c_try[1], ymin + g_keepout, ymax - g_keepout))
                if np.linalg.norm(c_try - agent_xy) < agent_clear:
                    continue
                candidate_pool.append(c_try)

        # If stochastic sampling missed everything, collect deterministic points.
        if len(candidate_pool) == 0:
            self.gremlin_spawn_t_min = 0.05
            self.gremlin_spawn_t_max = 0.95
            for t in np.linspace(0.05, 0.95, max(20, 6 * g_num)):
                self.gremlin_spawn_t_min = float(t)
                self.gremlin_spawn_t_max = float(t)
                c_try = self._sample_point_in_corridor(corridor_pts, 0.0)
                if c_try is None:
                    continue
                c_try[0] = float(np.clip(c_try[0], xmin + g_keepout, xmax - g_keepout))
                c_try[1] = float(np.clip(c_try[1], ymin + g_keepout, ymax - g_keepout))
                if np.linalg.norm(c_try - agent_xy) < agent_clear:
                    continue
                candidate_pool.append(c_try)

        # Hard failure only if no corridor point is agent-safe.
        if len(candidate_pool) == 0:
            raise ResamplingError('Failed to place gremlin_vels in corridor (no safe candidates)')

        self.random_generator.random_generator.shuffle(candidate_pool)

        # Greedy packing with relaxed spacing.
        for clear_scale in (1.0, 0.7, 0.4, 0.0):
            placements = []
            occ_xy = [x.copy() for x in occupied_xy]
            occ_keep = [float(k) for k in occupied_keepout]
            min_clearance = float(clear_scale * margin)
            for cand in candidate_pool:
                valid = True
                for occ_p, occ_k in zip(occ_xy, occ_keep):
                    if np.linalg.norm(cand - occ_p) < (g_keepout + occ_k + min_clearance):
                        valid = False
                        break
                if not valid:
                    continue
                placements.append(cand.copy())
                occ_xy.append(cand.copy())
                occ_keep.append(g_keepout)
                if len(placements) >= g_num:
                    break
            if len(placements) >= g_num:
                break

        # Fill remaining slots by reusing far candidates (still agent-safe).
        if len(placements) < g_num:
            cand_arr = np.asarray(candidate_pool, dtype=np.float64)
            d_agent = np.linalg.norm(cand_arr - agent_xy.reshape(1, 2), axis=1)
            order = np.argsort(-d_agent)  # farthest first
            for idx in order.tolist():
                cand = cand_arr[idx].copy()
                placements.append(cand)
                if len(placements) >= g_num:
                    break

        self.gremlin_spawn_t_min = old_t_min
        self.gremlin_spawn_t_max = old_t_max
        _apply_placements(placements[:g_num])

    def _corridor_polyline_from_layout(self) -> np.ndarray:
        """Construct an S-like corridor polyline from sampled layout."""
        start_xy = np.asarray(self.world_info.layout['agent'][:2], dtype=np.float64)
        goal_xy = np.asarray(self.world_info.layout['button0'][:2], dtype=np.float64)

        pillar_items = []
        for key, val in self.world_info.layout.items():
            if re.fullmatch(r'pillar\d+', key):
                pillar_items.append((key, np.asarray(val[:2], dtype=np.float64)))
        pillar_items.sort(key=lambda kv: float(kv[1][0]))
        pillar_xy = [p for _, p in pillar_items]

        xmin, ymin, xmax, ymax = [float(x) for x in self.placements_conf.extents]
        y_mid = 0.5 * (ymin + ymax)
        y_span = max(0.2, ymax - ymin)
        lane = min(0.42 * y_span, 0.85)
        y_min_safe = ymin + 0.25
        y_max_safe = ymax - 0.25

        pts = [start_xy]
        for p in pillar_xy:
            sign = -1.0 if float(p[1]) >= y_mid else 1.0
            y_target = y_mid + sign * lane + float(self.random_generator.uniform(-0.1, 0.1))
            y_target = float(np.clip(y_target, y_min_safe, y_max_safe))
            pts.append(np.asarray([float(np.clip(p[0], xmin + 0.2, xmax - 0.2)), y_target], dtype=np.float64))
        pts.append(goal_xy)
        return np.asarray(pts, dtype=np.float64)

    def _sample_point_in_corridor(self, corridor_pts: np.ndarray, radius: float) -> np.ndarray | None:
        """Sample a point from a tube around corridor polyline."""
        if corridor_pts.shape[0] < 2:
            return None

        segs = corridor_pts[1:] - corridor_pts[:-1]
        lens = np.linalg.norm(segs, axis=1)
        total = float(np.sum(lens))
        if total < 1e-8:
            return None

        t_min = float(np.clip(self.gremlin_spawn_t_min, 0.0, 1.0))
        t_max = float(np.clip(self.gremlin_spawn_t_max, 0.0, 1.0))
        if t_max < t_min:
            t_min, t_max = t_max, t_min
        if self.gremlin_spawn_center_bias > 1.0:
            # RandomGenerator does not expose beta(); use mean of uniforms to
            # create a symmetric center-biased sample in [0, 1].
            n = max(2, int(round(self.gremlin_spawn_center_bias)))
            u = float(np.mean(self.random_generator.uniform(0.0, 1.0, size=(n,))))
            s = float(t_min + (t_max - t_min) * u) * total
        else:
            s = float(self.random_generator.uniform(t_min, t_max)) * total

        rem = s
        seg_idx = 0
        for i, ll in enumerate(lens):
            if rem <= float(ll) or i == len(lens) - 1:
                seg_idx = i
                break
            rem -= float(ll)
        a = corridor_pts[seg_idx]
        b = corridor_pts[seg_idx + 1]
        seg_len = max(float(lens[seg_idx]), 1e-8)
        t = float(np.clip(rem / seg_len, 0.0, 1.0))
        base = a + t * (b - a)

        dir_vec = b - a
        seg_len = float(np.linalg.norm(dir_vec))
        if seg_len < 1e-8:
            return None
        dir_vec = dir_vec / seg_len
        normal = np.asarray([-dir_vec[1], dir_vec[0]], dtype=np.float64)
        offset = float(self.random_generator.uniform(-radius, radius))
        return base + offset * normal
