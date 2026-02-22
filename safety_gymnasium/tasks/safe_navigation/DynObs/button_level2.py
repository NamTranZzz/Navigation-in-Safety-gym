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
"""DynObs level 2."""

from __future__ import annotations

import numpy as np

from safety_gymnasium.tasks.safe_navigation.DynObs.button_level1 import DynObsLevel1
from safety_gymnasium.utils.common_utils import ResamplingError


class DynObsLevel2(DynObsLevel1):
    """Offset-corridor DynObs setting with moving obstacles."""

    def __init__(self, config) -> None:
        config = dict(config)
        self.layout_min_goal_dist = float(config.pop('layout_min_goal_dist', 2.6))
        self.layout_max_goal_dist = float(config.pop('layout_max_goal_dist', 4.8))
        self.layout_sample_tries = int(config.pop('layout_sample_tries', 250))
        self.corridor_jitter = float(config.pop('corridor_jitter', 0.18))

        config.setdefault('placements_conf.extents', [-2.4, -2.4, 2.4, 2.4])
        config.setdefault('placements_conf.margin', 0.08)
        config.setdefault('agent.placements', [(-2.1, -1.8, -1.0, 1.8)])
        # Keep initial goal sampling easy; update_world() will snap goal to button0.
        config.setdefault(
            'Goal',
            {
                'size': 0.2,
                'alpha': 1.0,
                'keepout': 0.05,
                'locations': [(0.0, 2.1)],
            },
        )
        config.setdefault(
            'Buttons',
            {
                'num': 1,
                'is_constrained': False,
                'keepout': 0.2,
                'placements': [(1.0, -1.8, 2.1, 1.8)],
            },
        )
        config.setdefault(
            'Pillars',
            {
                'num': 5,
                'keepout': 0.45,
                'size': 0.4,
                'height': 0.25,
            },
        )
        config.setdefault(
            'GremlinVels',
            {
                'num': 6,
                'travel': 0.3,
                'keepout': 0.14,
                'omega_low': 1.0,
                'omega_high': 3.0,
            },
        )
        super().__init__(config=config)

    def update_world(self):
        """Resample an offset-corridor map with bounded start-goal distance."""
        layout = self._sample_offset_corridor_layout()
        self.world_info.layout.update(layout)
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        self.world.rebuild(self.world_info.world_config_dict, state=False)
        if self.viewer:
            self._update_viewer(self.model, self.data)

        self.build_goal_button()
        self._reposition_gremlins_near_goal()
        self.last_dist_goal = self.dist_goal()
        self.buttons.reset_timer()  # pylint: disable=no-member

    def _sample_offset_corridor_layout(self) -> dict[str, np.ndarray]:
        xmin, ymin, xmax, ymax = [float(x) for x in self.placements_conf.extents]
        margin = float(self.placements_conf.margin)
        agent_keepout = float(getattr(self.agent, 'keepout', 0.35))
        button_keepout = float(self.buttons.keepout)  # pylint: disable=no-member
        pillar_keepout = float(self.pillars.keepout)  # pylint: disable=no-member
        corridor_radius = float(self.pillars.size) + 0.22  # pylint: disable=no-member
        pillar_num = int(self.pillars.num)  # pylint: disable=no-member
        if pillar_num <= 0:
            raise ResamplingError('Pillars.num must be > 0 for DynObsLevel2')

        x_left_min = xmin + agent_keepout + 0.25
        x_left_max = min(-0.95, -0.4)
        x_right_min = max(0.95, 0.4)
        x_right_max = xmax - button_keepout - 0.25
        y_min = ymin + max(agent_keepout, button_keepout) + 0.15
        y_max = ymax - max(agent_keepout, button_keepout) - 0.15
        if x_left_min >= x_left_max or x_right_min >= x_right_max or y_min >= y_max:
            raise ResamplingError('Invalid extents for DynObs offset-corridor sampling')

        pillar_anchors = self._build_corridor_anchors(
            num=pillar_num,
            pillar_keepout=pillar_keepout,
            margin=margin,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

        for _ in range(self.layout_sample_tries):
            pillar_xy = self._sample_pillar_positions(
                anchors=pillar_anchors,
                pillar_keepout=pillar_keepout,
                margin=margin,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
            if not self._is_corridor_bands(pillar_xy):
                continue
            for occ_scale in (1.0, 0.85, 0.7):
                occ_keepout = max(0.08, pillar_keepout * occ_scale)
                occupied = [(p, occ_keepout) for p in pillar_xy]

                start_xy = self._sample_point(
                    x_left_min,
                    x_left_max,
                    y_min,
                    y_max,
                    agent_keepout,
                    occupied,
                    margin,
                )
                if start_xy is None:
                    continue

                occupied_with_start = occupied + [(start_xy, agent_keepout)]
                goal_xy = self._sample_point(
                    x_right_min,
                    x_right_max,
                    y_min,
                    y_max,
                    button_keepout,
                    occupied_with_start,
                    margin,
                )
                if goal_xy is None:
                    continue

                dist = float(np.linalg.norm(goal_xy - start_xy))
                if not (self.layout_min_goal_dist <= dist <= self.layout_max_goal_dist):
                    continue
                if not self._line_blocked(start_xy, goal_xy, pillar_xy, corridor_radius):
                    continue

                layout = {'agent': start_xy, 'button0': goal_xy}
                for i, pos in enumerate(pillar_xy[:pillar_num]):
                    layout[f'pillar{i}'] = pos
                return layout

        raise ResamplingError(
            'Failed to sample offset-corridor DynObs layout with valid start-goal distance',
        )

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
        """Build pillar anchors adaptively from geometry settings.

        This keeps the layout robust when users modify pillar count/size/keepout.
        """
        min_sep = max(0.16, 1.8 * pillar_keepout + 0.3 * margin)
        x_lo = xmin + pillar_keepout + 0.25
        x_hi = xmax - pillar_keepout - 0.25
        y_lo = ymin + pillar_keepout + 0.2
        y_hi = ymax - pillar_keepout - 0.2
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ResamplingError('Invalid extents for adaptive pillar anchors')

        # Explicit corridor: split pillars into lower and upper boundary bands.
        n_upper = num // 2
        n_lower = num - n_upper
        count = max(n_upper, n_lower)
        ts = np.linspace(0.08, 0.92, count)
        half_width = max(0.42, self.pillars.size + 0.55 * pillar_keepout + 0.16)  # pylint: disable=no-member
        half_width = min(half_width, 0.5 * (y_hi - y_lo) - 0.08)
        if half_width <= 0.12:
            raise ResamplingError('Insufficient Y span for corridor layout')

        anchors = []
        upper_idx = np.linspace(0, count - 1, n_upper).round().astype(int) if n_upper > 0 else []
        lower_idx = np.linspace(0, count - 1, n_lower).round().astype(int) if n_lower > 0 else []

        for idx in lower_idx:
            t = float(ts[idx])
            x = x_lo + (x_hi - x_lo) * t
            center_y = 0.25 * np.sin((1.05 * np.pi) * (t - 0.08))
            y = center_y - half_width
            anchors.append([x, y])
        for idx in upper_idx:
            t = float(ts[idx])
            x = x_lo + (x_hi - x_lo) * t
            center_y = 0.25 * np.sin((1.05 * np.pi) * (t - 0.08))
            y = center_y + half_width
            anchors.append([x, y])

        anchors = np.asarray(anchors, dtype=np.float64)
        anchors[:, 0] = np.clip(anchors[:, 0], x_lo, x_hi)
        anchors[:, 1] = np.clip(anchors[:, 1], y_lo, y_hi)

        # Final safety: if user sets extreme keepout/num, spread by X and alternate sides.
        ok = True
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                if np.linalg.norm(anchors[i] - anchors[j]) < min_sep:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return anchors

        fallback = []
        side = -1.0
        for t in np.linspace(0.08, 0.92, num):
            x = x_lo + (x_hi - x_lo) * float(t)
            center_y = 0.18 * np.sin(1.1 * np.pi * (float(t) - 0.08))
            y = center_y + side * min(half_width, 0.35)
            fallback.append([x, np.clip(y, y_lo, y_hi)])
            side *= -1.0
        return np.asarray(fallback, dtype=np.float64)

    def _sample_pillar_positions(
        self,
        anchors: np.ndarray,
        pillar_keepout: float,
        margin: float,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.ndarray:
        """Jitter anchors while preserving feasible spacing and map bounds."""
        min_sep = max(0.16, 2.0 * pillar_keepout + 0.25 * margin)
        for scale in (1.0, 0.6, 0.3, 0.0):
            jitter = self.random_generator.uniform(
                -self.corridor_jitter * scale,
                self.corridor_jitter * scale,
                size=anchors.shape,
            )
            cand = anchors + jitter
            cand[:, 0] = np.clip(cand[:, 0], xmin + pillar_keepout + 0.1, xmax - pillar_keepout - 0.1)
            cand[:, 1] = np.clip(cand[:, 1], ymin + pillar_keepout + 0.1, ymax - pillar_keepout - 0.1)

            ok = True
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    if np.linalg.norm(cand[i] - cand[j]) < min_sep:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return cand

        return anchors.copy()

    def _sample_point(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        keepout: float,
        occupied: list[tuple[np.ndarray, float]],
        margin: float,
    ) -> np.ndarray | None:
        for _ in range(120):
            cand = np.array(
                [
                    float(self.random_generator.uniform(xmin, xmax)),
                    float(self.random_generator.uniform(ymin, ymax)),
                ],
                dtype=np.float64,
            )
            valid = True
            for occ_xy, occ_keepout in occupied:
                if np.linalg.norm(cand - occ_xy) < (keepout + occ_keepout + margin):
                    valid = False
                    break
            if valid:
                return cand
        return None

    @staticmethod
    def _is_corridor_bands(pillar_xy: np.ndarray) -> bool:
        """Check that pillars still form upper/lower bands (corridor-like)."""
        if pillar_xy.shape[0] < 3:
            return True
        ys = pillar_xy[:, 1]
        median = float(np.median(ys))
        upper = ys[ys > median]
        lower = ys[ys <= median]
        if len(upper) == 0 or len(lower) == 0:
            return False
        return float(np.mean(upper) - np.mean(lower)) > 0.35

    @staticmethod
    def _line_blocked(
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        pillar_xy: np.ndarray,
        block_radius: float,
    ) -> bool:
        seg = goal_xy - start_xy
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-8:
            return False
        seg_dir = seg / seg_norm
        for center in pillar_xy:
            proj = float(np.dot(center - start_xy, seg_dir))
            proj = float(np.clip(proj, 0.0, seg_norm))
            closest = start_xy + proj * seg_dir
            if np.linalg.norm(center - closest) <= block_radius:
                return True
        return False
