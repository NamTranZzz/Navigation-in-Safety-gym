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
"""Velocity-driven gremlins."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Mocap


@dataclass
class GremlinVels(Mocap):  # pylint: disable=too-many-instance-attributes
    """Gremlins that follow smooth randomized velocity trajectories."""

    name: str = 'gremlin_vels'
    num: int = 0
    size: float = 0.1
    height_scale: float = 0.5
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.45
    travel: float = 0.35
    contact_cost: float = 1.0
    density: float = 0.001

    # Trajectory shape parameters.
    omega_low: float = 0.4
    omega_high: float = 1.4
    amp_low_scale: float = 0.35
    amp_high_scale: float = 1.0

    color: np.array = COLOR['gremlin']
    alpha: float = 1.0
    group: np.array = GROUP['gremlin']
    is_lidar_observed: bool = False
    is_constrained: bool = True
    is_meshed: bool = False
    mesh_name: str = 'gremlin'

    _anchor_xy: np.ndarray | None = field(default=None, init=False, repr=False)
    _amp_x: np.ndarray | None = field(default=None, init=False, repr=False)
    _amp_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _omg_x: np.ndarray | None = field(default=None, init=False, repr=False)
    _omg_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _phase_x: np.ndarray | None = field(default=None, init=False, repr=False)
    _phase_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _vel_xy: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_time: float = field(default=-1.0, init=False, repr=False)

    def get_config(self, xy_pos, rot):
        """Build object and mocap configs."""
        return {'obj': self.get_obj(xy_pos, rot), 'mocap': self.get_mocap(xy_pos, rot)}

    def get_obj(self, xy_pos, rot):
        """Build free-geom body config."""
        half_h = self.size * self.height_scale
        body_z = half_h
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, body_z],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([self.size, self.size, half_h]),
                    'type': 'box',
                    'density': self.density,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'rgba': np.array([1, 1, 1, 1]),
                    'euler': [np.pi / 2, 0, 0],
                },
            )
            body['pos'][2] = 0.0
        return body

    def get_mocap(self, xy_pos, rot):
        """Build mocap body config."""
        half_h = self.size * self.height_scale
        body_z = half_h
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, body_z],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([self.size, self.size, half_h]),
                    'type': 'box',
                    'group': self.group,
                    # Keep mocap body fully invisible in human rendering.
                    'rgba': self.color * np.array([1, 1, 1, 0.0]),
                },
            ],
        }
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'rgba': np.array([1, 1, 1, 0]),
                    'euler': [np.pi / 2, 0, 0],
                },
            )
            body['pos'][2] = 0.0
        return body

    def _init_trajectory(self) -> None:
        """Initialize smooth randomized trajectory parameters at each reset."""
        rng = self.random_generator if self.random_generator is not None else np.random

        self._anchor_xy = np.zeros((self.num, 2), dtype=np.float64)
        for i in range(self.num):
            self._anchor_xy[i] = self.engine.data.body(f'gremlin_vel{i}obj').xpos[:2].copy()

        amp_low = float(self.amp_low_scale * self.travel)
        amp_high = float(self.amp_high_scale * self.travel)
        self._amp_x = rng.uniform(amp_low, amp_high, size=(self.num,))
        self._amp_y = rng.uniform(amp_low, amp_high, size=(self.num,))
        self._omg_x = rng.uniform(self.omega_low, self.omega_high, size=(self.num,))
        self._omg_y = rng.uniform(self.omega_low, self.omega_high, size=(self.num,))
        self._phase_x = rng.uniform(0.0, 2.0 * np.pi, size=(self.num,))
        self._phase_y = rng.uniform(0.0, 2.0 * np.pi, size=(self.num,))
        self._vel_xy = np.zeros((self.num, 2), dtype=np.float64)

    def cal_cost(self):
        """Contact-only cost for colliding with gremlin_vel objects."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_gremlin_vels'] = 0.0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('gremlin_vel') for n in geom_names) and any(
                n in self.agent.body_info.geom_names for n in geom_names
            ):
                cost['cost_gremlin_vels'] += self.contact_cost
        return cost

    def move(self):
        """Move each mocap body with a smooth non-circular velocity profile."""
        t = float(self.engine.data.time)
        if self._anchor_xy is None or t < self._last_time:
            self._init_trajectory()
        self._last_time = t

        for i in range(self.num):
            phase_x = self._omg_x[i] * t + self._phase_x[i]
            phase_y = self._omg_y[i] * t + self._phase_y[i]
            x = self._anchor_xy[i, 0] + self._amp_x[i] * np.sin(phase_x)
            y = self._anchor_xy[i, 1] + self._amp_y[i] * np.sin(phase_y)
            vx = self._amp_x[i] * self._omg_x[i] * np.cos(phase_x)
            vy = self._amp_y[i] * self._omg_y[i] * np.cos(phase_y)

            self._vel_xy[i, 0] = vx
            self._vel_xy[i, 1] = vy
            self.set_mocap_pos(
                f'gremlin_vel{i}mocap',
                np.array([x, y, self.size * self.height_scale], dtype=np.float64),
            )

    @property
    def pos(self):
        """Return world-frame positions."""
        return [self.engine.data.body(f'gremlin_vel{i}obj').xpos.copy() for i in range(self.num)]

    @property
    def vel(self):
        """Return world-frame XY velocities."""
        if self._vel_xy is None:
            return np.zeros((self.num, 2), dtype=np.float64)
        return self._vel_xy.copy()
