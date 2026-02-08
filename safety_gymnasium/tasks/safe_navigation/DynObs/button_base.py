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
"""DynObs task base."""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import ResamplingError


class DynObsBase(BaseTask):
    """Button task with pillar lidar and dynamic-obstacle state observation."""

    def __init__(self, config) -> None:
        assert 'Goal' in config, '`config` must have the field `Goal`'
        assert 'Buttons' in config, '`config` must have the field `Buttons`'
        assert 'Pillars' in config, '`config` must have the field `Pillars`'
        assert 'GremlinVels' in config, '`config` must have the field `GremlinVels`'
        super().__init__(config=config)

        self.last_dist_goal = None
        self._ep_step = 0
        # Terminate immediately when goal button is reached.
        self.mechanism_conf.continue_goal = False

    def build_observation_space(self):
        """Build observation space with explicit DynObs features."""
        obs_space_dict = OrderedDict()
        obs_space_dict.update(self.agent.build_sensor_observation_space())

        obs_space_dict['buttons_lidar'] = gymnasium.spaces.Box(
            0.0,
            1.0,
            (self.lidar_conf.num_bins,),
            dtype=np.float64,
        )
        obs_space_dict['pillars_lidar'] = gymnasium.spaces.Box(
            0.0,
            1.0,
            (self.lidar_conf.num_bins,),
            dtype=np.float64,
        )
        rel_dim = int(self.gremlin_vels.num) * 2  # pylint: disable=no-member
        obs_space_dict['gremlin_vels_rel_pos'] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            (rel_dim,),
            dtype=np.float64,
        )
        obs_space_dict['gremlin_vels_vel'] = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            (rel_dim,),
            dtype=np.float64,
        )

        if self.observe_vision:
            width, height = self.vision_env_conf.vision_size
            rows, cols = height, width
            self.vision_env_conf.vision_size = (rows, cols)
            obs_space_dict['vision'] = gymnasium.spaces.Box(
                0,
                255,
                (*self.vision_env_conf.vision_size, 3),
                dtype=np.uint8,
            )

        self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)
        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(self.obs_info.obs_space_dict)
        else:
            self.observation_space = self.obs_info.obs_space_dict

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.buttons.reward_distance  # pylint: disable=no-member
        self.last_dist_goal = dist_goal
        if self.goal_achieved:
            reward += self.buttons.reward_goal  # pylint: disable=no-member
            # Optional terminal bonus that scales with remaining episode budget.
            time_remain = max(0, int(self.num_steps) - int(self._ep_step + 1))
            coef = float(self.reward_conf.reward_goal_time_coef)
            if coef == 0.0:
                coef = float(self.reward_conf.goal_time_coef)
            reward += coef * float(time_remain)
        return reward

    def specific_reset(self):
        """Reset the buttons timer."""
        self.buttons.timer = 0  # pylint: disable=no-member
        self._ep_step = 0

    def specific_step(self):
        """Clock the buttons timer."""
        self.buttons.timer_tick()  # pylint: disable=no-member
        self._ep_step += 1

    def update_world(self):
        """Pick a new goal button and reset distance buffer."""
        assert self.buttons.num > 0, 'Must have at least one button.'  # pylint: disable=no-member
        self.build_goal_button()
        self._reposition_gremlins_near_goal()
        self.last_dist_goal = self.dist_goal()
        self.buttons.reset_timer()  # pylint: disable=no-member

    def _reposition_gremlins_near_goal(self) -> None:
        """Resample gremlin anchors around the current goal position."""
        goal_xy = np.asarray(self.goal.pos[:2], dtype=np.float64)  # pylint: disable=no-member
        xmin, ymin, xmax, ymax = [float(x) for x in self.placements_conf.extents]
        margin = float(self.placements_conf.margin)
        g_keepout = float(self.gremlin_vels.keepout)  # pylint: disable=no-member

        occupied_xy = []
        occupied_keepout = []

        # Keep gremlins away from agent and static obstacles.
        occupied_xy.append(np.asarray(self.agent.pos[:2], dtype=np.float64))
        occupied_keepout.append(float(getattr(self.agent, 'keepout', 0.35)))
        for pos in self.pillars.pos:  # pylint: disable=no-member
            occupied_xy.append(np.asarray(pos[:2], dtype=np.float64))
            occupied_keepout.append(float(self.pillars.keepout))  # pylint: disable=no-member

        placements = []
        for _ in range(int(self.gremlin_vels.num)):  # pylint: disable=no-member
            found = False
            # Try a near-goal ring first, then widen, then full-map random as fallback.
            for mode in ('near', 'mid', 'global'):
                tries = 250 if mode != 'global' else 500
                for _ in range(tries):
                    if mode == 'global':
                        cand = np.array(
                            [
                                float(self.random_generator.uniform(xmin + g_keepout, xmax - g_keepout)),
                                float(self.random_generator.uniform(ymin + g_keepout, ymax - g_keepout)),
                            ],
                            dtype=np.float64,
                        )
                    else:
                        theta = float(self.random_generator.uniform(0.0, 2.0 * np.pi))
                        r_low, r_high = (0.45, 1.25) if mode == 'near' else (0.45, 2.0)
                        r = float(self.random_generator.uniform(r_low, r_high))
                        cand = goal_xy + r * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

                        if not (xmin + g_keepout <= cand[0] <= xmax - g_keepout):
                            continue
                        if not (ymin + g_keepout <= cand[1] <= ymax - g_keepout):
                            continue

                    ok = True
                    for occ_xy, occ_keepout in zip(occupied_xy, occupied_keepout):
                        if np.linalg.norm(cand - occ_xy) < (g_keepout + occ_keepout + margin):
                            ok = False
                            break
                    if not ok:
                        continue

                    placements.append(cand)
                    occupied_xy.append(cand)
                    occupied_keepout.append(g_keepout)
                    found = True
                    break
                if found:
                    break
            if not found:
                raise ResamplingError('Failed to place gremlin_vels around goal')

        # Persist sampled gremlin positions in layout/world config, then rebuild world state.
        for i, xy in enumerate(placements):
            self.world_info.layout[f'gremlin_vel{i}'] = np.asarray(xy, dtype=np.float64)
            obj_key = f'gremlin_vel{i}obj'
            mocap_key = f'gremlin_vel{i}mocap'
            self.world_info.world_config_dict['free_geoms'][obj_key]['pos'][:2] = xy
            self.world_info.world_config_dict['mocaps'][mocap_key]['pos'][:2] = xy

        self.world.rebuild(self.world_info.world_config_dict, state=False)
        if self.viewer:
            self._update_viewer(self.model, self.data)

    def build_goal_button(self):
        """Pick a new goal button."""
        self.buttons.goal_button = self.random_generator.choice(self.buttons.num)  # pylint: disable=no-member
        new_goal_pos = self.buttons.pos[self.buttons.goal_button]  # pylint: disable=no-member
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = new_goal_pos[:2]
        self._set_goal(new_goal_pos[:2])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def obs(self):
        """Return the observation of our agent."""
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member
        obs = {}
        obs.update(self.agent.obs_sensor())

        obs['buttons_lidar'] = self._obs_lidar(self.buttons.pos, self.buttons.group)  # pylint: disable=no-member
        obs['pillars_lidar'] = self._obs_lidar(self.pillars.pos, self.pillars.group)  # pylint: disable=no-member

        if self.buttons.timer != 0:  # pylint: disable=no-member
            obs['buttons_lidar'] = np.zeros(self.lidar_conf.num_bins)

        rel_pos = []
        vel_ego = []
        gremlin_vels = self.gremlin_vels  # pylint: disable=no-member
        for i, pos in enumerate(gremlin_vels.pos):
            rel = self._ego_xy(np.asarray(pos[:2], dtype=np.float64))
            vel = np.asarray(gremlin_vels.vel[i], dtype=np.float64)
            vel3 = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
            vel_local = np.matmul(vel3, self.agent.mat)[:2]
            rel_pos.extend(rel.tolist())
            vel_ego.extend(vel_local.tolist())
        obs['gremlin_vels_rel_pos'] = np.asarray(rel_pos, dtype=np.float64)
        obs['gremlin_vels_vel'] = np.asarray(vel_ego, dtype=np.float64)

        if self.observe_vision:
            obs['vision'] = self._obs_vision()

        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'

        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        """Whether the goal button is pressed by the agent."""
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if any(n == f'button{self.buttons.goal_button}' for n in geom_names) and any(  # pylint: disable=no-member
                n in self.agent.body_info.geom_names for n in geom_names
            ):
                return True
        return False
