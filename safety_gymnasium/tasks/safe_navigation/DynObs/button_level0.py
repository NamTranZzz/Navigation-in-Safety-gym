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
"""DynObs level 0."""

from safety_gymnasium.tasks.safe_navigation.DynObs.button_base import DynObsBase


class DynObsLevel0(DynObsBase):
    """Button task with static pillars and velocity-driven moving obstacles."""

    def __init__(self, config) -> None:
        config = dict(config)
        # Random placement over full map with a margin to avoid clustering.
        config.setdefault('placements_conf.extents', [-2.2, -2.2, 2.2, 2.2])
        config.setdefault('placements_conf.margin', 0.08)
        # Single-goal setup: only one button is spawned and treated as the target.
        config.setdefault(
            'Buttons',
            {
                'num': 1,
                'is_constrained': False,
                'keepout': 0.2,
            },
        )
        config.setdefault('Goal', {'size': 0.2, 'alpha': 1.0})
        config.setdefault(
            'Pillars',
            {
                'num': 4,
                'keepout': 0.30,
                'size': 0.2,
            },
        )
        config.setdefault(
            'GremlinVels',
            {
                'num': 4,
                'travel': 0.35,
                'keepout': 0.34,
            },
        )

        super().__init__(config=config)

        # Default injected Goal size is adjusted to match button size unless user overrides explicitly.
        if 'Goal' not in config or 'size' not in config['Goal']:
            self.goal.size = self.buttons.size * 2  # pylint: disable=no-member
