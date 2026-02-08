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
"""DynObs level 1."""

from safety_gymnasium.tasks.safe_navigation.DynObs.button_level0 import DynObsLevel0


class DynObsLevel1(DynObsLevel0):
    """Slightly harder DynObs setting."""

    def __init__(self, config) -> None:
        config = dict(config)
        config.setdefault('placements_conf.extents', [-2.3, -2.3, 2.3, 2.3])
        config.setdefault('placements_conf.margin', 0.12)
        config.setdefault('Buttons', {'num': 1, 'is_constrained': False, 'keepout': 0.2})
        config.setdefault(
            'Pillars',
            {
                'num': 3,
                'keepout': 0.32,
                'size': 0.2,
            },
        )
        config.setdefault(
            'GremlinVels',
            {
                'num': 6,
                'travel': 0.3,
                'keepout': 0.16,
                'omega_low': 1.0,
                'omega_high': 3.0,
            },
        )
        super().__init__(config=config)
