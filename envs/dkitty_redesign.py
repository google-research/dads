# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DKitty redesign
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from robel.components.tracking import TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_walk-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    # 'root_vel',
    # 'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
)


class BaseDKittyWalk(BaseDKittyUprightEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty walk tasks."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            device_path: Optional[str] = None,
            torso_tracker_id: Optional[Union[str, int]] = None,
            frame_skip: int = 40,
            sticky_action_probability: float = 0.,
            upright_threshold: float = 0.9,
            upright_reward: float = 1,
            falling_reward: float = -500,
            expose_last_action: bool = True,
            expose_upright: bool = True,
            robot_noise_ratio: float = 0.05,
            **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            torso_tracker_id: The device index or serial of the tracking device
                for the D'Kitty torso.
            frame_skip: The number of simulation steps per environment step.
            sticky_action_probability: Repeat previous action with this
                probability. Default 0 (no sticky actions).
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._expose_last_action = expose_last_action
        self._expose_upright = expose_upright
        observation_keys = observation_keys[:-2]
        if self._expose_last_action:
            observation_keys += ('last_action',)
        if self._expose_upright:
            observation_keys += ('upright',)

        # robot_config = self.get_robot_config(device_path)
        # if 'sim_observation_noise' in robot_config.keys():
        #     robot_config['sim_observation_noise'] = robot_noise_ratio
 
        super().__init__(
            sim_model=get_asset_path(asset_path),
            # robot_config=robot_config,
            # tracker_config=self.get_tracker_config(
            #     torso=torso_tracker_id,
            # ),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._last_action = np.zeros(12)
        self._sticky_action_probability = sticky_action_probability
        self._time_step = 0

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
        })

        self._time_step = 0

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self._time_step += 1

        # Sticky actions
        rand = self.np_random.uniform() < self._sticky_action_probability
        action_to_apply = np.where(rand, self._last_action, action)

        # Apply action.
        self.robot.step({
            'dkitty': action_to_apply,
        })
        # Save the action to add to the observation.
        self._last_action = action

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        torso_track_state = self.tracker.get_state(
            ['torso'])[0]
        obs_dict = (('root_pos', torso_track_state.pos),
                    ('root_euler', torso_track_state.rot_euler),
                    ('root_vel', torso_track_state.vel),
                    ('root_angular_vel', torso_track_state.angular_vel),
                    ('kitty_qpos', robot_state.qpos),
                    ('kitty_qvel', robot_state.qvel))

        if self._expose_last_action:
            obs_dict += (('last_action', self._last_action),)

        # Add observation terms relating to being upright.
        if self._expose_upright:
            obs_dict += (*self._get_upright_obs(torso_track_state).items(),)

        return collections.OrderedDict(obs_dict)

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        reward_dict = collections.OrderedDict(())
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict(())

@configurable(pickleable=True)
class DKittyRandomDynamics(BaseDKittyWalk):
    """Walk straight towards a random location."""

    def __init__(self, *args, randomize_hfield=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._randomize_hfield = randomize_hfield
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, self._randomize_hfield),
        )
        # if self._randomize_hfield > 0.0:
        #     self.sim_scene.upload_height_field(0)
        super()._reset()
