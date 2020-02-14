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

"""Turn tasks with DClaw robots.

This is a single rotation of an object from an initial angle to a target angle.
"""

import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np

from robel.components.robot.dynamixel_robot import DynamixelRobotState
from robel.dclaw.base_env import BaseDClawObjectEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.resources import get_asset_path

# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'object_x',
    'object_y',
    'claw_qpos',
    'last_action',
)

# Reset pose for the claw joints.
RESET_POSE = [0, -np.pi / 3, np.pi / 3] * 3

DCLAW3_ASSET_PATH = 'robel/dclaw/assets/dclaw3xh_valve3_v0.xml'


class BaseDClawTurn(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
            interactive: If True, allows the hardware guide motor to freely
                rotate and its current angle is used as the goal.
            success_threshold: The difference threshold (in radians) of the
                object position and the goal position within which we consider
                as a sucesss.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._desired_claw_pos = RESET_POSE

        # The following are modified (possibly every reset) by subclasses.
        self._initial_object_pos = 0
        self._initial_object_vel = 0

    def _reset(self):
        """Resets the environment."""
        self._reset_dclaw_and_object(
            claw_pos=RESET_POSE,
            object_pos=self._initial_object_pos,
            object_vel=self._initial_object_vel)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        claw_state, object_state = self.robot.get_state(
            ['dclaw', 'object'])

        obs_dict = collections.OrderedDict((
            ('claw_qpos', claw_state.qpos),
            ('claw_qvel', claw_state.qvel),
            ('object_x', np.cos(object_state.qpos)),
            ('object_y', np.sin(object_state.qpos)),
            ('object_qvel', object_state.qvel),
            ('last_action', self._get_last_action()),
        ))
        # Add hardware-specific state if present.
        if isinstance(claw_state, DynamixelRobotState):
            obs_dict['claw_current'] = claw_state.current

        return obs_dict

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

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        return np.zeros_like([0], dtype=bool)


@configurable(pickleable=True)
class DClawTurnRandom(BaseDClawTurn):
    """Turns the object with a random initial and random target position."""

    def _reset(self):
        # Initial position is +/- 60 degrees.
        self._initial_object_pos = self.np_random.uniform(
            low=-np.pi / 3, high=np.pi / 3)
        super()._reset()


@configurable(pickleable=True)
class DClawTurnRandomDynamics(DClawTurnRandom):
    """Turns the object with a random initial and random target position.

    The dynamics of the simulation are randomized each episode.
    """

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dclaw').qvel_indices.tolist() +
            self.robot.get_config('object').qvel_indices.tolist())

    def _reset(self):
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            damping_range=(0.005, 0.1),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(1, 3),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        self._randomizer.randomize_bodies(
            ['mount'],
            position_perturb_range=(-0.01, 0.01),
        )
        self._randomizer.randomize_geoms(
            ['mount'],
            color_range=(0.2, 0.9),
        )
        self._randomizer.randomize_geoms(
            parent_body_names=['valve'],
            color_range=(0.2, 0.9),
        )
        super()._reset()
