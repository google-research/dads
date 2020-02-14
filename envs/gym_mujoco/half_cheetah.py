"""HalfCheetah environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self,
               expose_all_qpos=False,
               task='default',
               target_velocity=None,
               model_path='half_cheetah.xml'):
    # Settings from
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    self._expose_all_qpos = expose_all_qpos
    self._task = task
    self._target_velocity = target_velocity

    xml_path = "envs/assets/"
    model_path = os.path.abspath(os.path.join(xml_path, model_path))

    mujoco_env.MujocoEnv.__init__(
        self,
        model_path,
        5)
    utils.EzPickle.__init__(self)

  def step(self, action):
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    xvelafter = self.sim.data.qvel[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(action).sum()

    if self._task == 'default':
      reward_vel = 0.
      reward_run = (xposafter - xposbefore) / self.dt
      reward = reward_ctrl + reward_run
    elif self._task == 'target_velocity':
      reward_vel = -(self._target_velocity - xvelafter)**2
      reward = reward_ctrl + reward_vel
    elif self._task == 'run_back':
      reward_vel = 0.
      reward_run = (xposbefore - xposafter) / self.dt
      reward = reward_ctrl + reward_run

    done = False
    return ob, reward, done, dict(
        reward_run=reward_run, reward_ctrl=reward_ctrl, reward_vel=reward_vel)

  def _get_obs(self):
    if self._expose_all_qpos:
      return np.concatenate(
          [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
    return np.concatenate([
        self.sim.data.qpos.flat[1:],
        self.sim.data.qvel.flat,
    ])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        low=-.1, high=.1, size=self.sim.model.nq)
    qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .1
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5
