"""Humanoid environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env


def mass_center(sim):
  mass = np.expand_dims(sim.model.body_mass, 1)
  xpos = sim.data.xipos
  return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


# pylint: disable=missing-docstring
class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self, expose_all_qpos=False, model_path='humanoid.xml'):
    self._expose_all_qpos = expose_all_qpos
    # Settings from
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    xml_path = "envs/assets/"
    model_path = os.path.abspath(os.path.join(xml_path, model_path))
    mujoco_env.MujocoEnv.__init__(self, model_path, 5)
    utils.EzPickle.__init__(self)

  def _get_obs(self):
    data = self.sim.data
    if self._expose_all_qpos:
      return np.concatenate([
          data.qpos.flat, data.qvel.flat, data.cinert.flat, data.cvel.flat,
          data.qfrc_actuator.flat, data.cfrc_ext.flat
      ])
    return np.concatenate([
        data.qpos.flat[2:], data.qvel.flat, data.cinert.flat, data.cvel.flat,
        data.qfrc_actuator.flat, data.cfrc_ext.flat
    ])

  def step(self, a):
    pos_before = mass_center(self.sim)
    self.do_simulation(a, self.frame_skip)
    pos_after = mass_center(self.sim)
    alive_bonus = 5.0
    data = self.sim.data
    lin_vel_cost = 0.25 * (
        pos_after - pos_before) / self.sim.model.opt.timestep
    quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
    quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
    quad_impact_cost = min(quad_impact_cost, 10)
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
    qpos = self.sim.data.qpos
    done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
    return self._get_obs(), reward, done, dict(
        reward_linvel=lin_vel_cost,
        reward_quadctrl=-quad_ctrl_cost,
        reward_alive=alive_bonus,
        reward_impact=-quad_impact_cost)

  def reset_model(self):
    c = 0.01
    self.set_state(
        self.init_qpos + self.np_random.uniform(
            low=-c, high=c, size=self.sim.model.nq),
        self.init_qvel + self.np_random.uniform(
            low=-c,
            high=c,
            size=self.sim.model.nv,
        ))
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5
