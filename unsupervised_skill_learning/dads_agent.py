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

"""TF-Agents Class for DADS. Builds on top of the SAC agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
import tensorflow as tf

from tf_agents.agents.sac import sac_agent

import skill_dynamics

nest = tf.nest


class DADSAgent(sac_agent.SacAgent):

  def __init__(self,
               save_directory,
               skill_dynamics_observation_size,
               observation_modify_fn=None,
               restrict_input_size=0,
               latent_size=2,
               latent_prior='cont_uniform',
               prior_samples=100,
               fc_layer_params=(256, 256),
               normalize_observations=True,
               network_type='default',
               num_mixture_components=4,
               fix_variance=True,
               skill_dynamics_learning_rate=3e-4,
               reweigh_batches=False,
               agent_graph=None,
               skill_dynamics_graph=None,
               *sac_args,
               **sac_kwargs):
    self._skill_dynamics_learning_rate = skill_dynamics_learning_rate
    self._latent_size = latent_size
    self._latent_prior = latent_prior
    self._prior_samples = prior_samples
    self._save_directory = save_directory
    self._restrict_input_size = restrict_input_size
    self._process_observation = observation_modify_fn

    if agent_graph is None:
      self._graph = tf.get_default_graph()
    else:
      self._graph = agent_graph

    if skill_dynamics_graph is None:
      skill_dynamics_graph = self._graph

    # instantiate the skill dynamics
    self._skill_dynamics = skill_dynamics.SkillDynamics(
        observation_size=skill_dynamics_observation_size,
        action_size=self._latent_size,
        restrict_observation=self._restrict_input_size,
        normalize_observations=normalize_observations,
        fc_layer_params=fc_layer_params,
        network_type=network_type,
        num_components=num_mixture_components,
        fix_variance=fix_variance,
        reweigh_batches=reweigh_batches,
        graph=skill_dynamics_graph)

    super(DADSAgent, self).__init__(*sac_args, **sac_kwargs)
    self._placeholders_in_place = False

  def compute_dads_reward(self, input_obs, cur_skill, target_obs):
    if self._process_observation is not None:
      input_obs, target_obs = self._process_observation(
          input_obs), self._process_observation(target_obs)

    num_reps = self._prior_samples if self._prior_samples > 0 else self._latent_size - 1
    input_obs_altz = np.concatenate([input_obs] * num_reps, axis=0)
    target_obs_altz = np.concatenate([target_obs] * num_reps, axis=0)

    # for marginalization of the denominator
    if self._latent_prior == 'discrete_uniform' and not self._prior_samples:
      alt_skill = np.concatenate(
          [np.roll(cur_skill, i, axis=1) for i in range(1, num_reps + 1)],
          axis=0)
    elif self._latent_prior == 'discrete_uniform':
      alt_skill = np.random.multinomial(
          1, [1. / self._latent_size] * self._latent_size,
          size=input_obs_altz.shape[0])
    elif self._latent_prior == 'gaussian':
      alt_skill = np.random.multivariate_normal(
          np.zeros(self._latent_size),
          np.eye(self._latent_size),
          size=input_obs_altz.shape[0])
    elif self._latent_prior == 'cont_uniform':
      alt_skill = np.random.uniform(
          low=-1.0, high=1.0, size=(input_obs_altz.shape[0], self._latent_size))

    logp = self._skill_dynamics.get_log_prob(input_obs, cur_skill, target_obs)

    # denominator may require more memory than that of a GPU, break computation
    split_group = 20 * 4000
    if input_obs_altz.shape[0] <= split_group:
      logp_altz = self._skill_dynamics.get_log_prob(input_obs_altz, alt_skill,
                                                    target_obs_altz)
    else:
      logp_altz = []
      for split_idx in range(input_obs_altz.shape[0] // split_group):
        start_split = split_idx * split_group
        end_split = (split_idx + 1) * split_group
        logp_altz.append(
            self._skill_dynamics.get_log_prob(
                input_obs_altz[start_split:end_split],
                alt_skill[start_split:end_split],
                target_obs_altz[start_split:end_split]))
      if input_obs_altz.shape[0] % split_group:
        start_split = input_obs_altz.shape[0] % split_group
        logp_altz.append(
            self._skill_dynamics.get_log_prob(input_obs_altz[-start_split:],
                                              alt_skill[-start_split:],
                                              target_obs_altz[-start_split:]))
      logp_altz = np.concatenate(logp_altz)
    logp_altz = np.array(np.array_split(logp_altz, num_reps))

    # final DADS reward
    intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(
        np.clip(logp_altz - logp.reshape(1, -1), -50, 50)).sum(axis=0))

    return intrinsic_reward, {'logp': logp, 'logp_altz': logp_altz.flatten()}

  def get_experience_placeholder(self):
    self._placeholders_in_place = True
    self._placeholders = []
    for item in nest.flatten(self.collect_data_spec):
      self._placeholders += [
          tf.placeholder(
              item.dtype,
              shape=(None, 2) if len(item.shape) == 0 else
              (None, 2, item.shape[-1]),
              name=item.name)
      ]
    self._policy_experience_ph = nest.pack_sequence_as(self.collect_data_spec,
                                                       self._placeholders)
    return self._policy_experience_ph

  def build_agent_graph(self):
    with self._graph.as_default():
      self.get_experience_placeholder()
      self.agent_train_op = self.train(self._policy_experience_ph)
      self.summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
      return self.agent_train_op

  def build_skill_dynamics_graph(self):
    self._skill_dynamics.make_placeholders()
    self._skill_dynamics.build_graph()
    self._skill_dynamics.increase_prob_op(
        learning_rate=self._skill_dynamics_learning_rate)

  def create_savers(self):
    self._skill_dynamics.create_saver(
        save_prefix=os.path.join(self._save_directory, 'dynamics'))

  def set_sessions(self, initialize_or_restore_skill_dynamics, session=None):
    if session is not None:
      self._session = session
    else:
      self._session = tf.compat.v1.Session(graph=self._graph)
    self._skill_dynamics.set_session(
        initialize_or_restore_variables=initialize_or_restore_skill_dynamics,
        session=session)

  def save_variables(self, global_step):
    self._skill_dynamics.save_variables(global_step=global_step)

  def _get_dict(self, trajectories, batch_size=-1):
    tf.nest.assert_same_structure(self.collect_data_spec, trajectories)
    if batch_size > 0:
      shuffled_batch = np.random.permutation(
          trajectories.observation.shape[0])[:batch_size]
    else:
      shuffled_batch = np.arange(trajectories.observation.shape[0])

    return_dict = {}

    for placeholder, val in zip(self._placeholders, nest.flatten(trajectories)):
      return_dict[placeholder] = val[shuffled_batch]

    return return_dict

  def train_loop(self,
                 trajectories,
                 recompute_reward=False,
                 batch_size=-1,
                 num_steps=1):
    if not self._placeholders_in_place:
      return

    if recompute_reward:
      input_obs = trajectories.observation[:, 0, :-self._latent_size]
      cur_skill = trajectories.observation[:, 0, -self._latent_size:]
      target_obs = trajectories.observation[:, 1, :-self._latent_size]
      new_reward, info = self.compute_dads_reward(input_obs, cur_skill,
                                                  target_obs)
      trajectories = trajectories._replace(
          reward=np.concatenate(
              [np.expand_dims(new_reward, axis=1), trajectories.reward[:, 1:]],
              axis=1))

    # TODO(architsh):all agent specs should be the same as env specs, shift preprocessing to actor/critic networks
    if self._restrict_input_size > 0:
      trajectories = trajectories._replace(
          observation=trajectories.observation[:, :,
                                               self._restrict_input_size:])

    for _ in range(num_steps):
      self._session.run([self.agent_train_op, self.summary_ops],
                        feed_dict=self._get_dict(
                            trajectories, batch_size=batch_size))

    if recompute_reward:
      return new_reward, info
    else:
      return None, None

  @property
  def skill_dynamics(self):
    return self._skill_dynamics
