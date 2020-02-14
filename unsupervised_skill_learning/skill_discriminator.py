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

"""Skill Discriminator Prediction and Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import tanh_bijector_stable

class SkillDiscriminator:

  def __init__(
      self,
      observation_size,
      skill_size,
      skill_type,
      normalize_observations=False,
      # network properties
      fc_layer_params=(256, 256),
      fix_variance=False,
      input_type='diayn',
      # probably do not need to change these
      graph=None,
      scope_name='skill_discriminator'):

    self._observation_size = observation_size
    self._skill_size = skill_size
    self._skill_type = skill_type
    self._normalize_observations = normalize_observations

    # tensorflow requirements
    if graph is not None:
      self._graph = graph
    else:
      self._graph = tf.get_default_graph()
    self._scope_name = scope_name

    # discriminator network properties
    self._fc_layer_params = fc_layer_params
    self._fix_variance = fix_variance
    if not self._fix_variance:
      self._std_lower_clip = 0.3
      self._std_upper_clip = 10.0
    self._input_type = input_type

    self._use_placeholders = False
    self.log_probability = None
    self.disc_max_op = None
    self.disc_min_op = None
    self._session = None

    # saving/restoring variables
    self._saver = None

  def _get_distributions(self, out):
    if self._skill_type in ['gaussian', 'cont_uniform']:
      mean = tf.layers.dense(
          out, self._skill_size, name='mean', reuse=tf.AUTO_REUSE)
      if not self._fix_variance:
        stddev = tf.clip_by_value(
            tf.layers.dense(
                out,
                self._skill_size,
                activation=tf.nn.softplus,
                name='stddev',
                reuse=tf.AUTO_REUSE), self._std_lower_clip,
            self._std_upper_clip)
      else:
        stddev = tf.fill([tf.shape(out)[0], self._skill_size], 1.0)

      inference_distribution = tfp.distributions.MultivariateNormalDiag(
          loc=mean, scale_diag=stddev)

      if self._skill_type == 'gaussian':
        prior_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.] * self._skill_size, scale_diag=[1.] * self._skill_size)
      elif self._skill_type == 'cont_uniform':
        prior_distribution = tfp.distributions.Independent(
            tfp.distributions.Uniform(
                low=[-1.] * self._skill_size, high=[1.] * self._skill_size),
            reinterpreted_batch_ndims=1)

        # squash posterior to the right range of [-1, 1]
        bijectors = []
        bijectors.append(tanh_bijector_stable.Tanh())
        bijector_chain = tfp.bijectors.Chain(bijectors)
        inference_distribution = tfp.distributions.TransformedDistribution(
            distribution=inference_distribution, bijector=bijector_chain)

    elif self._skill_type == 'discrete_uniform':
      logits = tf.layers.dense(
          out, self._skill_size, name='logits', reuse=tf.AUTO_REUSE)
      inference_distribution = tfp.distributions.OneHotCategorical(
          logits=logits)
      prior_distribution = tfp.distributions.OneHotCategorical(
          probs=[1. / self._skill_size] * self._skill_size)
    elif self._skill_type == 'multivariate_bernoulli':
      print('Not supported yet')

    return inference_distribution, prior_distribution

  # simple dynamics graph
  def _default_graph(self, timesteps):
    out = timesteps
    for idx, layer_size in enumerate(self._fc_layer_params):
      out = tf.layers.dense(
          out,
          layer_size,
          activation=tf.nn.relu,
          name='hid_' + str(idx),
          reuse=tf.AUTO_REUSE)

    return self._get_distributions(out)

  def _get_dict(self,
                input_steps,
                target_skills,
                input_next_steps=None,
                batch_size=-1,
                batch_norm=False):
    if batch_size > 0:
      shuffled_batch = np.random.permutation(len(input_steps))[:batch_size]
    else:
      shuffled_batch = np.arange(len(input_steps))

    batched_input = input_steps[shuffled_batch, :]
    batched_skills = target_skills[shuffled_batch, :]
    if self._input_type in ['diff', 'both']:
      batched_targets = input_next_steps[shuffled_batch, :]

    return_dict = {
        self.timesteps_pl: batched_input,
        self.skills_pl: batched_skills,
    }

    if self._input_type in ['diff', 'both']:
      return_dict[self.next_timesteps_pl] = batched_targets
    if self._normalize_observations:
      return_dict[self.is_training_pl] = batch_norm

    return return_dict

  def make_placeholders(self):
    self._use_placeholders = True
    with self._graph.as_default(), tf.variable_scope(self._scope_name):
      self.timesteps_pl = tf.placeholder(
          tf.float32, shape=(None, self._observation_size), name='timesteps_pl')
      self.skills_pl = tf.placeholder(
          tf.float32, shape=(None, self._skill_size), name='skills_pl')
      if self._input_type in ['diff', 'both']:
        self.next_timesteps_pl = tf.placeholder(
            tf.float32,
            shape=(None, self._observation_size),
            name='next_timesteps_pl')
      if self._normalize_observations:
        self.is_training_pl = tf.placeholder(tf.bool, name='batch_norm_pl')

  def set_session(self, session=None, initialize_or_restore_variables=False):
    if session is None:
      self._session = tf.Session(graph=self._graph)
    else:
      self._session = session

    # only initialize uninitialized variables
    if initialize_or_restore_variables:
      if tf.gfile.Exists(self._save_prefix):
        self.restore_variables()
      with self._graph.as_default():
        is_initialized = self._session.run([
            tf.compat.v1.is_variable_initialized(v)
            for key, v in self._variable_list.items()
        ])
        uninitialized_vars = []
        for flag, v in zip(is_initialized, self._variable_list.items()):
          if not flag:
            uninitialized_vars.append(v[1])

        if uninitialized_vars:
          self._session.run(
              tf.compat.v1.variables_initializer(uninitialized_vars))

  def build_graph(self,
                  timesteps=None,
                  skills=None,
                  next_timesteps=None,
                  is_training=None):
    with self._graph.as_default(), tf.variable_scope(self._scope_name):
      if self._use_placeholders:
        timesteps = self.timesteps_pl
        skills = self.skills_pl
        if self._input_type in ['diff', 'both']:
          next_timesteps = self.next_timesteps_pl
        if self._normalize_observations:
          is_training = self.is_training_pl

      # use deltas
      if self._input_type == 'both':
        next_timesteps -= timesteps
        timesteps = tf.concat([timesteps, next_timesteps], axis=1)
      if self._input_type == 'diff':
        timesteps = next_timesteps - timesteps

      if self._normalize_observations:
        timesteps = tf.layers.batch_normalization(
            timesteps,
            training=is_training,
            name='input_normalization',
            reuse=tf.AUTO_REUSE)

      inference_distribution, prior_distribution = self._default_graph(
          timesteps)

      self.log_probability = inference_distribution.log_prob(skills)
      self.prior_probability = prior_distribution.log_prob(skills)
      return self.log_probability, self.prior_probability

  def increase_prob_op(self, learning_rate=3e-4):
    with self._graph.as_default():
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.disc_max_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                -tf.reduce_mean(self.log_probability))
        return self.disc_max_op

  def decrease_prob_op(self, learning_rate=3e-4):
    with self._graph.as_default():
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.disc_min_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                tf.reduce_mean(self.log_probability))
        return self.disc_min_op

  # only useful when training use placeholders, otherwise use ops directly
  def train(self,
            timesteps,
            skills,
            next_timesteps=None,
            batch_size=512,
            num_steps=1,
            increase_probs=True):
    if not self._use_placeholders:
      return

    if increase_probs:
      run_op = self.disc_max_op
    else:
      run_op = self.disc_min_op

    for _ in range(num_steps):
      self._session.run(
          run_op,
          feed_dict=self._get_dict(
              timesteps,
              skills,
              input_next_steps=next_timesteps,
              batch_size=batch_size,
              batch_norm=True))

  def get_log_probs(self, timesteps, skills, next_timesteps=None):
    if not self._use_placeholders:
      return

    return self._session.run([self.log_probability, self.prior_probability],
                             feed_dict=self._get_dict(
                                 timesteps,
                                 skills,
                                 input_next_steps=next_timesteps,
                                 batch_norm=False))

  def create_saver(self, save_prefix):
    if self._saver is not None:
      return self._saver
    else:
      with self._graph.as_default():
        self._variable_list = {}
        for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name):
          self._variable_list[var.name] = var
        self._saver = tf.train.Saver(self._variable_list, save_relative_paths=True)
        self._save_prefix = save_prefix

  def save_variables(self, global_step):
    if not tf.gfile.Exists(self._save_prefix):
      tf.gfile.MakeDirs(self._save_prefix)

    self._saver.save(
        self._session,
        os.path.join(self._save_prefix, 'ckpt'),
        global_step=global_step)

  def restore_variables(self):
    self._saver.restore(self._session,
                        tf.train.latest_checkpoint(self._save_prefix))
