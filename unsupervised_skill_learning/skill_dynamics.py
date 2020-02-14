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

"""Dynamics Prediction and Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# TODO(architsh): Implement the dynamics with last K step input
class SkillDynamics:

  def __init__(
      self,
      observation_size,
      action_size,
      restrict_observation=0,
      normalize_observations=False,
      # network properties
      fc_layer_params=(256, 256),
      network_type='default',
      num_components=1,
      fix_variance=False,
      reweigh_batches=False,
      graph=None,
      scope_name='skill_dynamics'):

    self._observation_size = observation_size
    self._action_size = action_size
    self._normalize_observations = normalize_observations
    self._restrict_observation = restrict_observation
    self._reweigh_batches = reweigh_batches

    # tensorflow requirements
    if graph is not None:
      self._graph = graph
    else:
      self._graph = tf.get_default_graph()
    self._scope_name = scope_name

    # dynamics network properties
    self._fc_layer_params = fc_layer_params
    self._network_type = network_type
    self._num_components = num_components
    self._fix_variance = fix_variance
    if not self._fix_variance:
      self._std_lower_clip = 0.3
      self._std_upper_clip = 10.0

    self._use_placeholders = False
    self.log_probability = None
    self.dyn_max_op = None
    self.dyn_min_op = None
    self._session = None
    self._use_modal_mean = False

    # saving/restoring variables
    self._saver = None

  def _get_distribution(self, out):
    if self._num_components > 1:
      self.logits = tf.layers.dense(
          out, self._num_components, name='logits', reuse=tf.AUTO_REUSE)
      means, scale_diags = [], []
      for component_id in range(self._num_components):
        means.append(
            tf.layers.dense(
                out,
                self._observation_size,
                name='mean_' + str(component_id),
                reuse=tf.AUTO_REUSE))
        if not self._fix_variance:
          scale_diags.append(
              tf.clip_by_value(
                  tf.layers.dense(
                      out,
                      self._observation_size,
                      activation=tf.nn.softplus,
                      name='stddev_' + str(component_id),
                      reuse=tf.AUTO_REUSE), self._std_lower_clip,
                  self._std_upper_clip))
        else:
          scale_diags.append(
              tf.fill([tf.shape(out)[0], self._observation_size], 1.0))

      self.means = tf.stack(means, axis=1)
      self.scale_diags = tf.stack(scale_diags, axis=1)
      return tfp.distributions.MixtureSameFamily(
          mixture_distribution=tfp.distributions.Categorical(
              logits=self.logits),
          components_distribution=tfp.distributions.MultivariateNormalDiag(
              loc=self.means, scale_diag=self.scale_diags))

    else:
      mean = tf.layers.dense(
          out, self._observation_size, name='mean', reuse=tf.AUTO_REUSE)
      if not self._fix_variance:
        stddev = tf.clip_by_value(
            tf.layers.dense(
                out,
                self._observation_size,
                activation=tf.nn.softplus,
                name='stddev',
                reuse=tf.AUTO_REUSE), self._std_lower_clip,
            self._std_upper_clip)
      else:
        stddev = tf.fill([tf.shape(out)[0], self._observation_size], 1.0)
      return tfp.distributions.MultivariateNormalDiag(
          loc=mean, scale_diag=stddev)

  # dynamics graph with separate pipeline for skills and timesteps
  def _graph_with_separate_skill_pipe(self, timesteps, actions):
    skill_out = actions
    with tf.variable_scope('action_pipe'):
      for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
        skill_out = tf.layers.dense(
            skill_out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=tf.AUTO_REUSE)

    ts_out = timesteps
    with tf.variable_scope('ts_pipe'):
      for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
        ts_out = tf.layers.dense(
            ts_out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=tf.AUTO_REUSE)

    # out = tf.layers.flatten(tf.einsum('ai,aj->aij', ts_out, skill_out))
    out = tf.concat([ts_out, skill_out], axis=1)
    with tf.variable_scope('joint'):
      for idx, layer_size in enumerate(self._fc_layer_param[1:]):
        out = tf.layers.dense(
            out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=tf.AUTO_REUSE)

    return self._get_distribution(out)

  # simple dynamics graph
  def _default_graph(self, timesteps, actions):
    out = tf.concat([timesteps, actions], axis=1)
    for idx, layer_size in enumerate(self._fc_layer_params):
      out = tf.layers.dense(
          out,
          layer_size,
          activation=tf.nn.relu,
          name='hid_' + str(idx),
          reuse=tf.AUTO_REUSE)

    return self._get_distribution(out)

  def _get_dict(self,
                input_data,
                input_actions,
                target_data,
                batch_size=-1,
                batch_weights=None,
                batch_norm=False,
                noise_targets=False,
                noise_std=0.5):
    if batch_size > 0:
      shuffled_batch = np.random.permutation(len(input_data))[:batch_size]
    else:
      shuffled_batch = np.arange(len(input_data))

    # if we are noising the input, it is better to create a new copy of the numpy arrays
    batched_input = input_data[shuffled_batch, :]
    batched_skills = input_actions[shuffled_batch, :]
    batched_targets = target_data[shuffled_batch, :]

    if self._reweigh_batches and batch_weights is not None:
      example_weights = batch_weights[shuffled_batch]

    if noise_targets:
      batched_targets += np.random.randn(*batched_targets.shape) * noise_std

    return_dict = {
        self.timesteps_pl: batched_input,
        self.actions_pl: batched_skills,
        self.next_timesteps_pl: batched_targets
    }
    if self._normalize_observations:
      return_dict[self.is_training_pl] = batch_norm
    if self._reweigh_batches and batch_weights is not None:
      return_dict[self.batch_weights] = example_weights

    return return_dict

  def _get_run_dict(self, input_data, input_actions):
    return_dict = {
        self.timesteps_pl: input_data,
        self.actions_pl: input_actions
    }
    if self._normalize_observations:
      return_dict[self.is_training_pl] = False

    return return_dict

  def make_placeholders(self):
    self._use_placeholders = True
    with self._graph.as_default(), tf.variable_scope(self._scope_name):
      self.timesteps_pl = tf.placeholder(
          tf.float32, shape=(None, self._observation_size), name='timesteps_pl')
      self.actions_pl = tf.placeholder(
          tf.float32, shape=(None, self._action_size), name='actions_pl')
      self.next_timesteps_pl = tf.placeholder(
          tf.float32,
          shape=(None, self._observation_size),
          name='next_timesteps_pl')
      if self._normalize_observations:
        self.is_training_pl = tf.placeholder(tf.bool, name='batch_norm_pl')
      if self._reweigh_batches:
        self.batch_weights = tf.placeholder(
            tf.float32, shape=(None,), name='importance_sampled_weights')

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
        var_list = tf.compat.v1.global_variables(
        ) + tf.compat.v1.local_variables()
        is_initialized = self._session.run(
            [tf.compat.v1.is_variable_initialized(v) for v in var_list])
        uninitialized_vars = []
        for flag, v in zip(is_initialized, var_list):
          if not flag:
            uninitialized_vars.append(v)

        if uninitialized_vars:
          self._session.run(
              tf.compat.v1.variables_initializer(uninitialized_vars))

  def build_graph(self,
                  timesteps=None,
                  actions=None,
                  next_timesteps=None,
                  is_training=None):
    with self._graph.as_default(), tf.variable_scope(
        self._scope_name, reuse=tf.AUTO_REUSE):
      if self._use_placeholders:
        timesteps = self.timesteps_pl
        actions = self.actions_pl
        next_timesteps = self.next_timesteps_pl
        if self._normalize_observations:
          is_training = self.is_training_pl

      # predict deltas instead of observations
      next_timesteps -= timesteps

      if self._restrict_observation > 0:
        timesteps = timesteps[:, self._restrict_observation:]

      if self._normalize_observations:
        timesteps = tf.layers.batch_normalization(
            timesteps,
            training=is_training,
            name='input_normalization',
            reuse=tf.AUTO_REUSE)
        self.output_norm_layer = tf.layers.BatchNormalization(
            scale=False, center=False, name='output_normalization')
        next_timesteps = self.output_norm_layer(
            next_timesteps, training=is_training)

      if self._network_type == 'default':
        self.base_distribution = self._default_graph(timesteps, actions)
      elif self._network_type == 'separate':
        self.base_distribution = self._graph_with_separate_skill_pipe(
            timesteps, actions)

      # if building multiple times, be careful about which log_prob you are optimizing
      self.log_probability = self.base_distribution.log_prob(next_timesteps)
      self.mean = self.base_distribution.mean()

      return self.log_probability

  def increase_prob_op(self, learning_rate=3e-4, weights=None):
    with self._graph.as_default():
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        if self._reweigh_batches:
          self.dyn_max_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate,
              name='adam_max').minimize(-tf.reduce_mean(self.log_probability *
                                                        self.batch_weights))
        elif weights is not None:
          self.dyn_max_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate,
              name='adam_max').minimize(-tf.reduce_mean(self.log_probability *
                                                        weights))
        else:
          self.dyn_max_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate,
              name='adam_max').minimize(-tf.reduce_mean(self.log_probability))

        return self.dyn_max_op

  def decrease_prob_op(self, learning_rate=3e-4, weights=None):
    with self._graph.as_default():
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        if self._reweigh_batches:
          self.dyn_min_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate, name='adam_min').minimize(
                  tf.reduce_mean(self.log_probability * self.batch_weights))
        elif weights is not None:
          self.dyn_min_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate, name='adam_min').minimize(
                  tf.reduce_mean(self.log_probability * weights))
        else:
          self.dyn_min_op = tf.train.AdamOptimizer(
              learning_rate=learning_rate,
              name='adam_min').minimize(tf.reduce_mean(self.log_probability))
        return self.dyn_min_op

  def create_saver(self, save_prefix):
    if self._saver is not None:
      return self._saver
    else:
      with self._graph.as_default():
        self._variable_list = {}
        for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name):
          self._variable_list[var.name] = var
        self._saver = tf.train.Saver(
            self._variable_list, save_relative_paths=True)
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

  # all functions here-on require placeholders----------------------------------
  def train(self,
            timesteps,
            actions,
            next_timesteps,
            batch_weights=None,
            batch_size=512,
            num_steps=1,
            increase_probs=True):
    if not self._use_placeholders:
      return

    if increase_probs:
      run_op = self.dyn_max_op
    else:
      run_op = self.dyn_min_op

    for _ in range(num_steps):
      self._session.run(
          run_op,
          feed_dict=self._get_dict(
              timesteps,
              actions,
              next_timesteps,
              batch_weights=batch_weights,
              batch_size=batch_size,
              batch_norm=True))

  def get_log_prob(self, timesteps, actions, next_timesteps):
    if not self._use_placeholders:
      return

    return self._session.run(
        self.log_probability,
        feed_dict=self._get_dict(
            timesteps, actions, next_timesteps, batch_norm=False))

  def predict_state(self, timesteps, actions):
    if not self._use_placeholders:
      return

    if self._use_modal_mean:
      all_means, modal_mean_indices = self._session.run(
          [self.means, tf.argmax(self.logits, axis=1)],
          feed_dict=self._get_run_dict(timesteps, actions))
      pred_state = all_means[[
          np.arange(all_means.shape[0]), modal_mean_indices
      ]]
    else:
      pred_state = self._session.run(
          self.mean, feed_dict=self._get_run_dict(timesteps, actions))

    if self._normalize_observations:
      with self._session.as_default(), self._graph.as_default():
        mean_correction, variance_correction = self.output_norm_layer.get_weights(
        )

      pred_state = pred_state * np.sqrt(variance_correction +
                                        1e-3) + mean_correction

    pred_state += timesteps
    return pred_state
