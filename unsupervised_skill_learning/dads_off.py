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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import flags, logging
import functools

import sys
sys.path.append(os.path.abspath('./'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories.trajectory import from_transition, to_transition
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import policy_step
# from tf_agents.policies import py_tf_policy
# from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

import dads_agent

from envs import skill_wrapper
from envs import video_wrapper
from envs.gym_mujoco import ant
from envs.gym_mujoco import half_cheetah
from envs.gym_mujoco import humanoid
from envs.gym_mujoco import point_mass

from envs import dclaw
from envs import dkitty_redesign
from envs import hand_block

from lib import py_tf_policy
from lib import py_uniform_replay_buffer

FLAGS = flags.FLAGS
nest = tf.nest

# general hyperparameters
flags.DEFINE_string('logdir', '~/tmp/dads', 'Directory for saving experiment data')

# environment hyperparameters
flags.DEFINE_string('environment', 'point_mass', 'Name of the environment')
flags.DEFINE_integer('max_env_steps', 200,
                     'Maximum number of steps in one episode')
flags.DEFINE_integer('reduced_observation', 0,
                     'Predict dynamics in a reduced observation space')
flags.DEFINE_integer(
    'min_steps_before_resample', 50,
    'Minimum number of steps to execute before resampling skill')
flags.DEFINE_float('resample_prob', 0.,
                   'Creates stochasticity timesteps before resampling skill')

# need to set save_model and save_freq
flags.DEFINE_string(
    'save_model', None,
    'Name to save the model with, None implies the models are not saved.')
flags.DEFINE_integer('save_freq', 100, 'Saving frequency for checkpoints')
flags.DEFINE_string(
    'vid_name', None,
    'Base name for videos being saved, None implies videos are not recorded')
flags.DEFINE_integer('record_freq', 100,
                     'Video recording frequency within the training loop')

# final evaluation after training is done
flags.DEFINE_integer('run_eval', 0, 'Evaluate learnt skills')

# evaluation type
flags.DEFINE_integer('num_evals', 0, 'Number of skills to evaluate')
flags.DEFINE_integer('deterministic_eval', 0,
                  'Evaluate all skills, only works for discrete skills')

# training
flags.DEFINE_integer('run_train', 0, 'Train the agent')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs')

# skill latent space
flags.DEFINE_integer('num_skills', 2, 'Number of skills to learn')
flags.DEFINE_string('skill_type', 'cont_uniform',
                    'Type of skill and the prior over it')
# network size hyperparameter
flags.DEFINE_integer(
    'hidden_layer_size', 512,
    'Hidden layer size, shared by actors, critics and dynamics')

# reward structure
flags.DEFINE_integer(
    'random_skills', 0,
    'Number of skills to sample randomly for approximating mutual information')

# optimization hyperparameters
flags.DEFINE_integer('replay_buffer_capacity', int(1e6),
                     'Capacity of the replay buffer')
flags.DEFINE_integer(
    'clear_buffer_every_iter', 0,
    'Clear replay buffer every iteration to simulate on-policy training, use larger collect steps and train-steps'
)
flags.DEFINE_integer(
    'initial_collect_steps', 2000,
    'Steps collected initially before training to populate the buffer')
flags.DEFINE_integer('collect_steps', 200, 'Steps collected per agent update')

# relabelling
flags.DEFINE_string('agent_relabel_type', None,
                    'Type of skill relabelling used for agent')
flags.DEFINE_integer(
    'train_skill_dynamics_on_policy', 0,
    'Train skill-dynamics on policy data, while agent train off-policy')
flags.DEFINE_string('skill_dynamics_relabel_type', None,
                    'Type of skill relabelling used for skill-dynamics')
flags.DEFINE_integer(
    'num_samples_for_relabelling', 100,
    'Number of samples from prior for relabelling the current skill when using policy relabelling'
)
flags.DEFINE_float(
    'is_clip_eps', 0.,
    'PPO style clipping epsilon to constrain importance sampling weights to (1-eps, 1+eps)'
)
flags.DEFINE_float(
    'action_clipping', 1.,
    'Clip actions to (-eps, eps) per dimension to avoid difficulties with tanh')
flags.DEFINE_integer('debug_skill_relabelling', 0,
                     'analysis of skill relabelling')

# skill dynamics optimization hyperparamaters
flags.DEFINE_integer('skill_dyn_train_steps', 8,
                     'Number of discriminator train steps on a batch of data')
flags.DEFINE_float('skill_dynamics_lr', 3e-4,
                   'Learning rate for increasing the log-likelihood')
flags.DEFINE_integer('skill_dyn_batch_size', 256,
                     'Batch size for discriminator updates')
# agent optimization hyperparameters
flags.DEFINE_integer('agent_batch_size', 256, 'Batch size for agent updates')
flags.DEFINE_integer('agent_train_steps', 128,
                     'Number of update steps per iteration')
flags.DEFINE_float('agent_lr', 3e-4, 'Learning rate for the agent')

# SAC hyperparameters
flags.DEFINE_float('agent_entropy', 0.1, 'Entropy regularization coefficient')
flags.DEFINE_float('agent_gamma', 0.99, 'Reward discount factor')
flags.DEFINE_string(
    'collect_policy', 'default',
    'Can use the OUNoisePolicy to collect experience for better exploration')

# skill-dynamics hyperparameters
flags.DEFINE_string(
    'graph_type', 'default',
    'process skill input separately for more representational power')
flags.DEFINE_integer('num_components', 4,
                     'Number of components for Mixture of Gaussians')
flags.DEFINE_integer('fix_variance', 1,
                     'Fix the variance of output distribution')
flags.DEFINE_integer('normalize_data', 1, 'Maintain running averages')

# debug
flags.DEFINE_integer('debug', 0, 'Creates extra summaries')

# DKitty
flags.DEFINE_integer('expose_last_action', 1, 'Add the last action to the observation')
flags.DEFINE_integer('expose_upright', 1, 'Add the upright angle to the observation')
flags.DEFINE_float('upright_threshold', 0.9, 'Threshold before which the DKitty episode is terminated')
flags.DEFINE_float('robot_noise_ratio', 0.05, 'Noise ratio for robot joints')
flags.DEFINE_float('root_noise_ratio', 0.002, 'Noise ratio for root position')
flags.DEFINE_float('scale_root_position', 1, 'Multiply the root coordinates the magnify the change')
flags.DEFINE_integer('run_on_hardware', 0, 'Flag for hardware runs')
flags.DEFINE_float('randomize_hfield', 0.0, 'Randomize terrain for better DKitty transfer')
flags.DEFINE_integer('observation_omission_size', 2, 'Dimensions to be omitted from policy input')

# Manipulation Environments
flags.DEFINE_integer('randomized_initial_distribution', 1, 'Fix the initial distribution or not')
flags.DEFINE_float('horizontal_wrist_constraint', 1.0, 'Action space constraint to restrict horizontal motion of the wrist')
flags.DEFINE_float('vertical_wrist_constraint', 1.0, 'Action space constraint to restrict vertical motion of the wrist')

# MPC hyperparameters
flags.DEFINE_integer('planning_horizon', 1, 'Number of primitives to plan in the future')
flags.DEFINE_integer('primitive_horizon', 1, 'Horizon for every primitive')
flags.DEFINE_integer('num_candidate_sequences', 50, 'Number of candidates sequence sampled from the proposal distribution')
flags.DEFINE_integer('refine_steps', 10, 'Number of optimization steps')
flags.DEFINE_float('mppi_gamma', 10.0, 'MPPI weighting hyperparameter')
flags.DEFINE_string('prior_type', 'normal', 'Uniform or Gaussian prior for candidate skill(s)')
flags.DEFINE_float('smoothing_beta', 0.9, 'Smooth candidate skill sequences used')
flags.DEFINE_integer('top_primitives', 5, 'Optimization parameter when using uniform prior (CEM style)')

# global variables for this script
observation_omit_size = 0
goal_coord = np.array([10., 10.])
sample_count = 0
iter_count = 0
episode_size_buffer = []
episode_return_buffer = []

# add a flag for state dependent std
def _normal_projection_net(action_spec, init_means_output_factor=0.1):
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)

def get_environment(env_name='point_mass'):
  global observation_omit_size
  if env_name == 'Ant-v1':
    env = ant.AntEnv(
        expose_all_qpos=True,
        task='motion')
    observation_omit_size = 2
  elif env_name == 'Ant-v1_goal':
    observation_omit_size = 2
    return wrap_env(
        ant.AntEnv(
            task='goal',
            goal=goal_coord,
            expose_all_qpos=True),
        max_episode_steps=FLAGS.max_env_steps)
  elif env_name == 'Ant-v1_foot_sensor':
    env = ant.AntEnv(
        expose_all_qpos=True,
        model_path='ant_footsensor.xml',
        expose_foot_sensors=True)
    observation_omit_size = 2
  elif env_name == 'HalfCheetah-v1':
    env = half_cheetah.HalfCheetahEnv(expose_all_qpos=True, task='motion')
    observation_omit_size = 1
  elif env_name == 'Humanoid-v1':
    env = humanoid.HumanoidEnv(expose_all_qpos=True)
    observation_omit_size = 2
  elif env_name == 'point_mass':
    env = point_mass.PointMassEnv(expose_goal=False, expose_velocity=False)
    observation_omit_size = 2
  elif env_name == 'DClaw':
    env = dclaw.DClawTurnRandom()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DClaw_randomized':
    env = dclaw.DClawTurnRandomDynamics()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_redesign':
    env = dkitty_redesign.BaseDKittyWalk(
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_randomized':
    env = dkitty_redesign.DKittyRandomDynamics(
        randomize_hfield=FLAGS.randomize_hfield,
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'HandBlock':
    observation_omit_size = 0
    env = hand_block.HandBlockCustomEnv(
        horizontal_wrist_constraint=FLAGS.horizontal_wrist_constraint,
        vertical_wrist_constraint=FLAGS.vertical_wrist_constraint,
        randomize_initial_position=bool(FLAGS.randomized_initial_distribution),
        randomize_initial_rotation=bool(FLAGS.randomized_initial_distribution))
  else:
    # note this is already wrapped, no need to wrap again
    env = suite_mujoco.load(env_name)
  return env

def hide_coords(time_step):
  global observation_omit_size
  if observation_omit_size > 0:
    sans_coords = time_step.observation[observation_omit_size:]
    return time_step._replace(observation=sans_coords)

  return time_step


def relabel_skill(trajectory_sample,
                  relabel_type=None,
                  cur_policy=None,
                  cur_skill_dynamics=None):
  global observation_omit_size
  if relabel_type is None or ('importance_sampling' in relabel_type and
                              FLAGS.is_clip_eps <= 1.0):
    return trajectory_sample, None

  # trajectory.to_transition, but for numpy arrays
  next_trajectory = nest.map_structure(lambda x: x[:, 1:], trajectory_sample)
  trajectory = nest.map_structure(lambda x: x[:, :-1], trajectory_sample)
  action_steps = policy_step.PolicyStep(
      action=trajectory.action, state=(), info=trajectory.policy_info)
  time_steps = ts.TimeStep(
      trajectory.step_type,
      reward=nest.map_structure(np.zeros_like, trajectory.reward),  # unknown
      discount=np.zeros_like(trajectory.discount),  # unknown
      observation=trajectory.observation)
  next_time_steps = ts.TimeStep(
      step_type=trajectory.next_step_type,
      reward=trajectory.reward,
      discount=trajectory.discount,
      observation=next_trajectory.observation)
  time_steps, action_steps, next_time_steps = nest.map_structure(
      lambda t: np.squeeze(t, axis=1),
      (time_steps, action_steps, next_time_steps))

  # just return the importance sampling weights for the given batch
  if 'importance_sampling' in relabel_type:
    old_log_probs = policy_step.get_log_probability(action_steps.info)
    is_weights = []
    for idx in range(time_steps.observation.shape[0]):
      cur_time_step = nest.map_structure(lambda x: x[idx:idx + 1], time_steps)
      cur_time_step = cur_time_step._replace(
          observation=cur_time_step.observation[:, observation_omit_size:])
      old_log_prob = old_log_probs[idx]
      cur_log_prob = cur_policy.log_prob(cur_time_step,
                                         action_steps.action[idx:idx + 1])[0]
      is_weights.append(
          np.clip(
              np.exp(cur_log_prob - old_log_prob), 1. / FLAGS.is_clip_eps,
              FLAGS.is_clip_eps))

    is_weights = np.array(is_weights)
    if relabel_type == 'normalized_importance_sampling':
      is_weights = is_weights / is_weights.mean()

    return trajectory_sample, is_weights

  new_observation = np.zeros(time_steps.observation.shape)
  for idx in range(time_steps.observation.shape[0]):
    alt_time_steps = nest.map_structure(
        lambda t: np.stack([t[idx]] * FLAGS.num_samples_for_relabelling),
        time_steps)

    # sample possible skills for relabelling from the prior
    if FLAGS.skill_type == 'cont_uniform':
      # always ensure that the original skill is one of the possible option for relabelling skills
      alt_skills = np.concatenate([
          np.random.uniform(
              low=-1.0,
              high=1.0,
              size=(FLAGS.num_samples_for_relabelling - 1, FLAGS.num_skills)),
          alt_time_steps.observation[:1, -FLAGS.num_skills:]
      ])

    # choose the skill which gives the highest log-probability to the current action
    if relabel_type == 'policy':
      cur_action = np.stack([action_steps.action[idx, :]] *
                            FLAGS.num_samples_for_relabelling)
      alt_time_steps = alt_time_steps._replace(
          observation=np.concatenate([
              alt_time_steps
              .observation[:,
                           observation_omit_size:-FLAGS.num_skills], alt_skills
          ],
                                     axis=1))
      action_log_probs = cur_policy.log_prob(alt_time_steps, cur_action)
      if FLAGS.debug_skill_relabelling:
        print('\n action_log_probs analysis----', idx,
              time_steps.observation[idx, -FLAGS.num_skills:])
        print('number of skills with higher log-probs:',
              np.sum(action_log_probs >= action_log_probs[-1]))
        print('Skills with log-probs higher than actual skill:')
        skill_dist = []
        for skill_idx in range(FLAGS.num_samples_for_relabelling):
          if action_log_probs[skill_idx] >= action_log_probs[-1]:
            print(alt_skills[skill_idx])
            skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
        print('average distance of skills with higher-log-prob:',
              np.mean(skill_dist))
      max_skill_idx = np.argmax(action_log_probs)

    # choose the skill which gets the highest log-probability under the dynamics posterior
    elif relabel_type == 'dynamics_posterior':
      cur_observations = alt_time_steps.observation[:, :-FLAGS.num_skills]
      next_observations = np.stack(
          [next_time_steps.observation[idx, :-FLAGS.num_skills]] *
          FLAGS.num_samples_for_relabelling)

      # max over posterior log probability is exactly the max over log-prob of transitin under skill-dynamics
      posterior_log_probs = cur_skill_dynamics.get_log_prob(
          process_observation(cur_observations), alt_skills,
          process_observation(next_observations))
      if FLAGS.debug_skill_relabelling:
        print('\n dynamics_log_probs analysis----', idx,
              time_steps.observation[idx, -FLAGS.num_skills:])
        print('number of skills with higher log-probs:',
              np.sum(posterior_log_probs >= posterior_log_probs[-1]))
        print('Skills with log-probs higher than actual skill:')
        skill_dist = []
        for skill_idx in range(FLAGS.num_samples_for_relabelling):
          if posterior_log_probs[skill_idx] >= posterior_log_probs[-1]:
            print(alt_skills[skill_idx])
            skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
        print('average distance of skills with higher-log-prob:',
              np.mean(skill_dist))

      max_skill_idx = np.argmax(posterior_log_probs)

    # make the new observation with the relabelled skill
    relabelled_skill = alt_skills[max_skill_idx]
    new_observation[idx] = np.concatenate(
        [time_steps.observation[idx, :-FLAGS.num_skills], relabelled_skill])

  traj_observation = np.copy(trajectory_sample.observation)
  traj_observation[:, 0] = new_observation
  new_trajectory_sample = trajectory_sample._replace(
      observation=traj_observation)

  return new_trajectory_sample, None


# hard-coding the state-space for dynamics
def process_observation(observation):

  def _shape_based_observation_processing(observation, dim_idx):
    if len(observation.shape) == 1:
      return observation[dim_idx:dim_idx + 1]
    elif len(observation.shape) == 2:
      return observation[:, dim_idx:dim_idx + 1]
    elif len(observation.shape) == 3:
      return observation[:, :, dim_idx:dim_idx + 1]

  # for consistent use
  if FLAGS.reduced_observation == 0:
    return observation

  # process observation for dynamics with reduced observation space
  if FLAGS.environment == 'HalfCheetah-v1':
    qpos_dim = 9
  elif FLAGS.environment == 'Ant-v1':
    qpos_dim = 15
  elif FLAGS.environment == 'Humanoid-v1':
    qpos_dim = 26
  elif 'DKitty' in FLAGS.environment:
    qpos_dim = 36

  # x-axis
  if FLAGS.reduced_observation in [1, 5]:
    red_obs = [_shape_based_observation_processing(observation, 0)]
  # x-y plane
  elif FLAGS.reduced_observation in [2, 6]:
    if FLAGS.environment == 'Ant-v1' or 'DKitty' in FLAGS.environment or 'DClaw' in FLAGS.environment:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1)
      ]
    else:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, qpos_dim)
      ]
  # x-y plane, x-y velocities
  elif FLAGS.reduced_observation in [4, 8]:
    if FLAGS.reduced_observation == 4 and 'DKittyPush' in FLAGS.environment:
      # position of the agent + relative position of the box
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, 3),
          _shape_based_observation_processing(observation, 4)
      ]
    elif FLAGS.environment in ['Ant-v1']:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, qpos_dim),
          _shape_based_observation_processing(observation, qpos_dim + 1)
      ]

  # (x, y, orientation), works only for ant, point_mass
  elif FLAGS.reduced_observation == 3:
    if FLAGS.environment in ['Ant-v1', 'point_mass']:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation,
                                              observation.shape[1] - 1)
      ]
    # x, y, z of the center of the block
    elif FLAGS.environment in ['HandBlock']:
      red_obs = [
          _shape_based_observation_processing(observation, 
                                              observation.shape[-1] - 7),
          _shape_based_observation_processing(observation, 
                                              observation.shape[-1] - 6),
          _shape_based_observation_processing(observation,
                                              observation.shape[-1] - 5)
      ]

  if FLAGS.reduced_observation in [5, 6, 8]:
    red_obs += [
        _shape_based_observation_processing(observation,
                                            observation.shape[1] - idx)
        for idx in range(1, 5)
    ]

  if FLAGS.reduced_observation == 36 and 'DKitty' in FLAGS.environment:
    red_obs = [
        _shape_based_observation_processing(observation, idx)
        for idx in range(qpos_dim)
    ]

  # x, y, z and the rotation quaternion
  if FLAGS.reduced_observation == 7 and FLAGS.environment == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(observation, observation.shape[-1] - idx)
        for idx in range(1, 8)
    ][::-1]

  # the rotation quaternion
  if FLAGS.reduced_observation == 4 and FLAGS.environment == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(observation, observation.shape[-1] - idx)
        for idx in range(1, 5)
    ][::-1]

  if isinstance(observation, np.ndarray):
    input_obs = np.concatenate(red_obs, axis=len(observation.shape) - 1)
  elif isinstance(observation, tf.Tensor):
    input_obs = tf.concat(red_obs, axis=len(observation.shape) - 1)
  return input_obs


def collect_experience(py_env,
                       time_step,
                       collect_policy,
                       buffer_list,
                       num_steps=1):

  episode_sizes = []
  extrinsic_reward = []
  step_idx = 0
  cur_return = 0.
  for step_idx in range(num_steps):
    if time_step.is_last():
      episode_sizes.append(step_idx)
      extrinsic_reward.append(cur_return)
      cur_return = 0.

    action_step = collect_policy.action(hide_coords(time_step))

    if FLAGS.action_clipping < 1.:
      action_step = action_step._replace(
          action=np.clip(action_step.action, -FLAGS.action_clipping,
                         FLAGS.action_clipping))

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      cur_action_log_prob = collect_policy.log_prob(
          nest_utils.batch_nested_array(hide_coords(time_step)),
          np.expand_dims(action_step.action, 0))
      action_step = action_step._replace(
          info=policy_step.set_log_probability(action_step.info,
                                               cur_action_log_prob))

    next_time_step = py_env.step(action_step.action)
    cur_return += next_time_step.reward

    # all modification to observations and training will be done within the agent
    for buffer_ in buffer_list:
      buffer_.add_batch(
          from_transition(
              nest_utils.batch_nested_array(time_step),
              nest_utils.batch_nested_array(action_step),
              nest_utils.batch_nested_array(next_time_step)))

    time_step = next_time_step

  # carry-over calculation for the next collection cycle
  episode_sizes.append(step_idx + 1)
  extrinsic_reward.append(cur_return)
  for idx in range(1, len(episode_sizes)):
    episode_sizes[-idx] -= episode_sizes[-idx - 1]

  return time_step, {
      'episode_sizes': episode_sizes,
      'episode_return': extrinsic_reward
  }


def run_on_env(env,
               policy,
               dynamics=None,
               predict_trajectory_steps=0,
               return_data=False,
               close_environment=True):
  time_step = env.reset()
  data = []

  if not return_data:
    extrinsic_reward = []
  while not time_step.is_last():
    action_step = policy.action(hide_coords(time_step))
    if FLAGS.action_clipping < 1.:
      action_step = action_step._replace(
          action=np.clip(action_step.action, -FLAGS.action_clipping,
                         FLAGS.action_clipping))

    env_action = action_step.action
    next_time_step = env.step(env_action)

    skill_size = FLAGS.num_skills
    if skill_size > 0:
      cur_observation = time_step.observation[:-skill_size]
      cur_skill = time_step.observation[-skill_size:]
      next_observation = next_time_step.observation[:-skill_size]
    else:
      cur_observation = time_step.observation
      next_observation = next_time_step.observation

    if dynamics is not None:
      if FLAGS.reduced_observation:
        cur_observation, next_observation = process_observation(
            cur_observation), process_observation(next_observation)
      logp = dynamics.get_log_prob(
          np.expand_dims(cur_observation, 0), np.expand_dims(cur_skill, 0),
          np.expand_dims(next_observation, 0))

      cur_predicted_state = np.expand_dims(cur_observation, 0)
      skill_expanded = np.expand_dims(cur_skill, 0)
      cur_predicted_trajectory = [cur_predicted_state[0]]
      for _ in range(predict_trajectory_steps):
        next_predicted_state = dynamics.predict_state(cur_predicted_state,
                                                      skill_expanded)
        cur_predicted_trajectory.append(next_predicted_state[0])
        cur_predicted_state = next_predicted_state
    else:
      logp = ()
      cur_predicted_trajectory = []

    if return_data:
      data.append([
          cur_observation, action_step.action, logp, next_time_step.reward,
          np.array(cur_predicted_trajectory)
      ])
    else:
      extrinsic_reward.append([next_time_step.reward])

    time_step = next_time_step

  if close_environment:
    env.close()

  if return_data:
    return data
  else:
    return extrinsic_reward


def eval_loop(eval_dir,
              eval_policy,
              dynamics=None,
              vid_name=None,
              plot_name=None):
  metadata = tf.gfile.Open(
      os.path.join(eval_dir, 'metadata.txt'), 'a')
  if FLAGS.num_skills == 0:
    num_evals = FLAGS.num_evals
  elif FLAGS.deterministic_eval:
    num_evals = FLAGS.num_skills
  else:
    num_evals = FLAGS.num_evals

  if plot_name is not None:
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_map = ['b', 'g', 'r', 'c', 'm', 'y']
    style_map = []
    for line_style in ['-', '--', '-.', ':']:
      style_map += [color + line_style for color in color_map]

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    # all_trajectories = []
    # all_predicted_trajectories = []

  for idx in range(num_evals):
    if FLAGS.num_skills > 0:
      if FLAGS.deterministic_eval:
        preset_skill = np.zeros(FLAGS.num_skills, dtype=np.int64)
        preset_skill[idx] = 1
      elif FLAGS.skill_type == 'discrete_uniform':
        preset_skill = np.random.multinomial(1, [1. / FLAGS.num_skills] *
                                             FLAGS.num_skills)
      elif FLAGS.skill_type == 'gaussian':
        preset_skill = np.random.multivariate_normal(
            np.zeros(FLAGS.num_skills), np.eye(FLAGS.num_skills))
      elif FLAGS.skill_type == 'cont_uniform':
        preset_skill = np.random.uniform(
            low=-1.0, high=1.0, size=FLAGS.num_skills)
      elif FLAGS.skill_type == 'multivariate_bernoulli':
        preset_skill = np.random.binomial(1, 0.5, size=FLAGS.num_skills)
    else:
      preset_skill = None

    eval_env = get_environment(env_name=FLAGS.environment)
    eval_env = wrap_env(
        skill_wrapper.SkillWrapper(
            eval_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=preset_skill,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps)

    # record videos for sampled trajectories
    if vid_name is not None:
      full_vid_name = vid_name + '_' + str(idx)
      eval_env = video_wrapper.VideoWrapper(eval_env, base_path=eval_dir, base_name=full_vid_name)

    mean_reward = 0.
    per_skill_evaluations = 1
    predict_trajectory_steps = 10
    # trajectories_per_skill = []
    # predicted_trajectories_per_skill = []
    for eval_idx in range(per_skill_evaluations):
      eval_trajectory = run_on_env(
          eval_env,
          eval_policy,
          dynamics=dynamics,
          predict_trajectory_steps=predict_trajectory_steps,
          return_data=True,
          close_environment=True if eval_idx == per_skill_evaluations -
          1 else False)

      trajectory_coordinates = np.array([
          eval_trajectory[step_idx][0][:2]
          for step_idx in range(len(eval_trajectory))
      ])

      # trajectory_states = np.array([
      #     eval_trajectory[step_idx][0]
      #     for step_idx in range(len(eval_trajectory))
      # ])
      # trajectories_per_skill.append(trajectory_states)
      if plot_name is not None:
        plt.plot(
            trajectory_coordinates[:, 0],
            trajectory_coordinates[:, 1],
            style_map[idx % len(style_map)],
            label=(str(idx) if eval_idx == 0 else None))
        # plt.plot(
        #     trajectory_coordinates[0, 0],
        #     trajectory_coordinates[0, 1],
        #     marker='o',
        #     color=style_map[idx % len(style_map)][0])
        if predict_trajectory_steps > 0:
          # predicted_states = np.array([
          #     eval_trajectory[step_idx][-1]
          #     for step_idx in range(len(eval_trajectory))
          # ])
          # predicted_trajectories_per_skill.append(predicted_states)
          for step_idx in range(len(eval_trajectory)):
            if step_idx % 20 == 0:
              plt.plot(eval_trajectory[step_idx][-1][:, 0],
                       eval_trajectory[step_idx][-1][:, 1], 'k:')

      mean_reward += np.mean([
          eval_trajectory[step_idx][-1]
          for step_idx in range(len(eval_trajectory))
      ])
      metadata.write(
          str(idx) + ' ' + str(preset_skill) + ' ' +
          str(trajectory_coordinates[-1, :]) + '\n')

    # all_predicted_trajectories.append(
    #     np.stack(predicted_trajectories_per_skill))
    # all_trajectories.append(np.stack(trajectories_per_skill))

  # all_predicted_trajectories = np.stack(all_predicted_trajectories)
  # all_trajectories = np.stack(all_trajectories)
  # print(all_trajectories.shape, all_predicted_trajectories.shape)
  # pkl.dump(
  #     all_trajectories,
  #     tf.gfile.GFile(
  #         os.path.join(vid_dir, 'skill_dynamics_full_obs_r100_actual_trajectories.pkl'),
  #         'wb'))
  # pkl.dump(
  #     all_predicted_trajectories,
  #     tf.gfile.GFile(
  #         os.path.join(vid_dir, 'skill_dynamics_full_obs_r100_predicted_trajectories.pkl'),
  #         'wb'))
  if plot_name is not None:
    full_image_name = plot_name + '.png'

    # to save images while writing to CNS
    buf = io.BytesIO()
    # plt.title('Trajectories in Continuous Skill Space')
    plt.savefig(buf, dpi=600, bbox_inches='tight')
    buf.seek(0)
    image = tf.gfile.Open(os.path.join(eval_dir, full_image_name), 'w')
    image.write(buf.read(-1))

    # clear before next plot
    plt.clf()


# discrete primitives only, useful with skill-dynamics
def eval_planning(env,
                  dynamics,
                  policy,
                  latent_action_space_size,
                  episode_horizon,
                  planning_horizon=1,
                  primitive_horizon=10,
                  **kwargs):
  """env: tf-agents environment without the skill wrapper."""
  global goal_coord

  # assuming only discrete action spaces
  high_level_action_space = np.eye(latent_action_space_size)
  time_step = env.reset()

  actual_reward = 0.
  actual_coords = [np.expand_dims(time_step.observation[:2], 0)]
  predicted_coords = []

  # planning loop
  for _ in range(episode_horizon // primitive_horizon):
    running_reward = np.zeros(latent_action_space_size)
    running_cur_state = np.array([process_observation(time_step.observation)] *
                                 latent_action_space_size)
    cur_coord_predicted = [np.expand_dims(running_cur_state[:, :2], 1)]

    # simulate all high level actions for K steps
    for _ in range(planning_horizon):
      predicted_next_state = dynamics.predict_state(running_cur_state,
                                                    high_level_action_space)
      cur_coord_predicted.append(np.expand_dims(predicted_next_state[:, :2], 1))

      # update running stuff
      running_reward += env.compute_reward(running_cur_state,
                                           predicted_next_state)
      running_cur_state = predicted_next_state

    predicted_coords.append(np.concatenate(cur_coord_predicted, axis=1))

    selected_high_level_action = np.argmax(running_reward)
    for _ in range(primitive_horizon):
      # concatenated observation
      skill_concat_observation = np.concatenate([
          time_step.observation,
          high_level_action_space[selected_high_level_action]
      ],
                                                axis=0)
      next_time_step = env.step(
          np.clip(
              policy.action(
                  hide_coords(
                      time_step._replace(
                          observation=skill_concat_observation))).action,
              -FLAGS.action_clipping, FLAGS.action_clipping))
      actual_reward += next_time_step.reward

      # prepare for next iteration
      time_step = next_time_step
      actual_coords.append(np.expand_dims(time_step.observation[:2], 0))

  actual_coords = np.concatenate(actual_coords)
  return actual_reward, actual_coords, predicted_coords


def eval_mppi(
    env,
    dynamics,
    policy,
    latent_action_space_size,
    episode_horizon,
    planning_horizon=1,
    primitive_horizon=10,
    num_candidate_sequences=50,
    refine_steps=10,
    mppi_gamma=10,
    prior_type='normal',
    smoothing_beta=0.9,
    # no need to change generally
    sparsify_rewards=False,
    # only for uniform prior mode
    top_primitives=5):
  """env: tf-agents environment without the skill wrapper.

     dynamics: skill-dynamics model learnt by DADS.
     policy: skill-conditioned policy learnt by DADS.
     planning_horizon: number of latent skills to plan in the future.
     primitive_horizon: number of steps each skill is executed for.
     num_candidate_sequences: number of samples executed from the prior per
     refining step of planning.
     refine_steps: number of steps for which the plan is iterated upon before
     execution (number of optimization steps).
     mppi_gamma: MPPI parameter for reweighing rewards.
     prior_type: 'normal' implies MPPI, 'uniform' implies a CEM like algorithm
     (not tested).
     smoothing_beta: for planning_horizon > 1, the every sampled plan is
     smoothed using EMA. (0-> no smoothing, 1-> perfectly smoothed)
     sparsify_rewards: converts a dense reward problem into a sparse reward
     (avoid using).
     top_primitives: number of elites to choose, if using CEM (not tested).
  """

  latent_action_space_size = FLAGS.num_skills
  future_steps = 1  # ensure primitive horizon is a multiple of future_steps
  episode_horizon = FLAGS.max_env_steps
  step_idx = 0

  def _smooth_primitive_sequences(primitive_sequences):
    for planning_idx in range(1, primitive_sequences.shape[1]):
      primitive_sequences[:,
                          planning_idx, :] = smoothing_beta * primitive_sequences[:, planning_idx - 1, :] + (
                              1. - smoothing_beta
                          ) * primitive_sequences[:, planning_idx, :]

    return primitive_sequences

  def _get_init_primitive_parameters():
    if prior_type == 'normal':
      prior_mean = functools.partial(
          np.random.multivariate_normal,
          mean=np.zeros(latent_action_space_size),
          cov=np.diag(np.ones(latent_action_space_size)))
      prior_cov = lambda: 1.5 * np.diag(np.ones(latent_action_space_size))
      return [prior_mean(), prior_cov()]

    elif prior_type == 'uniform':
      prior_low = lambda: np.array([-1.] * latent_action_space_size)
      prior_high = lambda: np.array([1.] * latent_action_space_size)
      return [prior_low(), prior_high()]

  def _sample_primitives(params):
    if prior_type == 'normal':
      sample = np.random.multivariate_normal(*params)
    elif prior_type == 'uniform':
      sample = np.random.uniform(*params)
    return np.clip(sample, -1., 1.)

  # update new primitive means for horizon sequence
  def _update_parameters(candidates, reward, primitive_parameters):
    # a more regular mppi
    if prior_type == 'normal':
      reward = np.exp(mppi_gamma * (reward - np.max(reward)))
      reward = reward / (reward.sum() + 1e-10)
      new_means = (candidates.T * reward).T.sum(axis=0)

      for planning_idx in range(candidates.shape[1]):
        primitive_parameters[planning_idx][0] = new_means[planning_idx]

    # TODO(architsh): closer to cross-entropy/shooting method, figure out a better update
    elif prior_type == 'uniform':
      chosen_candidates = candidates[np.argsort(reward)[-top_primitives:]]
      candidates_min = np.min(chosen_candidates, axis=0)
      candidates_max = np.max(chosen_candidates, axis=0)

      for planning_idx in range(candidates.shape[1]):
        primitive_parameters[planning_idx][0] = candidates_min[planning_idx]
        primitive_parameters[planning_idx][1] = candidates_max[planning_idx]

  def _get_expected_primitive(params):
    if prior_type == 'normal':
      return params[0]
    elif prior_type == 'uniform':
      return (params[0] + params[1]) / 2

  time_step = env.reset()
  actual_coords = [np.expand_dims(time_step.observation[:2], 0)]
  actual_reward = 0.
  distance_to_goal_array = []

  primitive_parameters = []
  chosen_primitives = []
  for _ in range(planning_horizon):
    primitive_parameters.append(_get_init_primitive_parameters())

  for _ in range(episode_horizon // primitive_horizon):
    for _ in range(refine_steps):
      # generate candidates sequences for primitives
      candidate_primitive_sequences = []
      for _ in range(num_candidate_sequences):
        candidate_primitive_sequences.append([
            _sample_primitives(primitive_parameters[planning_idx])
            for planning_idx in range(planning_horizon)
        ])

      candidate_primitive_sequences = np.array(candidate_primitive_sequences)
      candidate_primitive_sequences = _smooth_primitive_sequences(
          candidate_primitive_sequences)

      running_cur_state = np.array(
          [process_observation(time_step.observation)] *
          num_candidate_sequences)
      running_reward = np.zeros(num_candidate_sequences)
      for planning_idx in range(planning_horizon):
        cur_primitives = candidate_primitive_sequences[:, planning_idx, :]
        for _ in range(primitive_horizon // future_steps):
          predicted_next_state = dynamics.predict_state(running_cur_state,
                                                        cur_primitives)

          # update running stuff
          dense_reward = env.compute_reward(running_cur_state,
                                            predicted_next_state)
          # modification for sparse_reward
          if sparsify_rewards:
            sparse_reward = 5.0 * (dense_reward > -2) + 0.0 * (
                dense_reward <= -2)
            running_reward += sparse_reward
          else:
            running_reward += dense_reward

          running_cur_state = predicted_next_state

      _update_parameters(candidate_primitive_sequences, running_reward,
                         primitive_parameters)

    chosen_primitive = _get_expected_primitive(primitive_parameters[0])
    chosen_primitives.append(chosen_primitive)

    # a loop just to check what the chosen primitive is expected to do
    # running_cur_state = np.array([process_observation(time_step.observation)])
    # for _ in range(primitive_horizon // future_steps):
    #   predicted_next_state = dynamics.predict_state(
    #       running_cur_state, np.expand_dims(chosen_primitive, 0))
    #   running_cur_state = predicted_next_state
    # print('Predicted next co-ordinates:', running_cur_state[0, :2])

    for _ in range(primitive_horizon):
      # concatenated observation
      skill_concat_observation = np.concatenate(
          [time_step.observation, chosen_primitive], axis=0)
      next_time_step = env.step(
          np.clip(
              policy.action(
                  hide_coords(
                      time_step._replace(
                          observation=skill_concat_observation))).action,
              -FLAGS.action_clipping, FLAGS.action_clipping))
      actual_reward += next_time_step.reward
      distance_to_goal_array.append(next_time_step.reward)
      # prepare for next iteration
      time_step = next_time_step
      actual_coords.append(np.expand_dims(time_step.observation[:2], 0))
      step_idx += 1
      # print(step_idx)
    # print('Actual next co-ordinates:', actual_coords[-1])

    primitive_parameters.pop(0)
    primitive_parameters.append(_get_init_primitive_parameters())

  actual_coords = np.concatenate(actual_coords)
  return actual_reward, actual_coords, np.array(
      chosen_primitives), distance_to_goal_array


def main(_):
  # setting up
  start_time = time.time()
  tf.compat.v1.enable_resource_variables()
  logging.set_verbosity(logging.INFO)
  global observation_omit_size, goal_coord, sample_count, iter_count

  root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
  if not tf.gfile.Exists(root_dir):
    tf.gfile.MakeDirs(root_dir)
  log_dir = os.path.join(root_dir, FLAGS.environment)
  
  if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)
  save_dir = os.path.join(log_dir, 'models')
  if not tf.gfile.Exists(save_dir):
    tf.gfile.MakeDirs(save_dir)

  print('directory for recording experiment data:', log_dir)

  # in case training is paused and resumed, so can be restored
  try:
    sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
    iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
    episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
    episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
  except:
    sample_count = 0
    iter_count = 0
    episode_size_buffer = []
    episode_return_buffer = []

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(log_dir, 'train', 'in_graph_data'), flush_millis=10 * 1000)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(True):
    # environment related stuff
    py_env = get_environment(env_name=FLAGS.environment)
    py_env = wrap_env(
        skill_wrapper.SkillWrapper(
            py_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=None,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps)

    # all specifications required for all networks and agents
    py_action_spec = py_env.action_spec()
    tf_action_spec = tensor_spec.from_spec(
        py_action_spec)  # policy, critic action spec
    env_obs_spec = py_env.observation_spec()
    py_env_time_step_spec = ts.time_step_spec(
        env_obs_spec)  # replay buffer time_step spec
    if observation_omit_size > 0:
      agent_obs_spec = array_spec.BoundedArraySpec(
          (env_obs_spec.shape[0] - observation_omit_size,),
          env_obs_spec.dtype,
          minimum=env_obs_spec.minimum,
          maximum=env_obs_spec.maximum,
          name=env_obs_spec.name)  # policy, critic observation spec
    else:
      agent_obs_spec = env_obs_spec
    py_agent_time_step_spec = ts.time_step_spec(
        agent_obs_spec)  # policy, critic time_step spec
    tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

    if not FLAGS.reduced_observation:
      skill_dynamics_observation_size = (
          py_env_time_step_spec.observation.shape[0] - FLAGS.num_skills)
    else:
      skill_dynamics_observation_size = FLAGS.reduced_observation

    # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_agent_time_step_spec.observation,
        tf_action_spec,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
        (tf_agent_time_step_spec.observation, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      reweigh_batches_flag = True
    else:
      reweigh_batches_flag = False

    agent = dads_agent.DADSAgent(
        # DADS parameters
        save_dir,
        skill_dynamics_observation_size,
        observation_modify_fn=process_observation,
        restrict_input_size=observation_omit_size,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        normalize_observations=FLAGS.normalize_data,
        network_type=FLAGS.graph_type,
        num_mixture_components=FLAGS.num_components,
        fix_variance=FLAGS.fix_variance,
        reweigh_batches=reweigh_batches_flag,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
        # SAC parameters
        time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. /
        (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)

    # evaluation policy
    eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

    # collection policy
    if FLAGS.collect_policy == 'default':
      collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
    elif FLAGS.collect_policy == 'ou_noise':
      collect_policy = py_tf_policy.PyTFPolicy(
          ou_noise_policy.OUNoisePolicy(
              agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))

    # relabelling policy deals with batches of data, unlike collect and eval
    relabel_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)

    # constructing a replay buffer, need a python spec
    policy_step_spec = policy_step.PolicyStep(
        action=py_action_spec, state=(), info=())

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      policy_step_spec = policy_step_spec._replace(
          info=policy_step.set_log_probability(
              policy_step_spec.info,
              array_spec.ArraySpec(
                  shape=(), dtype=np.float32, name='action_log_prob')))

    trajectory_spec = from_transition(py_env_time_step_spec, policy_step_spec,
                                      py_env_time_step_spec)
    capacity = FLAGS.replay_buffer_capacity
    # for all the data collected
    rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        capacity=capacity, data_spec=trajectory_spec)

    if FLAGS.train_skill_dynamics_on_policy:
      # for on-policy data (if something special is required)
      on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
          capacity=FLAGS.initial_collect_steps + FLAGS.collect_steps + 10,
          data_spec=trajectory_spec)

    # insert experience manually with relabelled rewards and skills
    agent.build_agent_graph()
    agent.build_skill_dynamics_graph()
    agent.create_savers()

    # saving this way requires the saver to be out the object
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'agent'),
        agent=agent,
        global_step=global_step)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'policy'),
        policy=agent.policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=rbuffer)

    setup_time = time.time() - start_time
    print('Setup time:', setup_time)

    with tf.compat.v1.Session().as_default() as sess:
      train_checkpointer.initialize_or_restore(sess)
      rb_checkpointer.initialize_or_restore(sess)
      agent.set_sessions(
          initialize_or_restore_skill_dynamics=True, session=sess)

      meta_start_time = time.time()
      if FLAGS.run_train:

        train_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        common.initialize_uninitialized_variables(sess)
        sess.run(train_summary_writer.init())

        time_step = py_env.reset()
        episode_size_buffer.append(0)
        episode_return_buffer.append(0.)

        # maintain a buffer of episode lengths
        def _process_episodic_data(ep_buffer, cur_data):
          ep_buffer[-1] += cur_data[0]
          ep_buffer += cur_data[1:]

          # only keep the last 100 episodes
          if len(ep_buffer) > 101:
            ep_buffer = ep_buffer[-101:]

        # remove invalid transitions from the replay buffer
        def _filter_trajectories(trajectory):
          # two consecutive samples in the buffer might not have been consecutive in the episode
          valid_indices = (trajectory.step_type[:, 0] != 2)

          return nest.map_structure(lambda x: x[valid_indices], trajectory)

        if iter_count == 0:
          start_time = time.time()
          time_step, collect_info = collect_experience(
              py_env,
              time_step,
              collect_policy,
              buffer_list=[rbuffer] if not FLAGS.train_skill_dynamics_on_policy
              else [rbuffer, on_buffer],
              num_steps=FLAGS.initial_collect_steps)
          _process_episodic_data(episode_size_buffer,
                                 collect_info['episode_sizes'])
          _process_episodic_data(episode_return_buffer,
                                 collect_info['episode_return'])
          sample_count += FLAGS.initial_collect_steps
          initial_collect_time = time.time() - start_time
          print('Initial data collection time:', initial_collect_time)

        agent_end_train_time = time.time()
        while iter_count < FLAGS.num_epochs:
          print('iteration index:', iter_count)

          # model save
          if FLAGS.save_model is not None and iter_count % FLAGS.save_freq == 0:
            print('Saving stuff')
            train_checkpointer.save(global_step=iter_count)
            policy_checkpointer.save(global_step=iter_count)
            rb_checkpointer.save(global_step=iter_count)
            agent.save_variables(global_step=iter_count)

            np.save(os.path.join(log_dir, 'sample_count'), sample_count)
            np.save(os.path.join(log_dir, 'episode_size_buffer'), episode_size_buffer)
            np.save(os.path.join(log_dir, 'episode_return_buffer'), episode_return_buffer)
            np.save(os.path.join(log_dir, 'iter_count'), iter_count)

          collect_start_time = time.time()
          print('intermediate time:', collect_start_time - agent_end_train_time)
          time_step, collect_info = collect_experience(
              py_env,
              time_step,
              collect_policy,
              buffer_list=[rbuffer] if not FLAGS.train_skill_dynamics_on_policy
              else [rbuffer, on_buffer],
              num_steps=FLAGS.collect_steps)
          sample_count += FLAGS.collect_steps
          _process_episodic_data(episode_size_buffer,
                                 collect_info['episode_sizes'])
          _process_episodic_data(episode_return_buffer,
                                 collect_info['episode_return'])
          collect_end_time = time.time()
          print('Iter collection time:', collect_end_time - collect_start_time)

          # only for debugging skill relabelling
          if iter_count >= 1 and FLAGS.debug_skill_relabelling:
            trajectory_sample = rbuffer.get_next(
                sample_batch_size=5, num_steps=2)
            trajectory_sample = _filter_trajectories(trajectory_sample)
            # trajectory_sample, _ = relabel_skill(
            #     trajectory_sample,
            #     relabel_type='policy',
            #     cur_policy=relabel_policy,
            #     cur_skill_dynamics=agent.skill_dynamics)
            trajectory_sample, is_weights = relabel_skill(
                trajectory_sample,
                relabel_type='importance_sampling',
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics)
            print(is_weights)

          skill_dynamics_buffer = rbuffer
          if FLAGS.train_skill_dynamics_on_policy:
            skill_dynamics_buffer = on_buffer

          # TODO(architsh): clear_buffer_every_iter needs to fix these as well
          for _ in range(1 if FLAGS.clear_buffer_every_iter else FLAGS
                         .skill_dyn_train_steps):
            if FLAGS.clear_buffer_every_iter:
              trajectory_sample = rbuffer.gather_all_transitions()
            else:
              trajectory_sample = skill_dynamics_buffer.get_next(
                  sample_batch_size=FLAGS.skill_dyn_batch_size, num_steps=2)
            trajectory_sample = _filter_trajectories(trajectory_sample)

            # is_weights is None usually, unless relabelling involves importance_sampling
            trajectory_sample, is_weights = relabel_skill(
                trajectory_sample,
                relabel_type=FLAGS.skill_dynamics_relabel_type,
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics)
            input_obs = process_observation(
                trajectory_sample.observation[:, 0, :-FLAGS.num_skills])
            cur_skill = trajectory_sample.observation[:, 0, -FLAGS.num_skills:]
            target_obs = process_observation(
                trajectory_sample.observation[:, 1, :-FLAGS.num_skills])
            if FLAGS.clear_buffer_every_iter:
              agent.skill_dynamics.train(
                  input_obs,
                  cur_skill,
                  target_obs,
                  batch_size=FLAGS.skill_dyn_batch_size,
                  batch_weights=is_weights,
                  num_steps=FLAGS.skill_dyn_train_steps)
            else:
              agent.skill_dynamics.train(
                  input_obs,
                  cur_skill,
                  target_obs,
                  batch_size=-1,
                  batch_weights=is_weights,
                  num_steps=1)

          if FLAGS.train_skill_dynamics_on_policy:
            on_buffer.clear()

          skill_dynamics_end_train_time = time.time()
          print('skill_dynamics train time:',
                skill_dynamics_end_train_time - collect_end_time)

          running_dads_reward, running_logp, running_logp_altz = [], [], []

          # agent train loop analysis
          within_agent_train_time = time.time()
          sampling_time_arr, filtering_time_arr, relabelling_time_arr, train_time_arr = [], [], [], []
          for _ in range(
              1 if FLAGS.clear_buffer_every_iter else FLAGS.agent_train_steps):
            if FLAGS.clear_buffer_every_iter:
              trajectory_sample = rbuffer.gather_all_transitions()
            else:
              trajectory_sample = rbuffer.get_next(
                  sample_batch_size=FLAGS.agent_batch_size, num_steps=2)

            buffer_sampling_time = time.time()
            sampling_time_arr.append(buffer_sampling_time -
                                     within_agent_train_time)
            trajectory_sample = _filter_trajectories(trajectory_sample)

            filtering_time = time.time()
            filtering_time_arr.append(filtering_time - buffer_sampling_time)
            trajectory_sample, _ = relabel_skill(
                trajectory_sample,
                relabel_type=FLAGS.agent_relabel_type,
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics)
            relabelling_time = time.time()
            relabelling_time_arr.append(relabelling_time - filtering_time)

            # need to match the assert structure
            if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type:
              trajectory_sample = trajectory_sample._replace(policy_info=())

            if not FLAGS.clear_buffer_every_iter:
              dads_reward, info = agent.train_loop(
                  trajectory_sample,
                  recompute_reward=True,  # turn False for normal SAC training
                  batch_size=-1,
                  num_steps=1)
            else:
              dads_reward, info = agent.train_loop(
                  trajectory_sample,
                  recompute_reward=True,  # turn False for normal SAC training
                  batch_size=FLAGS.agent_batch_size,
                  num_steps=FLAGS.agent_train_steps)

            within_agent_train_time = time.time()
            train_time_arr.append(within_agent_train_time - relabelling_time)
            if dads_reward is not None:
              running_dads_reward.append(dads_reward)
              running_logp.append(info['logp'])
              running_logp_altz.append(info['logp_altz'])

          agent_end_train_time = time.time()
          print('agent train time:',
                agent_end_train_time - skill_dynamics_end_train_time)
          print('\t sampling time:', np.sum(sampling_time_arr))
          print('\t filtering_time:', np.sum(filtering_time_arr))
          print('\t relabelling time:', np.sum(relabelling_time_arr))
          print('\t train_time:', np.sum(train_time_arr))

          if len(episode_size_buffer) > 1:
            train_writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(
                        tag='episode_size',
                        simple_value=np.mean(episode_size_buffer[:-1]))
                ]), sample_count)
          if len(episode_return_buffer) > 1:
            train_writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(
                        tag='episode_return',
                        simple_value=np.mean(episode_return_buffer[:-1]))
                ]), sample_count)
          train_writer.add_summary(
              tf.Summary(value=[
                  tf.Summary.Value(
                      tag='dads/reward',
                      simple_value=np.mean(
                          np.concatenate(running_dads_reward)))
              ]), sample_count)

          train_writer.add_summary(
              tf.Summary(value=[
                  tf.Summary.Value(
                      tag='dads/logp',
                      simple_value=np.mean(np.concatenate(running_logp)))
              ]), sample_count)
          train_writer.add_summary(
              tf.Summary(value=[
                  tf.Summary.Value(
                      tag='dads/logp_altz',
                      simple_value=np.mean(np.concatenate(running_logp_altz)))
              ]), sample_count)

          if FLAGS.clear_buffer_every_iter:
            rbuffer.clear()
            time_step = py_env.reset()
            episode_size_buffer = [0]
            episode_return_buffer = [0.]

          # within train loop evaluation
          if FLAGS.record_freq is not None and iter_count % FLAGS.record_freq == 0:
            cur_vid_dir = os.path.join(log_dir, 'videos', str(iter_count))
            tf.gfile.MakeDirs(cur_vid_dir)
            eval_loop(
                cur_vid_dir,
                eval_policy,
                dynamics=agent.skill_dynamics,
                vid_name=FLAGS.vid_name,
                plot_name='traj_plot')

          iter_count += 1

        py_env.close()

        print('Final statistics:')
        print('\ttotal time for %d epochs: %f' %(FLAGS.num_epochs, time.time() - meta_start_time))
        print('\tsteps collected during this time: %d' %(rbuffer.size))

      # final evaluation, if any
      if FLAGS.run_eval:
        vid_dir = os.path.join(log_dir, 'videos', 'final_eval')
        if not tf.gfile.Exists(vid_dir):
          tf.gfile.MakeDirs(vid_dir)
        vid_name = FLAGS.vid_name

        # generic skill evaluation
        if FLAGS.deterministic_eval or FLAGS.num_evals > 0:
          eval_loop(
              vid_dir,
              eval_policy,
              dynamics=agent.skill_dynamics,
              vid_name=vid_name,
              plot_name='traj_plot')

        # for planning the evaluation directory is changed to save directory
        eval_dir = os.path.join(log_dir, 'eval')
        goal_coord_list = [
            np.array([10.0, 10.0]),
            np.array([-10.0, 10.0]),
            np.array([-10.0, -10.0]),
            np.array([10.0, -10.0]),
            np.array([0.0, -10.0]),
            np.array([5.0, 10.0])
        ]

        eval_dir = os.path.join(eval_dir, 'mpc_eval')
        if not tf.gfile.Exists(eval_dir):
          tf.gfile.MakeDirs(eval_dir)
        save_label = 'goal_'
        if 'discrete' in FLAGS.skill_type:
          planning_fn = eval_planning
          
        else:
          planning_fn = eval_mppi

        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        average_reward_all_goals = []
        _, ax1 = plt.subplots(1, 1)
        ax1.set_xlim(-20, 20)
        ax1.set_ylim(-20, 20)

        final_text = open(os.path.join(eval_dir, 'eval_data.txt'), 'w')

        # goal_list = []
        # for r in range(4, 50):
        #   for _ in range(10):
        #     theta = np.random.uniform(-np.pi, np.pi)
        #     goal_x = r * np.cos(theta)
        #     goal_y = r * np.sin(theta)
        #     goal_list.append([r, theta, goal_x, goal_y])

        # def _sample_goal():
        #   goal_coords = np.random.uniform(0, 5, size=2)
        #   # while np.linalg.norm(goal_coords) < np.linalg.norm([10., 10.]):
        #   #   goal_coords = np.random.uniform(-25, 25, size=2)
        #   return goal_coords

        # goal_coord_list = [_sample_goal() for _ in range(50)]

        for goal_idx, goal_coord in enumerate(goal_coord_list):
          # for goal_idx in range(1):
          print('Trying to reach the goal:', goal_coord)
          # eval_plan_env = video_wrapper.VideoWrapper(
          #     get_environment(env_name=FLAGS.environment + '_goal')
          #     base_path=eval_dir,
          #     base_name=save_label + '_' + str(goal_idx)))
          # goal_coord = np.array(item[2:])
          eval_plan_env = get_environment(env_name=FLAGS.environment + '_goal')
          # _, (ax1, ax2) = plt.subplots(1, 2)
          # ax1.set_xlim(-12, 12)
          # ax1.set_ylim(-12, 12)
          # ax2.set_xlim(-1, 1)
          # ax2.set_ylim(-1, 1)
          ax1.plot(goal_coord[0], goal_coord[1], marker='x', color='k')
          reward_list = []

          def _steps_to_goal(dist_array):
            for idx in range(len(dist_array)):
              if -dist_array[idx] < 1.5:
                return idx
            return -1

          for _ in range(1):
            reward, actual_coords, primitives, distance_to_goal_array = planning_fn(
                eval_plan_env, agent.skill_dynamics, eval_policy,
                latent_action_space_size=FLAGS.num_skills,
                episode_horizon=FLAGS.max_env_steps,
                planning_horizon=FLAGS.planning_horizon,
                primitive_horizon=FLAGS.primitive_horizon,
                num_candidate_sequences=FLAGS.num_candidate_sequences,
                refine_steps=FLAGS.refine_steps,
                mppi_gamma=FLAGS.mppi_gamma,
                prior_type=FLAGS.prior_type,
                smoothing_beta=FLAGS.smoothing_beta,
                top_primitives=FLAGS.top_primitives
            )
            reward /= (FLAGS.max_env_steps * np.linalg.norm(goal_coord))
            ax1.plot(
                actual_coords[:, 0],
                actual_coords[:, 1],
                color_map[goal_idx % len(color_map)],
                linewidth=1)
            # ax2.plot(
            #     primitives[:, 0],
            #     primitives[:, 1],
            #     marker='x',
            #     color=color_map[try_idx % len(color_map)],
            #     linewidth=1)
            final_text.write(','.join([
                str(item) for item in [
                    goal_coord[0],
                    goal_coord[1],
                    reward,
                    _steps_to_goal(distance_to_goal_array),
                    distance_to_goal_array[-3],
                    distance_to_goal_array[-2],
                    distance_to_goal_array[-1],
                ]
            ]) + '\n')
            print(reward)
            reward_list.append(reward)

          eval_plan_env.close()
          average_reward_all_goals.append(np.mean(reward_list))
          print('Average reward:', np.mean(reward_list))

        final_text.close()
        # to save images while writing to CNS
        buf = io.BytesIO()
        plt.savefig(buf, dpi=600, bbox_inches='tight')
        buf.seek(0)
        image = tf.gfile.Open(os.path.join(eval_dir, save_label + '.png'), 'w')
        image.write(buf.read(-1))
        plt.clf()

        # for iter_idx in range(1, actual_coords.shape[0]):
        #   _, ax1 = plt.subplots(1, 1)
        #   ax1.set_xlim(-2, 15)
        #   ax1.set_ylim(-2, 15)
        #   ax1.plot(
        #       actual_coords[:iter_idx, 0],
        #       actual_coords[:iter_idx, 1],
        #       linewidth=1.2)
        #   ax1.scatter(
        #       np.array(goal_coord_list)[:, 0],
        #       np.array(goal_coord_list)[:, 1],
        #       marker='x',
        #       color='k')
        #   buf = io.BytesIO()
        #   plt.savefig(buf, dpi=200, bbox_inches='tight')
        #   buf.seek(0)
        #   image = tf.gfile.Open(
        #       os.path.join(eval_dir,
        #                    save_label + '_' + '%04d' % (iter_idx) + '.png'),
        #       'w')
        #   image.write(buf.read(-1))
        #   plt.clf()

        plt.close()
        print('Average reward for all goals:', average_reward_all_goals)


if __name__ == '__main__':
  tf.compat.v1.app.run(main)
