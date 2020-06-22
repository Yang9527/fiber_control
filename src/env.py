from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

class FiberEnv(py_environment.PyEnvironment):
  def __init__(self):
    self.GOOD_RADIUM = 1.22
    self.EPS = 15.0 * 1e-3
    # self._action_spec = array_spec.BoundedArraySpec(
    #     shape=(2,), dtype=np.float32, minimum=0.0, maximum=[20.0, 100.0], name='action')
    # self._observation_spec = array_spec.BoundedArraySpec(
    #     shape=(2,), dtype=np.float32, minimum=0.0, maximum=[20.0, 1000.0], name='observation')

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=-1000, maximum=1000, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.float32, minimum=0.0, maximum=10.0, name='observation')
    self._radium = 0.0
    self._temperature = 0.0 
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    pass

  def _step(self, action):
    pass


env = FiberEnv()
train_env = tf_py_environment.TFPyEnvironment(env)
print(train_env.observation_spec())
fc_layer_params = (100,50,1)
learning_rate = 1e-3

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
  

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()
print("==" * 10)
print(tf_agent.collect_data_spec)


replay_buffer_capacity = 1000
batch_size = 32

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=batch_size,
    max_length=replay_buffer_capacity)


def collect_episode(filename):
  from .load_data import load_data, experience_to_traj
  trajs = [experience_to_traj(exp) for exp in load_data(filename)]
  return trajs


tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
#avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#returns = [avg_return]
num_iterations = 10
log_interval = 1
experience = collect_episode(
      "data/玻璃棒拉丝数据.xlsx")
for traj in experience:

  # Collect a few episodes using collect_policy and save to the replay buffer.
  # experience = collect_episode(
  #     "data/玻璃棒拉丝数据.xlsx")

  # Use data from the buffer and update the agent's network.
  #experience = replay_buffer.gather_all()
  train_loss = tf_agent.train(traj)
  #replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  # if step % eval_interval == 0:
  #   avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
  #   print('step = {0}: Average Return = {1}'.format(step, avg_return))
  #   returns.append(avg_return)