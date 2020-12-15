# Harry Chong
import gym
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PIL
import logging
import base64
import io
import IPython
import imageio
from tensorflow import keras

from gym.wrappers import TimeLimit
import tf_agents
from tf_agents.environments import suite_gym
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function

# Make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT_DIR = "."
env = suite_gym.load("Assault-v4")
env.seed(42)
env.reset();

# Environment Wrappers
# Wrapper to repeat same action n steps and accumlates the rewards (speed up training)
action_repeat_env = ActionRepeat(env, times=4)

# Wrapper to interrupt env if it runs longer than a max number of steps
time_limit_env = suite_gym.load("Assault-v4", 
    gym_env_wrappers = [lambda env: TimeLimit(env, max_episode_steps = 10000)], 
    env_wrappers=[lambda env: ActionRepeat(env, times = 4)]
)

# Atari Pre-Processing
max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "AssaultNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps = max_episode_steps,
    gym_env_wrappers = [AtariPreprocessing, FrameStack4]
)

tf_env = TFPyEnvironment(env)

# Demo Assault Env.
env.seed(42)
env.reset()

time_step = env.step(np.array(1)) # FIRE
for _ in range(4):
    time_step = env.step(np.array(3)) # LEFT

# Create Deep Q-Network with TF-Agents
preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(64, (8, 8), 4), (64, (3, 3), 2), (64, (3, 3), 1)]
fc_layer_params=[1024]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
    activation_fn=tf.keras.activations.relu)

# See TF-agents issue #113
#optimizer = keras.optimizers.RMSprop(lr = 2.5e-4, rho = 0.95, momentum = 0.0, epsilon = 0.00001, centered = True)
train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps

optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 1e-3, decay = 0.95, 
    momentum = 0.0, epsilon = 0.0001, centered = True)

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 1.0, # initial ?
    decay_steps = 25000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate = 0.01
) # final ?

agent = DqnAgent(tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=  q_net,
    optimizer = optimizer,
    target_update_period = 2000, # <=> 32,000 ALE frames
    td_errors_loss_fn = keras.losses.Huber(reduction="none"),
    gamma = 0.95, # discount factor
    train_step_counter = train_step,
    epsilon_greedy = lambda: epsilon_fn(train_step),
    reward_scale_factor = 1.5
)

agent.initialize()

# Create Replay Buffer and Observer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = tf_env.batch_size,
    max_length = 1000000)

replay_buffer_observer = replay_buffer.add_batch

# Custom observer that counts and displays the number of times it is called
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end = "")

# Create training metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

train_metrics[0].result()

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

# Create collect driver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration

# Collect the initial experiences, before training
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers = [replay_buffer.add_batch, ShowProgress(20000)],
    num_steps = 20000) # <=> 80,000 ALE frames

final_time_step, final_policy_state = init_driver.run()

tf.random.set_seed(888) # chosen to show an example of trajectory at the end of an episode

trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size=2, num_steps=3)

# trajectories._fields
# trajectories.observation.shape
time_steps, action_steps, next_time_steps = to_transition(trajectories)
time_steps.observation.shape
trajectories.step_type.numpy()

# Create dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# Establish checkpoint for later training
checkpoint_dir = os.path.join("lastModelCheckpoint")
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=agent)
manager = tf.train.CheckpointManager(checkpoint, './lastModelCheckpoint', max_to_keep=3)

# Convert the main functions to TF Functions for better performance
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
        
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
        if iteration % 5000 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

# Train the agent for 75,000 steps. Repeat as much as you want, the agent will keep improving.
# Load from latest checkpoint if possible.
train_agent(n_iterations=75000)

# Save frames for updated animated gif
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1))
        prev_lives = lives

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)

# Create updated animated gif of agent in action
image_path = os.path.join(PROJECT_ROOT_DIR, "myAgentPlays.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format = 'GIF',
                     append_images=frame_images[1:],
                     save_all = True,
                     duration = 30,
                     loop = 0)

# Save policy and model
policy_dir = os.path.join(PROJECT_ROOT_DIR, "savedPolicy")
tf_policy_saver = PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)
