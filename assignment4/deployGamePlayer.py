# Harry Chong
import gym
import numpy as np
import os
import tensorflow as tf
import logging
import base64
import io
import IPython
import imageio

from tensorflow import keras
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.policies.policy_saver import PolicySaver

# Helper Functions for video generation
def embed_gif(gif_buffer):
    """Embeds a gif file in the notebook."""
    tag = '<img src="data:image/gif;base64,{0}"/>'.format(base64.b64encode(gif_buffer).decode())
    return IPython.display.HTML(tag)

def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env):
    num_episodes = 3
    frames = []
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        frames.append(eval_py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            frames.append(eval_py_env.render())
    gif_file = io.BytesIO()
    imageio.mimsave(gif_file, frames, format='gif', fps=60)
    IPython.display.display(embed_gif(gif_file.getvalue()))

PROJECT_ROOT_DIR = "."
# Atari Pre-Processing
max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "AssaultNoFrameskip-v4"

env = suite_atari.load(    environment_name,
    max_episode_steps = max_episode_steps,
    gym_env_wrappers = [AtariPreprocessing, FrameStack4]
)

tf_env = TFPyEnvironment(env)

policy_dir = os.path.join(PROJECT_ROOT_DIR, "savedPolicy")
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# Run trained model 
run_episodes_and_create_video(saved_policy, tf_env, env)
