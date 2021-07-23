#!/usr/bin/env python
import multiprocessing as mp
import argparse
import copy
import yaml
import gym
import time
import neptune
from itertools import product
from rltoolkit import TD3
import matplotlib.pyplot as plt
import numpy as np


def visualise_episode_vanilla(algo):
    rad_penalties = []
    vel_dir_penalties = []
    orbit_vel_penalties = []

    obs = algo.env.reset()
    algo.env.render()
    done = False
    ep_ret = 0
    i = 0
    while not done:
        obs = algo.process_obs(obs)
        obs = algo.replay_buffer.normalize(obs)
        action, _ = algo._actor.act(obs, deterministic=True)
        action = algo.process_action(action, obs)
        obs, r, done, _ = algo.env.step(action)
        rad_penalties.append(algo.env.rad_penalty)
        vel_dir_penalties.append(algo.env.vel_dir_penalty)
        orbit_vel_penalties.append(algo.env.orbit_vel_penalty)
        frame = algo.env.render(mode="rgb_array")
        # plt.imsave(f"gifs/{i:05d}.png", frame)
        ep_ret += r
        i += 1
    print(f"ep_ret={ep_ret}")
    rad_penalties = np.array(rad_penalties)
    vel_dir_penalties = np.array(vel_dir_penalties)
    orbit_vel_penalties = np.array(orbit_vel_penalties)

    print(
        f"rad_penalties, mean={np.mean(rad_penalties)}, std={np.std(rad_penalties)}, min={np.min(rad_penalties)}, max={np.max(rad_penalties)}"
    )
    print(
        f"vel_dir_penalties, mean={np.mean(vel_dir_penalties)}, std={np.std(vel_dir_penalties)}, min={np.min(vel_dir_penalties)}, max={np.max(vel_dir_penalties)}"
    )
    print(
        f"orbit_vel_penalties, mean={np.mean(orbit_vel_penalties)}, std={np.std(orbit_vel_penalties)}, min={np.min(orbit_vel_penalties)}, max={np.max(orbit_vel_penalties)}"
    )


gym.envs.register(
    id="Orbit-v0",
    entry_point="gym_space.envs.orbit:OrbitEnv",
    max_episode_steps=300,
)

gym.envs.register(
    id="GoalContinuous1-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    kwargs={"n_planets": 1},
)

gym.envs.register(
    id="GoalContinuous2-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    kwargs={"n_planets": 2},
)

gym.envs.register(
    id="Kepler-v0",
    entry_point="gym_space.envs.kepler:KeplerContinuousEnv",
    max_episode_steps=500,
)

# model_path = "models/Jun30_13-12-28.489Orbit-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
# ENV_NAME = "Orbit-v0"
# model_path = "models/Jul05_22-42-01.624GoalContinuous2-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
# ENV_NAME = "GoalContinuous2-v0"

# model_path = "models/Jul15_18-21-47.854Kepler500-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2_env_vt0.1_rt0.1.pkl"
model_path = "models/Jul19_14-45-25.034Kepler500-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2_dense_reward4.pkl"
ENV_NAME = "Kepler-v0"

td3 = TD3(
    env_name=ENV_NAME,
    obs_norm=False,
)

td3.load(model_path)

# ret = td3.test(5)
# print(ret)

visualise_episode_vanilla(td3)
