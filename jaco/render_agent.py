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
    vel_x_penalties = []
    vel_y_penalties = []
    act_penalties = []

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
        vel_x_penalties.append(algo.env.vel_x_penalty)
        vel_y_penalties.append(algo.env.vel_y_penalty)
        act_penalties.append(algo.env.act_penalty)
        print(f"rad_penalty={algo.env.rad_penalty}")
        print(f"vel_x_penalty={algo.env.vel_x_penalty}")
        print(f"vel_y_penalty={algo.env.vel_y_penalty}")
        print(f"act_penalty={algo.env.act_penalty}")

        frame = algo.env.render(mode="rgb_array")
        # plt.imsave(f"gifs/{i:05d}.png", frame)
        ep_ret += r
        i += 1
    print(f"ep_ret={ep_ret}")
    rad_penalties = np.array(rad_penalties)
    vel_x_penalties = np.array(vel_x_penalties)
    vel_y_penalties = np.array(vel_y_penalties)

    print(
        f"rad_penalties, mean={np.mean(rad_penalties)}, median={np.median(rad_penalties)}, std={np.std(rad_penalties)}, min={np.min(rad_penalties)}"
    )
    print(
        f"vel_x_penalties, mean={np.mean(vel_x_penalties)}, median={np.median(vel_x_penalties)}, std={np.std(vel_x_penalties)}, min={np.min(vel_x_penalties)}"
    )
    print(
        f"vel_y_penalties, mean={np.mean(vel_y_penalties)}, median={np.median(vel_y_penalties)}, std={np.std(vel_y_penalties)}, min={np.min(vel_y_penalties)}"
    )
    print(
        f"act_penalties, mean={np.mean(act_penalties)}, median={np.median(act_penalties)}, std={np.std(act_penalties)}, min={np.min(act_penalties)}"
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
    max_episode_steps=1000,
)

# model_path = "models/Jun30_13-12-28.489Orbit-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
# ENV_NAME = "Orbit-v0"
# model_path = "models/Jul05_22-42-01.624GoalContinuous2-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
# ENV_NAME = "GoalContinuous2-v0"

# model_path = "models/Jul15_18-21-47.854Kepler500-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2_env_vt0.1_rt0.1.pkl"
model_path = "models/Jul27_11-07-49.186Kepler500-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2_dense_reward5_actp0.5_numC0.01_step0.1.pkl"
ENV_NAME = "Kepler-v0"

td3 = TD3(
    env_name=ENV_NAME,
    obs_norm=False,
)

td3.load(model_path)

# ret = td3.test(5)
# print(ret)

visualise_episode_vanilla(td3)
