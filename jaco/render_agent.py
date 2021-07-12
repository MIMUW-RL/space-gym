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

def visualise_episode_vanilla(algo):
    obs = algo.env.reset()
    algo.env.render()
    done = False
    ep_ret = 0
    i=0
    while(not done):       
        obs = algo.process_obs(obs)
        obs = algo.replay_buffer.normalize( obs )        
        action, _ = algo._actor.act( obs , deterministic=True)                
        action = algo.process_action(action, obs)
        obs, r, done, _ = algo.env.step(action)
        frame = algo.env.render(mode = "rgb_array")
        plt.imsave(f'gifs/{i:05d}.png', frame)        
        ep_ret += r
        i+=1
    print(f"ep_ret={ep_ret}")
        

gym.envs.register(
    id='Orbit-v0',
    entry_point='gym_space.envs.orbit:OrbitEnv',
    max_episode_steps=300,
)

gym.envs.register(
    id="GoalContinuous1-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    kwargs = {'n_planets': 1},
)

gym.envs.register(
    id="GoalContinuous2-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    kwargs = {'n_planets': 2},
)

gym.envs.register(
    id='Kepler-v0',
    entry_point='gym_space.envs.kepler:KeplerContinuousEnv',
)

#model_path = "models/Jun30_13-12-28.489Orbit-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
#ENV_NAME = "Orbit-v0"
#model_path = "models/Jul05_22-42-01.624GoalContinuous2-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
#ENV_NAME = "GoalContinuous2-v0"

model_path = "models/Jul10_15-56-08.399Kepler-v0-g0.99-spe5000-TD3-a_lr0.0003-rf0-noi0.2-obs_normFalse-pi_ufr2.pkl"
ENV_NAME = "Kepler-v0"

td3 = TD3(
    env_name=ENV_NAME,        
    obs_norm=False, 
    max_episode_steps = 500,   
)

td3.load(model_path)

#ret = td3.test(5)
#print(ret)

visualise_episode_vanilla(td3)

