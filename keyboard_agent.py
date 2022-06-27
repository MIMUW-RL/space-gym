#!/usr/bin/env python
import time
import gym
import numpy as np

if __name__ == "__main__":
    total_rewards = []
    episodes = 5

    gym.envs.register(
        id=f"KeplerDiscrete-v0",
        entry_point="gym_space.envs:KeplerDiscreteEnv",
        kwargs={
            "ship_steering": 1,
            "ship_moi": 0.01,
            "max_engine_force": 0.4,
            "reward_value": 0,
            "rad_penalty_C": 2,
            "numerator_C": 0.01,
            "act_penalty_C": 0.5,
            "step_size": 0.07,
            "randomize": False,
            "ref_orbit_a": 1.2,
            "ref_orbit_eccentricity": 0,
            "ref_orbit_angle": 0,
        },
    )
    #env = gym.make("KeplerDiscrete-v0")

    gym.envs.register(
        id="GoalDiscrete2-v0",
        entry_point="gym_space.envs.goal:GoalDiscreteEnv",
        max_episode_steps = 500,
        kwargs={
            "n_planets": 2,
            "ship_steering": 1,
            "ship_moi": 0.01,
            "survival_reward_scale": 0.2,
            "goal_vel_reward_scale": 5.0,
            "safety_reward_scale": 10.0,
            "goal_sparse_reward": 5.0,
            "max_engine_force": 1,
        },
    )
    gym.envs.register(
        id="GoalDiscrete3-v0",
        entry_point="gym_space.envs.goal:GoalDiscreteEnv",
        max_episode_steps = 500,
        kwargs={
            "n_planets": 3,
            "ship_steering": 1,
            "ship_moi": 0.01,
            "survival_reward_scale": 0.2,
            "goal_vel_reward_scale": 5.0,
            "safety_reward_scale": 10.0,
            "goal_sparse_reward": 5.0,
            "max_engine_force": 1,
        },
    )
    gym.envs.register(
        id="GoalDiscrete4-v0",
        entry_point="gym_space.envs.goal:GoalDiscreteEnv",
        max_episode_steps = 500,
        kwargs={
            "n_planets": 4,
            "ship_steering": 1,
            "ship_moi": 0.01,
            "survival_reward_scale": 0.2,
            "goal_vel_reward_scale": 5.0,
            "safety_reward_scale": 10.0,
            "goal_sparse_reward": 5.0,
            "max_engine_force": 1,
        },
    )
    env = gym.make(f"GoalDiscrete3-v0")

    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")
    ACTIONS = env.action_space.n
    print(ACTIONS)
    SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
    # can test what skip is still usable.

    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False

    def key_press(key, mod):
        global human_agent_action, human_wants_restart, human_sets_pause
        if key == 0xFF0D:
            human_wants_restart = True
        if key == 32:
            a = 1
        if key == 65361:  # left arrow
            a = 2
        if key == 65363:  # right arrow
            a = 3
        # a = int(key - ord("0"))
        if a <= 0 or a >= ACTIONS:
            return
        human_agent_action = a

    def key_release(key, mod):
        global human_agent_action
        # a = int(key - ord("0"))
        if key == 32:
            a = 1
        if key == 65361:  # left arrow
            a = 2
        if key == 65363:  # right arrow
            a = 3
        # if a <= 0 or a >= ACTIONS:
        #    return
        if human_agent_action == a:
            human_agent_action = 0

    env.reset()
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    
    def rollout(env):
        global human_agent_action, human_wants_restart, human_sets_pause
        human_wants_restart = False
        obser = env.reset()
        obser_max = np.abs(obser)
        skip = 0
        total_reward = 0
        total_timesteps = 0
        k = 0
        
        while 1:
            if not skip:
                # print("taking action {}".format(human_agent_action))
                a = human_agent_action
                total_timesteps += 1
                skip = SKIP_CONTROL
            else:
                skip -= 1

            obser, r, done, info = env.step(a)
            print(obser)
            print(obser.shape)
            obser_max = np.maximum(np.abs(obser), obser_max)
            total_reward += r
            print(f"step {k} total rew={total_reward}")

            window_still_open = env.render()
            if window_still_open == False:
                return False
            if done:
                break
            if human_wants_restart:
                break
            while human_sets_pause:
                env.render()
                time.sleep(0.01)
            k += 1
            time.sleep(0.1)
        print("END OF GAME! YOUR FINAL SCORE:")
        total_rewards.append(total_reward)
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
        print(obser_max)
        time.sleep(2)

    for e in range(episodes):
        window_still_open = rollout(env)
        if window_still_open == False:
            break

    print("HUMAN BASELINE SCORE:\n")
    print(np.mean(total_rewards))
    print(np.std(total_rewards))
