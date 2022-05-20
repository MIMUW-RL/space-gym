#!/usr/bin/env python
import time
import gym
import numpy as np

step_size = 0.15

gym.envs.register(
    id="KeplerDiscrete-v0",
    entry_point="gym_space.envs.kepler:KeplerDiscreteEnv",
    max_episode_steps=1000,
    kwargs={
        "reward_value": 0,
        "rad_penalty_C": 2,
        "numerator_C": 0.01,
        "act_penalty_C": 0.5,
        "step_size": step_size,
        "randomize": True,
    },
)

if __name__ == "__main__":
    # env = gym.make(f"gym_space:DoNotCrashDiscrete-v0")
    # env = gym.make(f"gym_space:KeplerDiscrete-v0")
    # env = gym.make(f"gym_space:GoalDiscrete-v0")

    gym.envs.register(
        id="GoalDiscrete3-v0",
        entry_point="gym_space.envs.goal:GoalDiscreteEnv",
        kwargs={
            "n_planets": 5,
            "ship_steering": 1,
            "ship_moi": 0.01,
            "survival_reward_scale": 0.25,
            "goal_vel_reward_scale": 0.25,
            "safety_reward_scale": 0.25,
            "goal_sparse_reward": 50.0,
        },
    )

    env = gym.make(f"gym_space:GoalDiscrete3-v0")

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
                time.sleep(0.1)
            k += 1
            time.sleep(0.1)
        print("END OF GAME! YOUR FINAL SCORE:")
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
        print(obser_max)
        time.sleep(5)

    print("ACTIONS={}".format(ACTIONS))
    print("Press left/right arrows (rotation) and space (engine)")
    print("No keys pressed is taking action 0")

    while 1:
        window_still_open = rollout(env)
        if window_still_open == False:
            break
