from gym.envs.registration import register

# Do not crash

register(
    id="DoNotCrashDiscrete-v0",
    entry_point="gym_space.envs:DoNotCrashDiscreteEnv",
    max_episode_steps=300,
)

register(
    id="DoNotCrashContinuous-v0",
    entry_point="gym_space.envs:DoNotCrashContinuousEnv",
    max_episode_steps=300,
)


# Goal

register(
    id="GoalDiscrete-v0",
    entry_point="gym_space.envs:GoalDiscreteEnv",
    max_episode_steps=10000,
)

register(
    id="GoalContinuous-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    max_episode_steps=300,
)
