from gym.envs.registration import register

# Do not crash

register(
    id="DoNotCrashDiscrete-v0",
    entry_point="gym_space.envs:DoNotCrashDiscreteEnv",
)

register(
    id="DoNotCrashContinuous-v0",
    entry_point="gym_space.envs:DoNotCrashContinuousEnv",
)


# Goal

register(
    id="GoalDiscrete-v0",
    entry_point="gym_space.envs:GoalDiscreteEnv",
)

register(
    id="GoalContinuous-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
)

# Orbit

register(
    id="KeplerDiscrete-v0",
    entry_point = "gym_space.envs:KeplerDiscreteEnv",
)

register(
    id="KeplerContinuous-v0",
    entry_point = "gym_space.envs:KeplerContinuousEnv",
)