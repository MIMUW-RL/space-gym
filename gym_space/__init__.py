from gym.envs.registration import register

# Hover 1D

register(
    id="Hover1DDiscrete-v0",
    entry_point="gym_space.envs:Hover1DDiscreteEnv",
)

register(
    id="Hover1DContinuous-v0",
    entry_point="gym_space.envs:Hover1DContinuousEnv",
)

# Do not crash

register(
    id="DoNotCrashDiscrete-v0",
    entry_point="gym_space.envs:DoNotCrashDiscreteEnv",
)

register(
    id="DoNotCrashContinuous-v0",
    entry_point="gym_space.envs:DoNotCrashContinuousEnv",
)

