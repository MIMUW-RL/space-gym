from gym.envs.registration import register

# Orbit

register(
    id="KeplerDiscrete-v0",
    entry_point = "gym_space.envs:KeplerDiscreteEnv",
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

#Kepler problem env

register(
    id="KeplerContinuous-v0",
    entry_point = "gym_space.envs:KeplerContinuousEnv",
)