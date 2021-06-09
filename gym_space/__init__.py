from gym.envs.registration import register

# Hover

register(
    id="SpaceshipHover1DDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipHover1DDiscreteEnv",
    max_episode_steps=300
)

register(
    id="SpaceshipHover1DContinuous-v0",
    entry_point="gym_space.envs:SpaceshipHover1DContinuousEnv",
    max_episode_steps=300
)


# Go to planet

register(
    id="SpaceshipGoToPlanetDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipGoToPlanetDiscreteEnv",
    max_episode_steps=350
)

register(
    id="SpaceshipGoToPlanetContinuous-v0",
    entry_point="gym_space.envs:SpaceshipGoToPlanetContinuousEnv",
    max_episode_steps=350
)
