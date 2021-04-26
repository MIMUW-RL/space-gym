from gym.envs.registration import register

register(
    id="SpaceshipLand-v0",
    entry_point="gym_space.envs:SpaceshipLandV0",
    max_episode_steps=3_600
)

register(
    id="SpaceshipOrbitDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipOrbitDiscreteV0",
    max_episode_steps=3_600
)

register(
    id="SpaceshipOrbitContinuous-v0",
    entry_point="gym_space.envs:SpaceshipOrbitContinuousV0",
    max_episode_steps=3_600
)

# no rewards, just to show env with two planets
register(
    id="SpaceshipTwoPlanets-v0",
    entry_point="gym_space.envs:SpaceshipTwoPlanetsV0",
    max_episode_steps=3_600
)