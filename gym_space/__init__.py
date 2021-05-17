from gym.envs.registration import register
from gym_space.envs.spaceship_env import DEFAULT_MAX_EPISODE_STEPS

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

# Do not crash

register(
    id="SpaceshipDoNotCrashDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipDoNotCrashDiscreteEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)

register(
    id="SpaceshipDoNotCrashContinuous-v0",
    entry_point="gym_space.envs:SpaceshipDoNotCrashContinuousEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)

# Orbit

register(
    id="SpaceshipOrbitDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipOrbitDiscreteEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)

register(
    id="SpaceshipOrbitContinuous-v0",
    entry_point="gym_space.envs:SpaceshipOrbitContinuousEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)

# Land

register(
    id="SpaceshipLandDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipLandDiscreteEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)

register(
    id="SpaceshipLandContinuous-v0",
    entry_point="gym_space.envs:SpaceshipLandContinuousEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS
)