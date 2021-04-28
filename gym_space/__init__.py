from gym.envs.registration import register
from gym_space.envs.spaceship_env import DEFAULT_MAX_EPISODE_STEPS

# Go to point

register(
    id="SpaceshipGoToPlanetDiscrete-v0",
    entry_point="gym_space.envs:SpaceshipGoToPlanetDiscreteEnv",
    max_episode_steps=500
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