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
    max_episode_steps=1000,
)

register(
    id="GoalContinuous2P-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    max_episode_steps=500,
    kwargs={
        "n_planets": 2,
        "ship_steering": 1,
        "ship_moi": 0.01,
        "survival_reward_scale": 0.2,
        "goal_vel_reward_scale": 5.0,
        "safety_reward_scale": 10.0,
        "goal_sparse_reward": 5.0,
        "max_engine_force": 0.4,
    },
)

register(
    id="GoalContinuous3P-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    max_episode_steps=500,
    kwargs={
        "n_planets": 3,
        "ship_steering": 1,
        "ship_moi": 0.01,
        "survival_reward_scale": 0.2,
        "goal_vel_reward_scale": 5.0,
        "safety_reward_scale": 10.0,
        "goal_sparse_reward": 5.0,
        "max_engine_force": 0.4,
    },
)

register(
    id="GoalContinuous4P-v0",
    entry_point="gym_space.envs:GoalContinuousEnv",
    max_episode_steps=500,
    kwargs={
        "n_planets": 4,
        "ship_steering": 1,
        "ship_moi": 0.01,
        "survival_reward_scale": 0.2,
        "goal_vel_reward_scale": 5.0,
        "safety_reward_scale": 10.0,
        "goal_sparse_reward": 5.0,
        "max_engine_force": 0.4,
    },
)

# selection of Kepler orbit problem environments

step_size = 0.07
max_episode_steps = 500

register(
    id="KeplerCircleOrbit-v0",
    entry_point="gym_space.envs:KeplerContinuousEnv",
    max_episode_steps=max_episode_steps,
    kwargs={
        "ship_steering": 1,
        "ship_moi": 0.01,
        "rad_penalty_C": 2,
        "numerator_C": 0.01,
        "act_penalty_C": 0.5,
        "step_size": step_size,
        "randomize": False,
        "ref_orbit_a": 1.2,
        "ref_orbit_eccentricity": 0,
        "ref_orbit_angle": 0,
    },
)

register(
    id="KeplerEllipseEasy-v0",
    entry_point="gym_space.envs:KeplerContinuousEnv",
    max_episode_steps=max_episode_steps,
    kwargs={
        "ship_steering": 1,
        "ship_moi": 0.01,
        "rad_penalty_C": 2,
        "numerator_C": 0.01,
        "act_penalty_C": 0.5,
        "step_size": step_size,
        "randomize": False,
        "ref_orbit_a": 1.2,
        "ref_orbit_eccentricity": 0.5,
        "ref_orbit_angle": 0.8,
    },
)

register(
    id="KeplerEllipseHard-v0",
    entry_point="gym_space.envs:KeplerContinuousEnv",
    max_episode_steps=max_episode_steps,
    kwargs={
        "ship_steering": 1,
        "ship_moi": 0.01,
        "rad_penalty_C": 2,
        "numerator_C": 0.01,
        "act_penalty_C": 0.5,
        "step_size": step_size,
        "randomize": False,
        "ref_orbit_a": 1.2,
        "ref_orbit_eccentricity": 0.725,
        "ref_orbit_angle": 3.925,
    },
)

register(
    id="KeplerRandomOrbits-v0",
    entry_point="gym_space.envs:KeplerContinuousEnv",
    max_episode_steps=max_episode_steps,
    kwargs={
        "ship_steering": 1,
        "ship_moi": 0.01,
        "rad_penalty_C": 2,
        "numerator_C": 0.01,
        "act_penalty_C": 0.5,
        "step_size": step_size,
        "randomize": True,
    },
)
