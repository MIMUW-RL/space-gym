from abc import ABC
import numpy as np
from scipy.stats import multivariate_normal

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space.sample_positions import maximize_dist
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9
    _max_position_sample_tries = 30

    def __init__(
        self,
        n_planets: int = 2,
        survival_reward_scale: float = 0.5,
        goal_dist_reward_scale: float = 0.5,
        economy_reward_scale: float = 0.0,
        goal_dist_reward_std: float = 0.1,
        test_env: bool = False,
        renderer_kwargs: dict = None,
    ):
        self.n_planets = n_planets
        world_size = 3.0
        n_objects = n_planets + 2
        n_objects_per_dim = np.ceil(np.sqrt(n_objects))
        self.planets_radius = 0.5 * world_size / n_objects_per_dim
        planets_mass = self._total_planets_mass / n_planets

        planets = [
            Planet(mass=planets_mass, radius=self.planets_radius)
            for _ in range(self.n_planets)
        ]
        ship = ShipParams(
            mass=1, moi=0.05, max_engine_force=0.3, max_thruster_force=0.05
        )

        assert (
            survival_reward_scale >= 0
            and goal_dist_reward_scale >= 0
            and economy_reward_scale >= 0
        ), "Reward scales have to be positive"
        assert (
            survival_reward_scale + goal_dist_reward_scale + economy_reward_scale == 1
        ), "Reward scales have to sum up to 1.0"

        self.survival_reward_scale = survival_reward_scale
        self.goal_dist_reward_scale = goal_dist_reward_scale
        self.economy_reward_scale = economy_reward_scale
        self.goal_dist_reward_std = goal_dist_reward_std

        self.test_env = test_env
        self._normal_pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * self.goal_dist_reward_std).pdf
        self._max_normal_pdf = self._normal_pdf(np.zeros(2))

        super().__init__(
            ship_params=ship,
            planets=planets,
            world_size=world_size,
            step_size=0.07,
            max_abs_vel_angle=5.0,
            vel_xy_std=np.ones(2),
            with_lidar=True,
            with_goal=True,
            renderer_kwargs=renderer_kwargs,
        )

    def _sample_positions(self):
        max_pos = self.world_size / 2 - self.planets_radius
        initial_pos = self._np_random.uniform(-max_pos, max_pos, size=(self.n_planets + 2, 2))
        return maximize_dist(initial_pos, max_pos)

    def _reset(self):
        *planets_pos, ship_pos, self.goal_pos = self._sample_positions()
        for pos, planet in zip(planets_pos, self.planets):
            planet.center_pos = pos
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self._ship_state.set(ship_pos, ship_angle, velocities_xy, angular_velocity)

    def _reward(self) -> float:
        reward = (
            self.survival_reward_scale
            + self.goal_dist_reward_scale * self._goal_dist_reward()
            + self.economy_reward_scale * self._economy_reward()
        )
        assert 0.0 <= reward <= 1
        return reward

    def _goal_dist_reward(self) -> float:
        r = self._normal_pdf(self._ship_state.pos_xy - self.goal_pos) / self._max_normal_pdf
        assert 0.0 <= r <= 1
        return r

    def _economy_reward(self) -> float:
        action_norm = np.linalg.norm(self.last_action)
        max_action_norm = np.sqrt(2)
        normalized_action_norm = action_norm / max_action_norm
        # function of action norm decreasing from 1 (no action) to 0 (max action)
        r = 1 - normalized_action_norm
        assert 0 <= r <= 1
        return r

class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass
