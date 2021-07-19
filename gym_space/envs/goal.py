from abc import ABC
import numpy as np
from scipy.stats import multivariate_normal

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space.sample_positions import maximize_dist
from gym_space import helpers
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


AREA_RATIO = 0.3


class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9
    _max_position_sample_tries = 30

    def __init__(
        self,
        n_planets: int = 1,
        survival_reward_scale: float = 0.5,
        goal_dist_reward_scale: float = 0.5,
        economy_reward_scale: float = 0.0,
        goal_dist_reward_std: float = 0.1,
        test_env: bool = False,
        renderer_kwargs: dict = None,
    ):
        self.n_planets = n_planets
        world_size = 3.0
        self.goal_radius = 0.2
        self.ship_radius = 0.2
        if n_planets == 1:
            self.planets_radius = 0.8
        elif n_planets == 2:
            self.planets_radius = 0.4
        else:
            # 4 (goal_r^2 + ship_r^2 + n planets_r^2) = AREA_RATIO * world_size^2
            # 4 n planets_r^2 = AREA_RATIO * world_size^2 - 4 (goal_r^2 + ship_r^2)
            # planets_r^2 = [ AREA_RATIO * world_size^2 - 4 (goal_r^2 + ship_r^2) ] / 4n
            # planets_r^2 = [ AREA_RATIO * world_size^2 / 4 - goal_r^2 - ship_r^2 ] / n
            self.planets_radius = np.sqrt((AREA_RATIO * world_size**2 / 4 - self.goal_radius**2 - self.ship_radius**2) / n_planets)

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
        self._normal_pdf = multivariate_normal(
            mean=np.zeros(2), cov=np.eye(2) * self.goal_dist_reward_std
        ).pdf
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
        if self.n_planets > 1:
            initial_pos = self._np_random.uniform(
                -max_pos, max_pos, size=(self.n_planets + 2, 2)
            )
            radii = [self.ship_radius, self.goal_radius, self.planets_radius]
            return maximize_dist(initial_pos, max_pos, radii)
        max_ship_pos = self.world_size / 2 - self.ship_radius
        max_goal_pos = self.world_size / 2 - self.goal_radius

        planet_world_center_dist = self._np_random.uniform(0, max_pos - 2 * max(self.ship_radius, self.goal_radius))
        planet_world_center_angle = self._np_random.uniform(0, 2 * np.pi)
        planet_pos = helpers.angle_to_unit_vector(planet_world_center_angle) * planet_world_center_dist
        ship_planet_angle = self._np_random.uniform(0, 2 * np.pi)
        goal_planet_angle = (ship_planet_angle - np.pi + 2 * self._np_random.normal()) % (2 * np.pi)
        ship_planet_unit_vec = helpers.angle_to_unit_vector(ship_planet_angle)
        goal_planet_unit_vec = helpers.angle_to_unit_vector(goal_planet_angle)

        def get_max_dist_in_direction(max_pos_, obj_pos, direction_unit_vec):
            candidate_max_dist = (
                (max_pos_ - obj_pos[0]) / direction_unit_vec[0],
                (- max_pos_ - obj_pos[0]) / direction_unit_vec[0],
                (max_pos_ - obj_pos[1]) / direction_unit_vec[1],
                (- max_pos_ - obj_pos[1]) / direction_unit_vec[1],
            )
            candidate_max_dist = filter(lambda x: x > 0, candidate_max_dist)
            return min(candidate_max_dist)

        ship_planet_center_max_dist = get_max_dist_in_direction(max_ship_pos, planet_pos, ship_planet_unit_vec)
        ship_planet_center_min_dist = self.planets_radius + self.ship_radius
        assert ship_planet_center_min_dist < ship_planet_center_max_dist
        ship_planet_center_dist = self._np_random.uniform(ship_planet_center_min_dist, ship_planet_center_max_dist)
        # ship_planet_center_dist = ship_planet_center_max_dist
        ship_pos = planet_pos + ship_planet_unit_vec * ship_planet_center_dist

        goal_planet_center_max_dist = get_max_dist_in_direction(max_goal_pos, planet_pos, goal_planet_unit_vec)
        goal_planet_center_min_dist = self.planets_radius + self.goal_radius
        assert goal_planet_center_min_dist < goal_planet_center_max_dist
        goal_planet_center_dist = self._np_random.uniform(goal_planet_center_min_dist, goal_planet_center_max_dist)
        # goal_planet_center_dist = goal_planet_center_max_dist
        goal_pos = planet_pos + goal_planet_unit_vec * goal_planet_center_dist

        return ship_pos, goal_pos, planet_pos

    def _reset(self):
        ship_pos, self.goal_pos, *planets_pos = self._sample_positions()
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
        r = (
            self._normal_pdf(self._ship_state.pos_xy - self.goal_pos)
            / self._max_normal_pdf
        )
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
