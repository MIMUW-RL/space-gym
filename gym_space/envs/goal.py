from abc import ABC
import numpy as np
from typing import Union
import itertools

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space import helpers
from gym_space.hexagonal_tiling import HexagonalTiling
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv

WORLD_SIZE = 3.0


class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9

    def __init__(
        self,
        n_planets: int = 2,
        survival_reward_scale: float = 0.0,
        goal_vel_reward_scale: float = 0.75,
        safety_reward_scale: float = 0.25,
        goal_sparse_reward: float = 10.0,
        renderer_kwargs: dict = None,
    ):
        self.n_planets = n_planets
        self.n_objects = self.n_planets + 2

        self._hexagonal_tiling = None
        if self.n_planets == 1:
            self._init_one_planet()
        else:
            self._init_many_planets(n_planets)

        planets_mass = self._total_planets_mass / n_planets
        planets = [
            Planet(mass=planets_mass, radius=self.planets_radius)
            for _ in range(self.n_planets)
        ]
        ship = ShipParams(
            mass=1, moi=0.05, max_engine_force=0.3, max_thruster_force=0.05
        )

        self.survival_reward_scale = survival_reward_scale
        self.goal_vel_reward_scale = goal_vel_reward_scale
        self.goal_sparse_reward = goal_sparse_reward
        self.safety_reward_scale = safety_reward_scale

        super().__init__(
            ship_params=ship,
            planets=planets,
            world_size=WORLD_SIZE,
            step_size=0.07,
            max_abs_vel_angle=5.0,
            vel_xy_std=np.ones(2),
            with_lidar=True,
            with_goal=True,
            renderer_kwargs=renderer_kwargs,
        )

    def seed(self, seed=None):
        super().seed(seed)
        self._hexagonal_tiling.seed(seed)

    def _init_one_planet(self):
        self.planets_radius = 0.8
        self.goal_radius = 0.2
        self.ship_radius = 0.2

    def _init_many_planets(self, n_planets: int):
        self._hexagonal_tiling = HexagonalTiling(n_planets, WORLD_SIZE)
        self.planets_radius = self._hexagonal_tiling.planets_radius
        self.goal_radius = self._hexagonal_tiling.ship_radius
        self.goal_radius = self._hexagonal_tiling.goal_radius

    def _sample_position_outside_one_planet(self, planet_pos: np.ndarray, clearance: float):
        max_obj_pos = self.world_size / 2 - clearance
        obj_planet_angle = self._np_random.uniform(0, 2 * np.pi)
        obj_planet_unit_vec = helpers.angle_to_unit_vector(obj_planet_angle)
        obj_planet_center_max_dist = helpers.get_max_dist_in_direction(max_obj_pos, planet_pos, obj_planet_unit_vec)
        obj_planet_center_min_dist = self.planets_radius + clearance
        assert obj_planet_center_min_dist < obj_planet_center_max_dist
        obj_planet_center_dist = self._np_random.uniform(obj_planet_center_min_dist, obj_planet_center_max_dist)
        return planet_pos + obj_planet_unit_vec * obj_planet_center_dist

    def _sample_positions_with_one_planet(self):
        max_pos = self.world_size / 2 - self.planets_radius
        planet_world_center_dist = self._np_random.uniform(0, max_pos - 2 * max(self.ship_radius, self.goal_radius))
        planet_world_center_angle = self._np_random.uniform(0, 2 * np.pi)
        planet_pos = helpers.angle_to_unit_vector(planet_world_center_angle) * planet_world_center_dist

        ship_pos = self._sample_position_outside_one_planet(planet_pos, self.ship_radius)
        goal_pos = self._sample_position_outside_one_planet(planet_pos, self.goal_radius)

        return ship_pos, goal_pos, planet_pos

    def _sample_position_with_many_planets(self):
        return self._hexagonal_tiling.reset()

    def _sample_positions(self):
        if self.n_planets == 1:
            return self._sample_positions_with_one_planet()
        else:
            return self._sample_position_with_many_planets()

    def _find_new_goal_with_one_planet(self):
        return self._sample_position_outside_one_planet(self.planets[0].center_pos, self.goal_radius)

    def _find_new_goal_with_many_planets(self):
        return self._hexagonal_tiling.find_new_goal()

    def _resample_goal(self):
        if self.n_planets == 1:
            self.goal_pos = self._find_new_goal_with_one_planet()
        else:
            self.goal_pos = self._find_new_goal_with_many_planets()
        if self._renderer is not None:
            self._renderer.move_goal(self.goal_pos)

    def _reset(self):
        ship_pos, *planets_pos = self._sample_positions()
        self._resample_goal()
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
                self.survival_reward_scale +
                self.goal_vel_reward_scale * self._goal_vel_reward() +
                self.safety_reward_scale * self._safety_reward()
        )
        threshold = 0.9 if self._hexagonal_tiling._debug else self.goal_radius
        if np.linalg.norm(self.goal_pos - self._ship_state.pos_xy) < threshold:
            reward += self.goal_sparse_reward
            self._resample_goal()
        return reward

    def _goal_vel_reward(self) -> float:
        ship_goal_vec = self.goal_pos - self._ship_state.pos_xy
        ship_goal_vec_norm = np.linalg.norm(ship_goal_vec)
        if np.isclose(ship_goal_vec_norm, 0.0):
            return 0.0
        ship_goal_unit_vec = ship_goal_vec / ship_goal_vec_norm
        # project velocity vector onto line from ship to goal
        vel_toward_goal = ship_goal_unit_vec @ self._ship_state.vel_xy
        # no negative reward that could encourage crashing
        if vel_toward_goal < 0:
            return 0.0
        # don't encourage very high velocities
        r = np.tanh(5 * vel_toward_goal)
        assert 0.0 <= r <= 1
        return r

    def _safety_reward(self) -> float:
        """Give reward for the time it would take to crash if velocity didn't change"""
        vel_x, vel_y = self._ship_state.vel_xy
        if np.isclose(vel_x, 0):
            # TODO
            return 1
        a = vel_y / vel_x
        ship_x, ship_y = self._ship_state.pos_xy
        b = ship_y - a * ship_x

        min_dist = np.inf
        # Find intersections with planets
        for planet in self.planets:
            x0, y0 = planet.center_pos
            r = planet.radius
            # (x-x0)^2 + (y-y0)^2 = r^2
            # ax + b = y
            #
            # a_ x^2 + b_ x + c_ = 0
            # where
            a_ = a**2 + 1
            b_ = 2 * a * (b - y0) - 2 * x0
            c_ = x0**2 - r**2 + (b - y0)**2
            delta = b_**2 - 4 * a_ * c_
            if delta < 0:
                continue
            sqrt_delta = np.sqrt(delta)
            x_sol_0 = (-b_ + sqrt_delta) / (2 * a_)
            x_sol_1 = (-b_ - sqrt_delta) / (2 * a_)
            for x_sol in (x_sol_0, x_sol_1):
                if np.sign(x_sol - ship_x) != np.sign(vel_x):
                    continue
                y_sol = a * x_sol + b
                dist = np.sqrt((ship_x - x_sol)**2 + (ship_y - y_sol)**2)
                min_dist = min(min_dist, dist)

        # Find intersections with world boundary
        s = self.world_size / 2
        if vel_x > 0 and -s <= (y := a * s + b) <= s:
            dist = np.sqrt((ship_x - s)**2 + (ship_y - y)**2)
            min_dist = min(min_dist, dist)
        if vel_x < 0 and -s <= (y := - a * s + b) <= s:
            dist = np.sqrt((ship_x + s)**2 + (ship_y - y)**2)
            min_dist = min(min_dist, dist)
        if a != 0:
            if vel_y > 0 and -s <= (x := (s - b) / a) <= s:
                dist = np.sqrt((ship_x - x)**2 + (ship_y - s)**2)
                min_dist = min(min_dist, dist)
            if vel_y < 0 and -s <= (x := (- s - b) / a) <= s:
                dist = np.sqrt((ship_x - x)**2 + (ship_y + s)**2)
                min_dist = min(min_dist, dist)

        time_to_crash = min_dist / np.linalg.norm(self._ship_state.vel_xy)
        reward = np.tanh(time_to_crash / 5)
        return reward

class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass
