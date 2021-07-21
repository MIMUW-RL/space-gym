from abc import ABC
import numpy as np

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space import helpers
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv

WORLD_SIZE = 3.0
MAX_OBJ_TILES_RATIO = 0.6
PLANET_TILE_RATIO = 0.8  # TODO: dynamic

class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9
    _max_position_sample_tries = 30

    def __init__(
        self,
        n_planets: int = 2,
        survival_reward_scale: float = 0.25,
        goal_vel_reward_scale: float = 0.75,
        goal_sparse_reward: float = 10.0,
        renderer_kwargs: dict = None,
    ):
        self.n_planets = n_planets
        if self.n_planets == 1:
            self.planets_radius = 0.8
            self.goal_radius = 0.2
            self.ship_radius = 0.2
        else:
            n_objects = self.n_planets + 2
            min_tiles = int(np.ceil(n_objects / MAX_OBJ_TILES_RATIO))
            r_ = (
                (np.sqrt(1 + 8 * (3 + np.sqrt(3)) * min_tiles) - 1) /
                (4 *  np.sqrt(3))
            )
            r = int(np.ceil(r_))
            while True:
                c = int(np.floor((2 / (np.sqrt(3) + 1)) *  (r * np.sqrt(3) + 0.5)))
                if r * c >= min_tiles:
                    break
                r += 1
            a = WORLD_SIZE / (0.5 * np.sqrt(3) + r * np.sqrt(3))
            self._hex_rows = r
            self._hex_cols = c
            self._hex_a = a
            self.planets_radius = 0.5 * a * np.sqrt(3) * PLANET_TILE_RATIO

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

    def _sample_position_outside_planet(self, planet_pos: np.ndarray, clearance: float):
        max_obj_pos = self.world_size / 2 - clearance
        obj_planet_angle = self._np_random.uniform(0, 2 * np.pi)
        obj_planet_unit_vec = helpers.angle_to_unit_vector(obj_planet_angle)
        obj_planet_center_max_dist = helpers.get_max_dist_in_direction(max_obj_pos, planet_pos, obj_planet_unit_vec)
        obj_planet_center_min_dist = self.planets_radius + clearance
        assert obj_planet_center_min_dist < obj_planet_center_max_dist
        obj_planet_center_dist = self._np_random.uniform(obj_planet_center_min_dist, obj_planet_center_max_dist)
        return planet_pos + obj_planet_unit_vec * obj_planet_center_dist

    def _sample_positions(self):
        max_pos = self.world_size / 2 - self.planets_radius
        if self.n_planets > 1:
            raise NotImplementedError

        planet_world_center_dist = self._np_random.uniform(0, max_pos - 2 * max(self.ship_radius, self.goal_radius))
        planet_world_center_angle = self._np_random.uniform(0, 2 * np.pi)
        planet_pos = helpers.angle_to_unit_vector(planet_world_center_angle) * planet_world_center_dist

        ship_pos = self._sample_position_outside_planet(planet_pos, self.ship_radius)
        goal_pos = self._sample_position_outside_planet(planet_pos, self.goal_radius)

        return ship_pos, goal_pos, planet_pos

    def _resample_goal(self):
        if self.n_planets == 1:
            self.goal_pos = self._sample_position_outside_planet(self.planets[0].center_pos, self.goal_radius)
        else:
            raise NotImplementedError
        if self._renderer is not None:
            self._renderer.move_goal(self.goal_pos)

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
        reward = self.survival_reward_scale + self.goal_vel_reward_scale * self._goal_vel_reward()
        if np.linalg.norm(self.goal_pos - self._ship_state.pos_xy) < 0.3:
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
        r = np.tanh(3 * vel_toward_goal)
        assert 0.0 <= r <= 1
        return r


class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass
