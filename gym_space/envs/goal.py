from abc import ABC
import numpy as np

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space import helpers
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


AREA_RATIO = 0.3


class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9
    _max_position_sample_tries = 30

    def __init__(
        self,
        n_planets: int = 1,
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
            raise ValueError

        planet_world_center_dist = self._np_random.uniform(0, max_pos - 2 * max(self.ship_radius, self.goal_radius))
        planet_world_center_angle = self._np_random.uniform(0, 2 * np.pi)
        planet_pos = helpers.angle_to_unit_vector(planet_world_center_angle) * planet_world_center_dist

        ship_pos = self._sample_position_outside_planet(planet_pos, self.ship_radius)
        goal_pos = self._sample_position_outside_planet(planet_pos, self.goal_radius)

        return ship_pos, goal_pos, planet_pos

    def _resample_goal(self):
        self.goal_pos = self._sample_position_outside_planet(self.planets[0].center_pos, self.goal_radius)
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
        reward = self._goal_vel_reward()
        if np.linalg.norm(self.goal_pos - self._ship_state.pos_xy) < 0.3:
            reward += 10
            self._resample_goal()
        print(reward)
        return reward

    def _goal_vel_reward(self) -> float:
        ship_goal_vec = self.goal_pos - self._ship_state.pos_xy
        ship_goal_vec_norm = np.linalg.norm(ship_goal_vec)
        if np.isclose(ship_goal_vec_norm, 0.0):
            return 0.0
        ship_goal_unit_vec = ship_goal_vec / ship_goal_vec_norm
        # project velocity vector onto line from ship to goal
        vel_toward_goal = ship_goal_unit_vec @ self._ship_state.vel_xy
        if vel_toward_goal < 0:
            return 0.0
        r = np.tanh(vel_toward_goal)
        assert 0.0 <= r <= 1
        return r


class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass
