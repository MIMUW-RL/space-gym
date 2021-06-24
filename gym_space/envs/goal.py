from abc import ABC
import numpy as np

from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import ConstantRewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class GoalEnv(SpaceshipEnv, ABC):
    max_episode_steps = 300
    _n_planets = 4
    _planets_radius = 0.3
    _planets_mass = 2e8


    def __init__(self):
        planets = [Planet(mass=self._planets_mass, radius=self._planets_radius) for _ in range(self._n_planets)]
        ship = Ship(mass=1, moi=0.05, max_engine_force=0.3, max_thruster_torque=0.05)

        super().__init__(
            ship=ship,
            planets=planets,
            rewards=ConstantRewards(100 / self.max_episode_steps),  # TODO
            world_size=3.0,
            step_size=0.07,
            max_abs_angular_velocity=5.0,
            velocity_xy_std=np.ones(2),
            with_lidar=True,
            with_goal=True
        )

    def _sample_positions(self):
        n_found = 0
        while n_found < self._n_planets + 2:
            new_pos = self._np_random.uniform(-0.5, 0.5, 2) * self.world_size
            for other_planet in self.planets[:n_found]:
                if other_planet.distance(new_pos) < self._planets_radius:
                    break
            else:
                n_found += 1
                if n_found <= self._n_planets:
                    self.planets[n_found - 1].center_pos = new_pos
                elif n_found <= self._n_planets + 2:
                    # return ship position, then goal position
                    yield new_pos

    def _reset(self):
        pos_xy, goal_pos = self._sample_positions()
        self.goal_pos = goal_pos
        if self._renderer is not None:
            self._renderer.move_goal(self.goal_pos)
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_angular_velocity
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self.internal_state = np.array([*pos_xy, ship_angle, *velocities_xy, angular_velocity])


class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass