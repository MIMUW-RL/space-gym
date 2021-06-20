from abc import ABC
import numpy as np
import torch

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import ConstantRewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class DoNotCrashEnv(SpaceshipEnv, ABC):
    planet_radius = 0.25
    border_radius = 1.0

    def __init__(self, with_accelerations: bool=False, test_env=False):
        planet = Planet(center_pos=np.zeros(2), mass=1e6, radius=self.planet_radius)
        border = Planet(center_pos=np.zeros(2), mass=0.0, radius=self.border_radius)
        ship = Ship(mass=1, moi=1, max_engine_force=1e-3, max_thruster_torque=3e-3)

        super().__init__(
            ship=ship,
            planets=[planet, ],  # FIXME!
            rewards=ConstantRewards(100 / self.max_episode_steps),
            with_accelerations=with_accelerations
        )

        self.test_env = test_env

    def _sample_initial_state(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(self.planet_radius + 0.2, self.border_radius - 0.15)
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.015
        if self.test_env:
            velocities_xy /= 2
        max_abs_ang_vel = 0.9 * self.max_abs_angular_velocity
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        return np.array([*pos_xy, ship_angle, *velocities_xy, angular_velocity])


class DoNotCrashDiscreteEnv(DoNotCrashEnv, DiscreteSpaceshipEnv):
    pass


class DoNotCrashContinuousEnv(DoNotCrashEnv, ContinuousSpaceshipEnv):
    pass