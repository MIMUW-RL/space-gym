from abc import ABC
from dataclasses import dataclass
import numpy as np

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from gym_space.helpers import vector_to_angle
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


@dataclass
class OrbitPlanetRewards(Rewards):
    planet: Planet
    step_size: float = ...  # FIXME

    def reward(self, state: np.array, action: np.array):
        ship_xy = state[:2]
        ship_planet_angle = vector_to_angle(ship_xy - self.planet.center_pos)
        ship_velocity_xy = state[3:5]
        ship_velocity_xy_angle = vector_to_angle(ship_velocity_xy)
        angle_diff = (ship_velocity_xy_angle - ship_planet_angle) % (2 * np.pi)
        angular_velocity_around_planet = np.sin(angle_diff) * np.linalg.norm(ship_velocity_xy)
        return angular_velocity_around_planet / (self.step_size * 1e3)


class SpaceshipOrbitEnv(SpaceshipEnv, ABC):
    # FIXME: it doesn't work
    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=5.972e24, radius=6.371e6)
        super().__init__(
            ship=Ship(mass=1e4, moi=1, max_engine_force=1e5, max_thruster_torque=1e-5),
            planets=[planet],
            rewards=OrbitPlanetRewards(planet),
        )

    def _sample_initial_state(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_angle = (planet_angle - np.pi/2) % (2 * np.pi)
        pos_xy = angle_to_unit_vector(planet_angle) * self.planets[0].radius * 1.3
        velocities_xy = - angle_to_unit_vector(ship_angle) * 5e3
        return np.array([*pos_xy, ship_angle, *velocities_xy, 0.0])


class SpaceshipOrbitDiscreteEnv(SpaceshipOrbitEnv, DiscreteSpaceshipEnv):
    pass


class SpaceshipOrbitContinuousEnv(SpaceshipOrbitEnv, ContinuousSpaceshipEnv):
    pass