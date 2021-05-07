from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import ConstantRewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class SpaceshipDoNotCrashEnv(SpaceshipEnv, ABC):
    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=5.972e24, radius=6.371e6)
        ship = Ship(mass=1e4, moi=1, max_engine_force=6.04e4, max_thruster_torque=2e-5)

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=ConstantRewards(1.0),
        )

    def _sample_initial_state(self):
        planet_angle = np.random.uniform(0, 2 * np.pi)
        ship_angle = (planet_angle + (4/3) * np.pi) % (2 * np.pi)
        pos_xy = angle_to_unit_vector(planet_angle) * self.planets[0].radius * 1.3
        velocities_xy = - angle_to_unit_vector(ship_angle) * 2e3
        return np.array([*pos_xy, ship_angle, *velocities_xy, 0.0])


class SpaceshipDoNotCrashDiscreteEnv(SpaceshipDoNotCrashEnv, DiscreteSpaceshipEnv):
    pass


class SpaceshipDoNotCrashContinuousEnv(SpaceshipDoNotCrashEnv, ContinuousSpaceshipEnv):
    pass