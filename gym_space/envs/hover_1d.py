from abc import ABC
from dataclasses import dataclass
import numpy as np

from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv, DEFAULT_STEP_SIZE

MAX_HEIGHT = 3


@dataclass
class Hover1DRewards(Rewards):
    planet_radius: float
    step_size: float = DEFAULT_STEP_SIZE

    def reward(self, state: np.array, _action: np.array, _done: bool):
        height_above_planet = state[1] - self.planet_radius
        partitions = 1
        score_between_0_and_1 = max(MAX_HEIGHT - height_above_planet, 0) / MAX_HEIGHT
        step_score = np.ceil(partitions * score_between_0_and_1)
        return step_score / (partitions * 3)


class SpaceshipHover1DEnv(SpaceshipEnv, ABC):
    def __init__(self):
        radius = 2.0
        planet = Planet(center_pos=np.zeros(2), mass=4e4, radius=radius)
        ship = Ship(mass=1.0, moi=1.0, max_engine_force=1e-6, max_thruster_torque=0.0)

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=Hover1DRewards(radius),
        )
        self._world_min = np.array([-3, 0.0])
        self._world_max = np.array([3, 7.5])

    def _sample_initial_state(self):
        height_above_planet = np.random.uniform(0.0, MAX_HEIGHT)
        x = 0.0
        y = self.planets[0].radius + height_above_planet
        angle = 1.5 * np.pi
        velocities = np.zeros(3)
        return np.array([x, y, angle, *velocities])


class SpaceshipHover1DDiscreteEnv(SpaceshipHover1DEnv, DiscreteSpaceshipEnv):
    pass


class SpaceshipHover1DContinuousEnv(SpaceshipHover1DEnv, ContinuousSpaceshipEnv):
    pass