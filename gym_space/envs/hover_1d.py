from abc import ABC
from dataclasses import dataclass
import numpy as np

from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from .spaceship_env import (
    SpaceshipEnv,
    DiscreteSpaceshipEnv,
    ContinuousSpaceshipEnv
)


@dataclass
class Hover1DRewards(Rewards):
    planet_radius: float
    max_height: float
    partitions: int
    max_episode_steps: int

    def reward(self, state: np.array, _action: np.array, _done: bool):
        height_above_planet = state[1] - self.planet_radius
        score_between_0_and_1 = (
            max(self.max_height - height_above_planet, 0) / self.max_height
        )
        step_score = np.ceil(self.partitions * score_between_0_and_1) / self.partitions
        return step_score * 100 / self.max_episode_steps


class Hover1DEnv(SpaceshipEnv, ABC):
    def __init__(
        self,
        *,
        planet_radius: float = 10.0,
        planet_mass: float = 5e7,
        ship_mass: float = 0.1,
        ship_engine_force: float = 7e-6,
        step_size: float = 18.0,
        max_episode_steps: int = 300,
        reward_max_height: float = 3.0,
        reward_partitions: int = 1
    ):
        planet = Planet(center_pos=np.zeros(2), mass=planet_mass, radius=planet_radius)
        ship = Ship(
            mass=ship_mass,
            moi=1.0,
            max_engine_force=ship_engine_force,
            max_thruster_torque=0.0,
        )

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=Hover1DRewards(
                planet_radius, reward_max_height, reward_partitions, max_episode_steps
            ),
            step_size=step_size,
            max_episode_steps=max_episode_steps,
        )
        self._world_min = np.array([-3, planet_radius - 2])
        self._world_max = np.array([3, planet_radius + 1.5 * reward_max_height])
        self.reward_max_height = reward_max_height

    def _sample_initial_state(self):
        height_above_planet = self._np_random.uniform(0.0, self.reward_max_height)
        x = 0.0
        y = self.planets[0].radius + height_above_planet
        angle = 1.5 * np.pi
        velocities = np.zeros(3)
        return np.array([x, y, angle, *velocities])


class Hover1DDiscreteEnv(Hover1DEnv, DiscreteSpaceshipEnv):
    pass


class Hover1DContinuousEnv(Hover1DEnv, ContinuousSpaceshipEnv):
    pass
