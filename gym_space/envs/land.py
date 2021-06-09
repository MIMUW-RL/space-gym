from dataclasses import dataclass
from abc import ABC
import numpy as np

from gym_space.ship import Ship
from gym_space.planet import Planet
from gym_space.helpers import bounded_linear, bounded_square
from gym_space.rewards import Rewards
from .spaceship_env import (
    SpaceshipEnv,
    DiscreteSpaceshipEnv,
    ContinuousSpaceshipEnv
)


@dataclass
class LandRewards(Rewards):
    destination_reward: float
    max_destination_distance_penalty: float
    max_reasonable_distance: float
    max_fuel_penalty: float
    max_landing_velocity_penalty: float
    max_landing_angle_penalty: float
    destination_planet: Planet
    max_episode_steps: int

    def _destination_distance(self, state):
        ship_xy = state[:2]
        distance_from_planet_center = np.linalg.norm(
            ship_xy - self.destination_planet.center_pos
        )
        return distance_from_planet_center - self.destination_planet.radius

    def _destination_distance_penalty(self, destination_distance: float):
        max_penalty_per_step = (
            self.max_destination_distance_penalty / self.max_episode_steps
        )
        penalty_bounded_by_one = (
            bounded_linear(destination_distance, self.max_reasonable_distance)
            / self.max_reasonable_distance
        )
        return max_penalty_per_step * penalty_bounded_by_one

    def _fuel_penalty(self, action):
        max_penalty_per_step = self.max_fuel_penalty / self.max_episode_steps
        engine_action = action[0]  # in [0, 1]
        return engine_action * max_penalty_per_step

    def _landing_velocity_penalty(self, state: np.array):
        velocity_norm = np.linalg.norm(state[3:])
        return bounded_square(velocity_norm, self.max_landing_velocity_penalty)

    def _landing_angle_penalty(self, state: np.array):
        landing_spot_on_planet = state[:2] - self.destination_planet.center_pos
        landing_spot_on_planet_angle = np.arctan2(
            landing_spot_on_planet[1], landing_spot_on_planet[0]
        )
        perfect_landing_angle = (landing_spot_on_planet_angle - np.pi / 2) % (2 * np.pi)
        return (
            self.max_landing_angle_penalty
            * np.sin(state[2] - perfect_landing_angle) ** 2
        )

    def reward(self, state: np.array, action: np.array, done: bool):
        reward = 0.0
        destination_distance = self._destination_distance(state)
        reward -= self._destination_distance_penalty(destination_distance)
        reward -= self._fuel_penalty(action)
        if done and np.isclose(0, destination_distance):
            reward += self.destination_reward
            reward -= self._landing_velocity_penalty(state)
            reward -= self._landing_angle_penalty(state)
        return reward


class SpaceshipLandEnv(SpaceshipEnv, ABC):
    def __init__(self):
        ship = Ship(mass=5.5e4, moi=1, max_engine_force=1e6, max_thruster_torque=1e-4)
        planet = Planet(center_pos=np.zeros(2), mass=5.972e24, radius=6.371e6)
        rewards = LandRewards(
            destination_reward=10_000,
            max_destination_distance_penalty=400,
            max_reasonable_distance=5e6,
            max_fuel_penalty=100,
            max_landing_velocity_penalty=2_500,
            max_landing_angle_penalty=2_500,
            destination_planet=planet
        )

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=rewards,
        )

    def _sample_initial_state(self):
        try_nr = 0
        while True:
            if try_nr > 100:
                raise ValueError("Could not find correct initial state")
            try_nr += 1
            pos_xy = self._np_random.uniform(low=self.world_min, high=self.world_max)
            for planet in self.planets:
                if planet.distance(pos_xy) < 0:
                    break
                gravity = np.linalg.norm(planet.gravity(pos_xy, self.ship.mass))
                if gravity > self.ship.max_engine_force:
                    break
            else:
                break
        pos_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.normal(size=2) * 10
        velocity_angle = 0.0
        return np.array([*pos_xy, pos_angle, *velocities_xy, velocity_angle])


class SpaceshipLandDiscreteEnv(SpaceshipLandEnv, DiscreteSpaceshipEnv):
    pass


class SpaceshipLandContinuousEnv(SpaceshipLandEnv, ContinuousSpaceshipEnv):
    pass
