from dataclasses import dataclass
from .planet import Planet
from .helpers import bounded_linear, bounded_square, angle_to_unit_vector, vector_to_angle
import numpy as np

class Rewards:
    def reward(self, state: np.array, action: np.array, done: bool):
        raise NotImplementedError


class NoRewards(Rewards):
    def reward(self, state: np.array, action: np.array, done: bool):
        return 0.0


@dataclass
class OrbitPlanetRewards(Rewards):
    planet: Planet
    step_size: float

    def reward(self, state: np.array, action: np.array, done: bool):
        ship_xy = state[:2]
        ship_planet_angle = vector_to_angle(ship_xy - self.planet.center_pos)
        ship_velocity_xy = state[3:5]
        ship_velocity_xy_angle = vector_to_angle(ship_velocity_xy)
        angle_diff = (ship_velocity_xy_angle - ship_planet_angle) % (2 * np.pi)
        angular_velocity_around_planet = np.sin(angle_diff) * np.linalg.norm(ship_velocity_xy)
        return angular_velocity_around_planet / (self.step_size * 1e3)

@dataclass
class LandOnPlanetRewards(Rewards):
    destination_reward: float
    max_destination_distance_penalty: float
    max_reasonable_distance: float
    max_fuel_penalty: float
    max_landing_velocity_penalty: float
    max_landing_angle_penalty: float
    max_episode_steps: int
    destination_planet: Planet = None

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
        penalty_bounded_by_one = bounded_linear(destination_distance, self.max_reasonable_distance) / self.max_reasonable_distance
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
