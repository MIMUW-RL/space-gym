from dataclasses import dataclass
from .planet import Planet
import numpy as np

@dataclass
class Rewards:
    destination_planet: Planet
    destination_final_reward: float
    destination_distance_penalty_scale: float
    fuel_penalty_scale: float
    max_bad_landing_penalty: float

    def _destination_distance(self, state):
        ship_xy = state[:2]
        distance_from_planet_center = np.linalg.norm(ship_xy - self.destination_planet.center_pos)
        return distance_from_planet_center - self.destination_planet.radius

    def _destination_distance_penalty(self, state: np.array):
        # TODO: asymptotic bound
        return self._destination_distance(state) * self.destination_distance_penalty_scale

    def _fuel_penalty(self, action):
        engine_power = action[0]
        return engine_power * self.fuel_penalty_scale

    def _bad_landing_penalty(self, state):
        # TODO: asymptotic bound
        ...

    def reward(self, state: np.array, action: np.array, done: bool):
        reward = 0.0
        reward -= self._destination_distance_penalty(state)
        reward -= self._fuel_penalty(action)
        if done and np.isclose(0, self._destination_distance(state)):
            reward += self.destination_final_reward
            reward -= self._bad_landing_penalty(state)
        return reward