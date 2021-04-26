from dataclasses import dataclass
import numpy as np
from typing import List


# gravitational constant
G = 6.6743e-11


@dataclass
class Planet:
    center_pos: np.array
    mass: float
    radius: float

    def distance(self, pos: np.array):
        return np.linalg.norm(pos - self.center_pos) - self.radius

    def gravity(self, pos: np.array, mass: float):
        pos_diff = self.center_pos - pos
        center_distance = np.linalg.norm(pos_diff)
        force_direction = pos_diff / center_distance
        scalar_force = G * mass * self.mass / center_distance ** 2
        return force_direction * scalar_force

def planets_min_max(planets: List[Planet]):
    min_ = np.full(2, np.inf)
    max_ = np.full(2, -np.inf)
    for planet in planets:
        min_ = np.minimum(min_, planet.center_pos - planet.radius)
        max_ = np.maximum(max_, planet.center_pos + planet.radius)
    return min_, max_