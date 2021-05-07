from abc import ABC
from dataclasses import dataclass
import numpy as np

from gym_space.helpers import angle_to_unit_vector, vector_to_angle, bounded_linear, bounded_square
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv, DEFAULT_STEP_SIZE


@dataclass
class GoToPlanetRewards(Rewards):
    step_size: float = DEFAULT_STEP_SIZE
    ship_position_penalty_scale: float = 1e-3
    angular_velocity_penalty_scale: float = 5e-3

    def _ship_position_penalty(self, state: np.array):
        def unbounded_penalty(angle: float):
            assert 0.0 <= angle <= 2 * np.pi
            # [0, 2pi] -> [-pi, pi]
            angle -= np.pi
            # [-pi, pi] -> [-pi / 2, pi / 2]
            angle /= 2
            return np.tan(angle) ** 2
        planet_ship_vector = state[:2]
        planet_ship_angle = vector_to_angle(planet_ship_vector)
        ship_angle = state[2]
        angle_diff = (planet_ship_angle - ship_angle) % (2 * np.pi)
        penalty = unbounded_penalty(angle_diff)
        return bounded_linear(penalty, 100.0) * self.ship_position_penalty_scale

    def _angular_velocity_penalty(self, state: np.array):
        return bounded_square(5e3 * state[5], 100.0) * self.angular_velocity_penalty_scale / self.step_size

    def _velocity_reward(self, state: np.array):
        point_ship_vector = state[:2]
        unit_ship_point_vec = - point_ship_vector / np.linalg.norm(point_ship_vector)
        velocity_vector = state[3:5]
        return velocity_vector @ unit_ship_point_vec * 100 / self.step_size

    def reward(self, state: np.array, action: np.array, done: bool):
        reward = self._velocity_reward(state)
        reward -= self._ship_position_penalty(state)
        reward -= self._angular_velocity_penalty(state)
        reward -= 1e-2
        if done:
            reward += 10.0
        return reward


class SpaceshipGoToPlanetEnv(SpaceshipEnv, ABC):
    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=2e4, radius=2)
        ship = Ship(mass=1.0, moi=1.0, max_engine_force=1e-6, max_thruster_torque=1e-6)

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=GoToPlanetRewards(),
        )
        self._world_min = np.full(2, -15.0)
        self._world_max = np.full(2, 15.0)

    def _sample_initial_state(self):
        point_angle, ship_angle = np.random.uniform(0, 2 * np.pi, size=2)
        pos_xy = angle_to_unit_vector(point_angle) * 7.5
        velocities = np.zeros(3)
        return np.array([*pos_xy, ship_angle, *velocities])


class SpaceshipGoToPlanetDiscreteEnv(SpaceshipGoToPlanetEnv, DiscreteSpaceshipEnv):
    pass


class SpaceshipGoToPlanetContinuousEnv(SpaceshipGoToPlanetEnv, ContinuousSpaceshipEnv):
    pass