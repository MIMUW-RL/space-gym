from abc import ABC
import numpy as np
import torch

from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.rewards import ConstantRewards
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class Hover1DEnv(SpaceshipEnv, ABC):
    max_height = 3.0
    step_size = 15.0

    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=5e7, radius=10.0)
        ship = Ship(
            mass=0.1,
            moi=1.0,
            max_engine_force=6e-6,
            max_thruster_torque=0.0,
        )

        state_mean = np.zeros(6)
        # y position
        state_mean[1] = planet.radius + self.max_height / 2
        # angle position
        state_mean[2] = 1.5 * np.pi
        # y velocity
        state_mean[4] = 1e-3

        state_std = np.ones(6)
        # # y position
        # state_std[1] = max_height / 2
        # y velocity
        state_std[4] = 2e-3

        super().__init__(
            ship=ship,
            planets=[planet],
            rewards=ConstantRewards(100 / self.max_episode_steps),
            world_min=np.array([-0.5, planet.radius - 0.3]),
            world_max=np.array([0.5, planet.radius + self.max_height]),
            state_mean=state_mean,
            state_std=state_std,
        )

    def _sample_initial_state(self):
        height_above_planet = self._np_random.uniform(0.0, self.max_height)
        y = self.planets[0].radius + height_above_planet
        angle = 1.5 * np.pi
        return np.array([0.0, y, angle, 0.0, 0.0, 0.0])

    def termination_fn(
        self, _act: torch.Tensor, normalized_next_obs: torch.Tensor
    ) -> torch.Tensor:
        assert len(normalized_next_obs.shape) == 2

        next_obs = self.denormalize_state(normalized_next_obs)
        height_above_planet = next_obs[:, 1] - self.planets[0].radius
        done = (height_above_planet >= 0) * (height_above_planet <= self.max_height)
        done = done[:, None]
        return done


class Hover1DDiscreteEnv(Hover1DEnv, DiscreteSpaceshipEnv):
    pass


class Hover1DContinuousEnv(Hover1DEnv, ContinuousSpaceshipEnv):
    pass
