from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class DoNotCrashEnv(SpaceshipEnv, ABC):
    max_episode_steps = 300
    _planet_radius = 0.25
    _border_radius = 1.0

    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=6e8, radius=self._planet_radius)
        # here we use planet outline as external border, i.e. we fly "inside planet"
        border = Planet(center_pos=np.zeros(2), mass=0.0, radius=self._border_radius)
        ship = ShipParams(
            mass=1, moi=0.05, max_engine_force=0.3, max_thruster_force=0.05
        )

        super().__init__(
            ship_params=ship,
            planets=[planet, border],
            world_size=2 * self._border_radius,
            step_size=0.07,
            max_abs_vel_angle=5.0,
            vel_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False,
        )

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(
            self._planet_radius + 0.2, self._border_radius - 0.15
        )
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self._ship_state.set(pos_xy, ship_angle, velocities_xy, angular_velocity)

    def _reward(self) -> float:
        return 100 / self.max_episode_steps


class DoNotCrashDiscreteEnv(DoNotCrashEnv, DiscreteSpaceshipEnv):
    pass


class DoNotCrashContinuousEnv(DoNotCrashEnv, ContinuousSpaceshipEnv):
    pass
