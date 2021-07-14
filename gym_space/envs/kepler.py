from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector, G
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class KeplerEnv(SpaceshipEnv, ABC):
    _planet_radius = 0.25
    _border_radius = 2.0

    def _energy(self, pos_xy, vel_xy) -> float:
        vel_xy_n = vel_xy / np.linalg.norm(vel_xy)
        E = 0.5 * vel_xy_n * vel_xy_n / self.ship_params.mass - G * self.planets[
            0
        ].mass * self.ship_params.mass / np.linalg.norm(pos_xy)
        return E

    def _A(self, pos_xy, vel_xy) -> np.array:
        """ the Laplace-Runge-Lenz-vector (conservation law)"""
        L = pos_xy[0] * vel_xy[1] - pos_xy[1] * vel_xy[0]
        A = np.zeros((2,))
        pos_xy_n = pos_xy / np.linalg.norm(pos_xy)
        m = self.ship_params.mass
        M = self.planets[0].mass
        A[0] = vel_xy[1] * L - m * G * m * M * pos_xy_n[0]
        A[1] = -vel_xy[0] * L - m * G * m * M * pos_xy_n[1]
        return A

    def _sparse_reward(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward added if pos_xy and vel_xy is sufficiently close to the reference"""
        final_r = 0.0
        pos_r = np.linalg.norm(pos_xy)

        print(f"r={np.abs(pos_r - radius_ref)}")
        print(f"vel={np.abs(vel_xy[0] - vel_ref[0]) + np.abs(vel_xy[1] - vel_ref[1])}")
        if np.abs(pos_r - radius_ref) < self.sparse_r_thresh:
            final_r += 1.0
        if (
            np.abs(vel_xy[0] - vel_ref[0]) + np.abs(vel_xy[1] - vel_ref[1])
            < self.sparse_vel_thresh
        ):
            final_r += 1.0
        print(f"rew={final_r}")
        return final_r

    def _sparse_scaled_reward(self, pos_r, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward added if pos_xy and vel_xy close to the reference and increased when closer"""
        pass

    def _dense_reward(self, pos_r, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward increasing with decreasing distance to reference values"""
        vel_penalty = (vel_xy[0] - vel_ref[0]) ** 2 + (vel_xy[1] - vel_ref[1]) ** 2
        pass

    def _circle_orbit_reference_vel(self, pos_xy, radius=None) -> np.array:
        """for given (x,y) ship position compute the reference
        tangential component of the velocity, such that ship
        will enter the circle orbit (fixed radius r) around
        the planet (Kepler problem)
        """
        Vt = np.zeros((2,))
        if radius is not None:
            pos_xy = (radius * pos_xy) / np.linalg.norm(pos_xy)
        Vt[0] = np.sign(pos_xy[1])
        Vt[1] = -np.sign(pos_xy[1]) * pos_xy[0] / pos_xy[1]

        Vt = Vt / np.linalg.norm(Vt)
        # assume that there is a single planet planets[0]
        alpha = G * (self.ship_params.mass + self.planets[0].mass)
        p = np.linalg.norm(pos_xy)
        Vt = Vt * np.sqrt(alpha / p)
        return Vt

    def _reward(self) -> float:
        pos_xy = self._ship_state.pos_xy
        vel_xy = self._ship_state.vel_xy
        vel_ref = self._circle_orbit_reference_vel(pos_xy)
        # print(vel_xy)
        # print(f"A:{self._A(pos_xy, vel_xy)}")

        sparse_r = self._sparse_reward(
            np.linalg.norm(pos_xy), vel_xy, self.ref_orbit_radius, vel_ref
        )

        return self.reward_value + sparse_r

    def __init__(
        self,
        ref_orbit_radius=1.5,
        reward_value=1.0,
        ref_orbit_eccentricity=1.0,
        sparse_vel_thresh=0.1,
        sparse_r_thresh=0.1,
    ):
        planet = Planet(center_pos=np.zeros(2), mass=6e8, radius=self._planet_radius)
        # here we use planet outline as external border, i.e. we fly "inside planet"
        border = Planet(center_pos=np.zeros(2), mass=0.0, radius=self._border_radius)
        ship_params = ShipParams(
            mass=1, moi=0.05, max_engine_force=0.3, max_thruster_force=0.05
        )

        super().__init__(
            ship_params=ship_params,
            planets=[planet, border],
            world_size=2 * self._border_radius,
            max_abs_vel_angle=1.5,
            step_size=0.2,
            vel_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False,
            renderer_kwargs={"num_prev_pos_vis": 50, "prev_pos_color_decay": 0.95},
        )

        self.ref_orbit_radius = ref_orbit_radius
        self.reward_value = reward_value
        self.ref_orbit_eccentricity = ref_orbit_eccentricity
        self.sparse_r_thresh = sparse_r_thresh
        self.sparse_vel_thresh = sparse_vel_thresh

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(
            self._planet_radius + 0.5, self._border_radius - 0.5
        )
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.05

        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 5
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)

        self._ship_state.set(pos_xy, ship_angle, velocities_xy, angular_velocity)


class KeplerDiscreteEnv(KeplerEnv, DiscreteSpaceshipEnv):
    pass


class KeplerContinuousEnv(KeplerEnv, ContinuousSpaceshipEnv):
    pass