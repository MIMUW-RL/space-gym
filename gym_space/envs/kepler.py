from abc import ABC
import numpy as np

from gym_space.helpers import (
    angle_to_unit_vector,
    G,
    vector_to_angle,
    angle_to_unit_vector,
)
from gym_space.planet import Planet
from gym.spaces import Box
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class KeplerEnv(SpaceshipEnv, ABC):
    _planet_radius = 0.2
    _border_radius = 3.0

    def _H(self, pos_xy, vel_xy) -> float:
        """Hamiltonian , not used currently, can potentially add to observation"""
        E = 0.5 * (
            vel_xy[0] * vel_xy[0] + vel_xy[1] * vel_xy[1]
        ) / self.ship_params.mass - G * self.planets[
            0
        ].mass * self.ship_params.mass / np.linalg.norm(
            pos_xy
        )
        return E

    def _A(self, pos_xy, vel_xy) -> np.array:
        """the Laplace-Runge-Lenz-vector (conservation law for orbits),
        not used currently, can potentially add to observation"""
        L = pos_xy[0] * vel_xy[1] - pos_xy[1] * vel_xy[0]
        A = np.zeros((2,))
        pos_xy_n = pos_xy / np.linalg.norm(pos_xy)
        m = self.ship_params.mass
        M = self.planets[0].mass
        A[0] = vel_xy[1] * L - m * G * m * M * pos_xy_n[0]
        A[1] = -vel_xy[0] * L - m * G * m * M * pos_xy_n[1]
        return A

    def _b(self, a, ecc):
        """semi-minor axis of the ellipse"""
        return np.sqrt(a * a * (1 - ecc * ecc))

    def _c(self, a, b):
        """distance of a focal point from the ellipse centre """
        return np.sqrt(a * a - b * b)

    def _rotate(self, pos_xy, alpha):
        R = np.zeros((2, 2))
        R[0, 0] = np.cos(alpha)
        R[0, 1] = np.sin(alpha)
        R[1, 0] = -np.sin(alpha)
        R[1, 1] = np.cos(alpha)
        pos_wz = R.dot(pos_xy)
        return pos_wz

    def _orbit_vel(self, r, ref_a):
        alpha = G * self.planets[0].mass
        return np.sqrt(alpha * (2 / r - 1 / ref_a))

    def _orbit_target_vel(self, pos_xy, ref_angle, ref_a, ecc=0, curl=1) -> np.array:
        """for given (x,y) ship position compute the reference
        tangential component of the velocity, such that ship
        will enter the orbit (characterized by ref_angle, ref_a, eccentricity)
        round the planet (Kepler problem)
        """
        # position in rotated coordinates by the reference angle
        Vt = np.zeros((2,))
        a = ref_a
        b = self._b(a, ecc)
        pos_wz = self._rotate(pos_xy, ref_angle)
        c = self._c(a, b)
        pos_wz[0] = pos_wz[0] - c
        theta = vector_to_angle(pos_wz)
        # project pos_wz onto orbit to compute the target velocity
        target_rad = b / np.sqrt(1 - (ecc * np.cos(theta)) ** 2)
        pos_wz = pos_wz * target_rad / np.linalg.norm(pos_wz)
        Vt[0] = -curl * a / b * pos_wz[1]
        Vt[1] = curl * b / a * pos_wz[0]
        # orbit velocity r = distance between orbiting bodies

        r = np.linalg.norm(pos_wz + np.array([c, 0]))
        Vt = Vt * self._orbit_vel(r, a) / (np.linalg.norm(Vt))
        Vt = self._rotate(Vt, -ref_angle)
        return Vt

    def _orbit_cur_rad(self, pos_xy, ref_angle, ref_a, ecc):
        a = ref_a
        b = self._b(a, ecc)
        c = self._c(a, b)
        pos_wz = self._rotate(pos_xy, ref_angle)
        pos_wz[0] = pos_wz[0] - c
        return np.linalg.norm(pos_wz)

    def _orbit_target_rad(self, pos_xy, ref_angle, ref_a, ecc) -> np.array:
        """ get the radius for the current angle theta for the orbit having given ecc."""
        a = ref_a
        b = self._b(a, ecc)
        c = self._c(a, b)

        pos_wz = self._rotate(pos_xy, ref_angle)
        pos_wz[0] = pos_wz[0] - c
        theta = vector_to_angle(pos_wz)
        # return np.sqrt((a*np.cos(theta))**2 + (b*np.sin(theta))**2)
        # for some reason the one below is more precise
        return b / np.sqrt(1 - (ecc * np.cos(theta)) ** 2)

    def _dense_reward5(self, pos_xy, vel_xy) -> np.array:
        """
        reward increasing with decreasing distance to the reference orbit values
        there is still room for improvement
        """
        C = self.numerator_C
        cur_rad = self._orbit_cur_rad(
            pos_xy, self.ref_orbit_angle, self.ref_orbit_a, self.ref_orbit_eccentricity
        )

        target_vel = self._orbit_target_vel(
            pos_xy,
            self.ref_orbit_angle,
            self.ref_orbit_a,
            self.ref_orbit_eccentricity,
        )

        target_rad = self._orbit_target_rad(
            pos_xy,
            self.ref_orbit_angle,
            self.ref_orbit_a,
            self.ref_orbit_eccentricity,
        )
        rad_penalty = np.abs(cur_rad - target_rad)
        vel_x_penalty = np.abs(target_vel[0] - vel_xy[0])
        vel_y_penalty = np.abs(target_vel[1] - vel_xy[1])
        act_penalty = np.linalg.norm(self.last_action)

        reward = C / (
            self.rad_penalty_C * rad_penalty
            + vel_x_penalty
            + vel_y_penalty
            + self.act_penalty_C * act_penalty
            + C
        )
        self.rad_penalty = rad_penalty
        self.vel_x_penalty = vel_x_penalty
        self.vel_y_penalty = vel_y_penalty
        self.act_penalty = act_penalty
        return reward

    def _reward(self) -> float:
        pos_xy = self._ship_state.pos_xy
        vel_xy = self._ship_state.vel_xy
        dense_r = self._dense_reward5(pos_xy, vel_xy)
        return dense_r

    # def _init_observation_space(self):
    #     obs_low = [-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -1.0, 0, 0, 1]
    #     obs_high = [1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0, 2 * np.pi, 0.7, 2]
    #     if self.with_lidar:
    #         # as normalized world is [-1, 1]^2, the highest distance between two points is 2 sqrt(2)
    #         # (x, y) vector for each planet
    #         obs_high += 2 * len(self.planets) * [2 * np.sqrt(2)]
    #         if self.with_goal:
    #             obs_high += 2 * [2 * np.sqrt(2)]
    #     obs_high = np.array(obs_high)
    #     self.observation_space = Box(low=-obs_high, high=obs_high)

    def _make_observation(self):
        super()._make_observation()
        # add target orbit parameters into observation
        # observation = self.observation
        # self.observation = np.concatenate(
        #     [
        #         observation,
        #         np.array(
        #             [
        #                 self.ref_orbit_angle,
        #                 self.ref_orbit_eccentricity,
        #                 self.ref_orbit_a,
        #             ]
        #         ),
        #     ]
        # )

    def __init__(
        self,
        randomize=False,
        ref_orbit_a=1.2,
        ref_orbit_eccentricity=0.5,
        ref_orbit_angle=3.75,
        reward_value=0,
        numerator_C=0.01,
        rad_penalty_C=2.0,
        act_penalty_C=0.5,
        step_size=0.1,
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
            max_abs_vel_angle=2,
            step_size=step_size,
            vel_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False,
            renderer_kwargs={"num_prev_pos_vis": 75, "prev_pos_color_decay": 0.95},
        )
        self.ref_orbit_a = ref_orbit_a
        self.reward_value = reward_value
        self.ref_orbit_eccentricity = ref_orbit_eccentricity
        self.ref_orbit_angle = ref_orbit_angle
        self.numerator_C = numerator_C
        self.rad_penalty_C = rad_penalty_C
        self.act_penalty_C = act_penalty_C
        self.randomize = randomize

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(
            self._planet_radius + 0.5, self._border_radius - 0.5
        )
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)

        # set velocity of the target orbit
        # ecc = self.ref_orbit_eccentricity
        # a = self.ref_orbit_a
        # b = self._b(a, ecc)
        # c = self._c(a, b)
        # pos_xy = angle_to_unit_vector(self.ref_orbit_angle) * (a + c)
        # ref_angle = vector_to_angle(pos_xy)
        # self.ref_orbit_angle = ref_angle
        # velocities_xy = self._orbit_target_vel(
        #     pos_xy,
        #     self.ref_orbit_angle,
        #     self.ref_orbit_a,
        #     self.ref_orbit_eccentricity,
        # )

        # reset goal orbits if randomize is on, random eccentricity and angle
        if self.randomize:
            self.ref_orbit_eccentricity = np.random.uniform() * 0.7
            self.ref_orbit_angle = np.random.uniform() * 2 * np.pi

        velocities_xy = self._np_random.standard_normal(2) * 0.05

        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 5
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)

        self._ship_state.set(pos_xy, ship_angle, velocities_xy, angular_velocity)


class KeplerDiscreteEnv(KeplerEnv, DiscreteSpaceshipEnv):
    pass


class KeplerContinuousEnv(KeplerEnv, ContinuousSpaceshipEnv):
    pass