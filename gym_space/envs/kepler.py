from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector, G, vector_to_angle
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class KeplerEnv(SpaceshipEnv, ABC):
    _planet_radius = 0.2
    _border_radius = 3.0

    def _H(self, pos_xy, vel_xy) -> float:
        """Hamiltonian """
        E = 0.5 * (
            vel_xy[0] * vel_xy[0] + vel_xy[1] * vel_xy[1]
        ) / self.ship_params.mass - G * self.planets[
            0
        ].mass * self.ship_params.mass / np.linalg.norm(
            pos_xy
        )
        return E

    def _L(self, pos_xy, vel_xy) -> float:
        return pos_xy[0] * vel_xy[1] - pos_xy[1] * vel_xy[0]

    def _A(self, pos_xy, vel_xy) -> np.array:
        """ the Laplace-Runge-Lenz-vector (conservation law)"""
        L = self._L(pos_xy, vel_xy)
        A = np.zeros((2,))
        pos_xy_n = pos_xy / np.linalg.norm(pos_xy)
        m = self.ship_params.mass
        M = self.planets[0].mass
        A[0] = vel_xy[1] * L - m * G * m * M * pos_xy_n[0]
        A[1] = -vel_xy[0] * L - m * G * m * M * pos_xy_n[1]
        return A

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
        will enter the circle orbit (fixed radius r) around
        the planet (Kepler problem)
        """
        # position in rotated coordinates by the reference angle
        Vt = np.zeros((2,))
        a = ref_a
        print(f"ecc={ecc}")
        b = np.sqrt(a * a * (1 - ecc * ecc))

        pos_wz = self._rotate(pos_xy, ref_angle)
        print(f"angle={ref_angle}")

        c = np.sqrt(a * a - b * b)
        pos_wz[0] = pos_wz[0] - c
        print(f"pos_xy={pos_xy}, pos_wz={pos_wz}")
        print(f"c={c}")
        theta = vector_to_angle(pos_wz)
        print(f"theta={theta}")
        print(f"a={a} rad={np.linalg.norm(pos_wz)}")
        dir = np.sign(pos_wz[1]) * curl

        Vt[0] = -a / b * pos_wz[1]
        Vt[1] = b / a * pos_wz[0]

        # orbit velocity r = distance between orbiting bodies
        r = np.linalg.norm(pos_xy)
        Vt = Vt * self._orbit_vel(r, a) / (np.linalg.norm(Vt))

        Vt = self._rotate(Vt, -ref_angle)
        print(f"Vt={Vt}")

        return Vt

    def _orbit_target_rad(self, pos_xy, ref_angle, ref_a, ecc) -> np.array:
        """ get the radius for the current angle theta for the orbit having given ecc."""

        a = ref_a
        b = np.sqrt(a * a * (1 - ecc * ecc))

        pos_wz = self._rotate(pos_xy, ref_angle)
        c = np.sqrt(a * a - b * b)
        pos_wz[0] = pos_wz[0] + c
        theta = vector_to_angle(pos_wz)
        # in this coordinates radius is equal to (asin(theta))^2+(bcos(theta))^2
        return a / (1 + ecc * np.cos(theta))

    def _sparse_reward(self, pos_xy, vel_xy, ref_radius, vel_ref) -> np.array:
        """ reward added if pos_xy and vel_xy is sufficiently close to the reference"""
        final_r = 0.0
        pos_r = np.linalg.norm(pos_xy)
        if np.abs(pos_r - ref_radius) < self.sparse_r_thresh:
            final_r += self.rad_reward_value
        if np.abs(pos_xy.dot(vel_xy)) < 0.01:
            final_r += self.vel_reward_value
        orbit_vel = self._orbit_vel(pos_xy)
        if np.abs(orbit_vel - np.linalg.norm(vel_xy)) < 0.01:
            final_r += self.vel_reward_value
        max_step_reward = 3.0
        return final_r, max_step_reward

    def _dense_reward(self, pos_xy, vel_xy) -> np.array:
        """reward increasing with decreasing distance to reference values,
        using tangent_vel (zero scalar prod) and orbit_vel condition"""
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        target_rad = self._orbit_target_rad(
            pos_xy, self.ref_orbit_a, self.ref_orbit_eccentricity
        )
        rad_penalty = np.abs(pos_r - target_rad)
        vel_dir_penalty = np.abs(pos_xy.dot(vel_xy))
        orbit_vel_penalty = np.abs(orbit_vel - np.linalg.norm(vel_xy))

        reward = C / (rad_penalty + vel_dir_penalty + orbit_vel_penalty + C)
        self.rad_penalty = rad_penalty
        self.vel_dir_penalty = vel_dir_penalty
        self.orbit_vel_penalty = orbit_vel_penalty
        return reward

    def _dense_reward5(self, pos_xy, vel_xy) -> np.array:
        """ reward increasing with decreasing distance to reference values"""
        C = self.numerator_C
        cur_rad = np.linalg.norm(pos_xy)

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
        print(f"vel_xy={vel_xy}")
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

    def _make_observation(self):
        super()._make_observation()
        # dodac parametry orbity docelowej do self.observation

    def __init__(
        self,
        test_env=False,
        ref_orbit_a=1.2,
        ref_orbit_eccentricity=0.5,
        ref_orbit_angle=0,
        sparse_vel_thresh=0.1,
        sparse_r_thresh=0.1,
        reward_value=1.0,
        vel_reward_value=1.0,
        rad_reward_value=1.0,
        numerator_C=0.01,
        rad_penalty_C=1.0,
        act_penalty_C=1.0,
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
            renderer_kwargs={"num_prev_pos_vis": 50, "prev_pos_color_decay": 0.95},
        )
        self.test_env = test_env
        self.ref_orbit_a = ref_orbit_a
        self.reward_value = reward_value
        self.ref_orbit_eccentricity = ref_orbit_eccentricity
        self.ref_orbit_angle = ref_orbit_angle
        self.sparse_r_thresh = sparse_r_thresh
        self.sparse_vel_thresh = sparse_vel_thresh
        self.vel_reward_value = vel_reward_value
        self.rad_reward_value = rad_reward_value
        self.numerator_C = numerator_C
        self.rad_penalty_C = rad_penalty_C
        self.act_penalty_C = act_penalty_C

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(
            self._planet_radius + 0.5, self._border_radius - 0.5
        )
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        # velocities_xy = self._np_random.standard_normal(2) * 0.05
        ecc = 0.8
        a = self.ref_orbit_a
        b = np.sqrt(a * a * (1 - ecc * ecc))
        c = np.sqrt(a * a - b * b)
        pos_xy = pos_xy / np.linalg.norm(pos_xy) * (a + c)
        ref_angle = vector_to_angle(pos_xy)
        self.ref_orbit_angle = ref_angle

        velocities_xy = self._orbit_target_vel(
            pos_xy,
            self.ref_orbit_angle,
            self.ref_orbit_a,
            self.ref_orbit_eccentricity,
        )

        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 5
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)

        self._ship_state.set(pos_xy, ship_angle, velocities_xy, angular_velocity)


class KeplerDiscreteEnv(KeplerEnv, DiscreteSpaceshipEnv):
    pass


class KeplerContinuousEnv(KeplerEnv, ContinuousSpaceshipEnv):
    pass