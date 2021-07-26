from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector, G
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class KeplerEnv(SpaceshipEnv, ABC):
    _planet_radius = 0.25
    _border_radius = 2.0

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

    def _orbit_vel(self, pos_xy):
        alpha = G * (self.ship_params.mass + self.planets[0].mass)
        r = np.linalg.norm(pos_xy)
        return np.sqrt(alpha / r)


    def _orbit_reference_vel(self, pos_xy, ecc = 1, curl = 1) -> np.array:
        """for given (x,y) ship position compute the reference
        tangential component of the velocity, such that ship
        will enter the circle orbit (fixed radius r) around
        the planet (Kepler problem)
        """
        Vt = np.zeros((2,))

        dir = np.sign(pos_xy[1]) * curl
        Vt[0] = dir
        Vt[1] = -dir * pos_xy[0] / pos_xy[1]
    
        Vt = Vt * self._orbit_vel(pos_xy) / np.linalg.norm(Vt)
        return Vt

    def _sparse_reward(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward added if pos_xy and vel_xy is sufficiently close to the reference"""
        final_r = 0.0
        pos_r = np.linalg.norm(pos_xy)

        if np.abs(pos_r - radius_ref) < self.sparse_r_thresh:
            final_r += self.rad_reward_value
        if np.abs(pos_xy.dot(vel_xy)) < 0.01:
            final_r += self.vel_reward_value
        orbit_vel = self._orbit_vel(pos_xy)
        if np.abs(orbit_vel - np.linalg.norm(vel_xy)) < 0.01:
            final_r += self.vel_reward_value

        max_step_reward = 3.0
        return final_r, max_step_reward

    def _dense_reward(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward increasing with decreasing distance to reference values"""        
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        rad_penalty = np.abs(pos_r - radius_ref)
        vel_dir_penalty = np.abs(pos_xy.dot(vel_xy))
        orbit_vel_penalty = np.abs(orbit_vel - np.linalg.norm(vel_xy))

        reward = C / (rad_penalty + vel_dir_penalty + orbit_vel_penalty + C)
        self.rad_penalty = rad_penalty
        self.vel_dir_penalty = vel_dir_penalty
        self.orbit_vel_penalty = orbit_vel_penalty        
        return reward

    def _dense_reward5(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward increasing with decreasing distance to reference values"""        
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        rad_penalty = np.abs(pos_r - radius_ref)
        vel_ref = self._orbit_reference_vel(pos_xy)
        vel_x_penalty = np.abs(vel_ref[0] - vel_xy[0])
        vel_y_penalty = np.abs(vel_ref[1] - vel_xy[1])
        act_penalty = np.linalg.norm(self.last_action)

        reward = C / (self.rad_penalty_C*rad_penalty + vel_x_penalty + vel_y_penalty + self.act_penalty_C*act_penalty + C)
        self.rad_penalty = rad_penalty
        self.vel_x_penalty = vel_x_penalty
        self.vel_y_penalty = vel_y_penalty
        self.act_penalty = act_penalty   
        return reward



    def _dense_reward2(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """reward increasing with decreasing distance to reference values ,
        does not work well"""
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        rad_penalty = np.abs(pos_r - radius_ref)
        vel_dir_penalty = np.abs(pos_xy.dot(vel_xy))
        orbit_vel_penalty = np.abs(orbit_vel - np.linalg.norm(vel_xy))

        reward = C * (
            1 / (rad_penalty + C)
            + 1 / (vel_dir_penalty + C)
            + 1 / (orbit_vel_penalty + C)
        )
        self.rad_penalty = rad_penalty
        self.vel_dir_penalty = vel_dir_penalty
        self.orbit_vel_penalty = orbit_vel_penalty
        # print(f"d_reward={reward}")
        # print(f"p1_reward={C / (rad_penalty + C)}")
        # print(f"p2_reward={C / (vel_dir_penalty + C)}")
        # print(f"p3_reward={C / (orbit_vel_penalty + C)}")
        return reward

    def _dense_reward3(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """reward increasing with decreasing distance to reference values ,
        does not work well"""
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        rad_penalty = np.abs(pos_r - radius_ref)
        vel_dir_penalty = np.abs(pos_xy.dot(vel_xy))
        orbit_vel_penalty = np.abs(orbit_vel - np.linalg.norm(vel_xy))

        reward = C / (rad_penalty * vel_dir_penalty * orbit_vel_penalty + C)
        self.rad_penalty = rad_penalty
        self.vel_dir_penalty = vel_dir_penalty
        self.orbit_vel_penalty = orbit_vel_penalty
        # print(f"d_reward={reward}")
        # print(f"p1_reward={C / (rad_penalty + C)}")
        # print(f"p2_reward={C / (vel_dir_penalty + C)}")
        # print(f"p3_reward={C / (orbit_vel_penalty + C)}")
        return reward

    def _dense_reward4(self, pos_xy, vel_xy, radius_ref, vel_ref) -> np.array:
        """ reward increasing with decreasing distance to reference values"""        
        C = self.numerator_C
        pos_r = np.linalg.norm(pos_xy)
        orbit_vel = self._orbit_vel(pos_xy)
        rad_penalty = np.abs(pos_r - radius_ref)
        vel_dir_penalty = np.abs(pos_xy.dot(vel_xy))
        orbit_vel_penalty = np.abs(orbit_vel - np.linalg.norm(vel_xy))

        reward = C / (rad_penalty + vel_dir_penalty + orbit_vel_penalty + C)**2
        self.rad_penalty = rad_penalty
        self.vel_dir_penalty = vel_dir_penalty
        self.orbit_vel_penalty = orbit_vel_penalty
        # print(f"d_reward={reward}")
        # print(f"p1_reward={C / (rad_penalty + C)}")
        # print(f"p2_reward={C / (vel_dir_penalty + C)}")
        # print(f"p3_reward={C / (orbit_vel_penalty + C)}")
        return reward



    def _reward(self) -> float:
        pos_xy = self._ship_state.pos_xy
        vel_xy = self._ship_state.vel_xy
        vel_ref = self._orbit_reference_vel(pos_xy)
        
        dense_r = self._dense_reward5(pos_xy, vel_xy, self.ref_orbit_radius, vel_ref)

        return dense_r

    def _make_observation(self):
        super()._make_observation()
        #dodac parametry orbity docelowej do self.observation

    def __init__(
        self,
        test_env=False,
        ref_orbit_radius=1.5,
        ref_orbit_eccentricity=1.0,
        sparse_vel_thresh=0.1,
        sparse_r_thresh=0.1,
        reward_value=1.0,
        vel_reward_value=1.0,
        rad_reward_value=1.0,
        numerator_C = 0.01,
        rad_penalty_C = 1.,
        act_penalty_C = 1.
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
            step_size=0.2,
            vel_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False,
            renderer_kwargs={"num_prev_pos_vis": 50, "prev_pos_color_decay": 0.95},
        )
        self.test_env = test_env
        self.ref_orbit_radius = ref_orbit_radius
        self.reward_value = reward_value
        self.ref_orbit_eccentricity = ref_orbit_eccentricity
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
        velocities_xy = self._np_random.standard_normal(2) * 0.05
        # velocities_xy = self._circle_orbit_reference_vel(pos_xy)

        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 5
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)

        self._ship_state.set(pos_xy, ship_angle, velocities_xy, angular_velocity)


class KeplerDiscreteEnv(KeplerEnv, DiscreteSpaceshipEnv):
    pass


class KeplerContinuousEnv(KeplerEnv, ContinuousSpaceshipEnv):
    pass