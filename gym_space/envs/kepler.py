from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship import Ship
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv
from gym_space.helpers import vector_to_angle

class KeplerEnv(SpaceshipEnv, ABC):
    max_episode_steps = 300
    _planet_radius = 0.25
    _border_radius = 2.0

    C_referencevel = 2.
    reward_value = 1.
    
    def _circle_orbit_reference_vel(self, pos_xy) -> np.array:
        """ for given (x,y) ship position compute the reference 
            tangential component of the velocity, such that ship 
            will enter the circle orbit around planet (Kepler problem)
        """
        Vt = np.zeros((2,))
        Vt[0] = 1.
        Vt[1] = - pos_xy[0] / pos_xy[1]
        Vt = Vt / np.linalg.norm(Vt)       
        #assume that there is a single planet planets[0] 
        alpha = 6.6743e-11 * (self.ship.mass + self.planets[0].mass)
        p = np.linalg.norm(pos_xy)
        Vt = Vt * np.sqrt(alpha / p)
        return Vt

    def _reward(self) -> float:
        internal_state = self.internal_state
        pos_xy = self.get_ship_pos_xy(internal_state)
        vel_xy = self.get_ship_vel_xy(internal_state)

        vel_ref = self._circle_orbit_reference_vel(pos_xy)

        return self.reward_value  - self.C_referencevel * ( (vel_xy[0] - vel_ref[0])**2 + (vel_xy[1] - vel_ref[1])**2 )

    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=6e8, radius=self._planet_radius)
        # here we use planet outline as external border, i.e. we fly "inside planet"
        border = Planet(center_pos=np.zeros(2), mass=0.0, radius=self._border_radius)
        ship = Ship(mass=1, moi=0.05, max_engine_force=0.3, max_thruster_torque=0.05)

        super().__init__(
            ship=ship,
            planets=[planet, border],            
            world_size=2 * self._border_radius,
            step_size=0.2,
            max_abs_angular_velocity=5.0,
            velocity_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False,
            renderer_kwargs = {'num_prev_pos_vis': 50, 'prev_pos_color_decay': 0.95},
        )

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(self._planet_radius + 0.2, self._border_radius - 0.15)
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        #construct initial velocity for circle Kepler orbit
        '''
        Vt = np.zeros((2,))
        Vt[0] = 1.
        Vt[1] = - pos_xy[0] / pos_xy[1]
        Vt = Vt / np.linalg.norm(Vt)        
        alpha = 6.6743e-11 * (self.ship.mass + self.planets[0].mass)
        p = np.linalg.norm(pos_xy)
        Vt = Vt * np.sqrt(alpha / p)
        velocities_xy = Vt
        '''        
        
        max_abs_ang_vel = 0.7 * self.max_abs_angular_velocity
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self.internal_state = np.array([*pos_xy, ship_angle, *velocities_xy, angular_velocity])

class KeplerDiscreteEnv(KeplerEnv, DiscreteSpaceshipEnv):
    pass


class KeplerContinuousEnv(KeplerEnv, ContinuousSpaceshipEnv):
    pass