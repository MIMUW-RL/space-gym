from abc import ABC
import numpy as np

from gym_space.helpers import angle_to_unit_vector
from gym_space.planet import Planet
from gym_space.ship import Ship
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv
from gym_space.helpers import vector_to_angle

class OrbitEnv(ContinuousSpaceshipEnv):
    max_episode_steps = 300
    _planet_radius = 0.25
    _border_radius = 1.0

    C_xy = 0.2
    C_rotationvel = 2.
    C_action = 0.1
    reward_value = 1.

    #define reward function for OrbitEnv, reward defining constants are are defined in-class
    #to tune them
    def reward( self, action, prev_state ):
        state = self.external_state
        
        prev_xy = prev_state[:2]
        xy = state[:2]

        prev_xy = prev_xy / np.linalg.norm(prev_xy)
        xy = xy / np.linalg.norm(xy)
        angle_diff = np.arccos( prev_xy.dot(xy) )        
        angle_diff = np.nan_to_num(angle_diff)

        vel_xy = np.linalg.norm(state[3:-1])
        #print(f"{prev_xy} {xy} {prev_xy.dot(xy)} {angle_diff}")
        rotationvel = np.abs(angle_diff)        

        action_n = np.linalg.norm(action)

        return self.reward_value + self.C_rotationvel * rotationvel - self.C_action * action_n

    def _reward(self):
        return self.reward_value

    def __init__(self):
        planet = Planet(center_pos=np.zeros(2), mass=6e8, radius=self._planet_radius)
        # here we use planet outline as external border, i.e. we fly "inside planet"
        border = Planet(center_pos=np.zeros(2), mass=0.0, radius=self._border_radius)
        ship = Ship(mass=1, moi=0.05, max_engine_force=0.3, max_thruster_torque=0.05)

        super().__init__(
            ship=ship,
            planets=[planet, border],            
            world_size=2 * self._border_radius,
            step_size=0.07,
            max_abs_angular_velocity=5.0,
            velocity_xy_std=np.ones(2),
            with_lidar=False,
            with_goal=False
        )

    def _reset(self):
        planet_angle = self._np_random.uniform(0, 2 * np.pi)
        ship_planet_center_distance = self._np_random.uniform(self._planet_radius + 0.2, self._border_radius - 0.15)
        pos_xy = angle_to_unit_vector(planet_angle) * ship_planet_center_distance
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_angular_velocity
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self.internal_state = np.array([*pos_xy, ship_angle, *velocities_xy, angular_velocity])