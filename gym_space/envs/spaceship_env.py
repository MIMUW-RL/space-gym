import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.action_space import ActionSpaceType
from gym_space.rewards import Rewards
from typing import List
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

G = 6.6743e-11


class SpaceshipEnv(gym.Env):
    """Spaceship starts away from any planet, it's goal is to land on planet nr 0"""

    def __init__(
        self,
        ship: Ship,
        planets: List[Planet],
        action_space_type: ActionSpaceType,
        reward_specification: Rewards,
        control_thruster: bool = True,
        step_length: float = 0.02
    ):
        self.ship = ship
        self.planets = planets
        self.action_space_type = action_space_type
        self.control_thruster = control_thruster
        self.step_length = step_length
        self.reward_specification = reward_specification
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self.state = None


    def _init_action_space(self):
        if self.action_space_type is ActionSpaceType.DISCRETE:
            # engine can be turned on or off: 2 options
            if self.control_thruster:
                # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
                self.action_space = Discrete(2 * 3)
            else:
                self.action_space = Discrete(2)
        elif self.action_space_type is ActionSpaceType.CONTINUOUS:
            min_engine_force = 0
            min_thruster_torque = -self.ship.max_thruster_torque
            if self.control_thruster:
                self.action_space = Box(
                    low=(min_engine_force, min_thruster_torque),
                    high=(self.ship.max_engine_force, self.ship.max_thruster_torque)
                )
            else:
                self.action_space = Box(
                    low=(min_engine_force,),
                    high=(self.ship.max_engine_force,)
                )

    def _init_boundary_events(self):
        def event(planet: Planet, _t, state):
            ship_xy_pos, _ = state[:2]
            distance_from_planet_center = np.linalg.norm(ship_xy_pos - planet.center_pos)
            return distance_from_planet_center - planet.radius

        self._boundary_events = []
        for planet_ in self.planets:
            event_ = partial(event, planet_)
            event_.terminal = True
            self._boundary_events.append(event_)

    @staticmethod
    def _external_force_and_torque(action: np.array, state: np.array):
        engine_force, thruster_torque = action
        angle = state[2]
        engine_force_direction = np.array([np.cos(angle), np.sin(angle)])
        force_xy = engine_force_direction * engine_force
        return np.concatenate([force_xy, thruster_torque])

    def _vector_field(self, action, _time, state: np.array):
        external_force_and_torque = self._external_force_and_torque(action, state)

        position, velocity = np.split(state, 2)
        external_force_xy, external_torque = np.split(external_force_and_torque, [2])

        force_xy = external_force_xy
        for planet in self.planets:
            distance2 = (position - planet.center_pos)**2
            force_xy += G * self.ship.mass * planet.mass / distance2
        acceleration_xy = force_xy / self.ship.mass

        torque = external_torque
        acceleration_angle = torque / self.ship.moi

        acceleration = np.concatenate(acceleration_xy, acceleration_angle)
        return np.concatenate(acceleration, velocity)

    def _update_state(self, action):
        ode_solution = solve_ivp(
            partial(self._vector_field, action),
            t_span=(0, self.step_length),
            y0=self.state,
            events=self._boundary_events,
        )
        assert ode_solution.success
        self.state = ode_solution.y[:, -1]
        self._normalize_angle()
        done = ode_solution.status == 1
        return done

    def _normalize_angle(self):
        self.state[2] %= 2 * np.pi

    def _discrete_to_continuous_action(self, discrete_action: int):
        engine_active = discrete_action % 2
        engine_action = engine_active * self.ship.max_engine_force
        thruster_action_direction = discrete_action // 2 - 1
        thruster_action = thruster_action_direction * self.ship.max_thruster_torque
        return np.array([engine_action, thruster_action])


    def _fuel_penalty(self, action):
        engine_power = action[0]
        return engine_power * self.fuel_penalty

    def _destination_reward(self):
        reward = 500



    def _reward(self, action):
        reward = 0.0
        engine_power = action[0]
        reward -= engine_power * self.fuel_penalty

    def step(self, action_):
        assert self.action_space.contains(action_)
        if self.action_space_type is ActionSpaceType.DISCRETE:
            action = self._discrete_to_continuous_action(action_)
        else:
            action = action_

        done = self._update_state(action)
        return self.state, self._reward(action), done, {}