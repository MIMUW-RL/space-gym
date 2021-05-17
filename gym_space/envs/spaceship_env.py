from abc import ABC

import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet, planets_min_max
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from gym_space.rendering import Renderer
from gym_space.helpers import angle_to_unit_vector
from typing import List
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

DEFAULT_STEP_SIZE = 36.0
DEFAULT_MAX_EPISODE_STEPS = 1_000

# TODO: seed()
class SpaceshipEnv(gym.Env):
    def __init__(
        self,
        ship: Ship,
        planets: List[Planet],
        rewards: Rewards,
        step_size: float = DEFAULT_STEP_SIZE,
    ):
        self.ship = ship
        self.planets = planets
        self._world_min, self._world_max = None, None
        self._init_world_min_max()
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf, 0.0, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, 2 * np.pi, np.inf, np.inf, np.inf])
        )
        self.rewards = rewards
        self.step_size = step_size
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            # FIXME: it doesn't work
            'video.frames_per_second': 60 / step_size  # one minute in one second
        }
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self.state = None
        self.last_action = None
        self._renderer = None

    def _init_world_min_max(self):
        if len(self.planets) > 1:
            self._world_min, self._world_max = planets_min_max(self.planets)
        else:
            planet = self.planets[0]
            self._world_min = planet.center_pos - 4 * planet.radius
            self._world_max = planet.center_pos + 4 * planet.radius

    def _init_action_space(self):
        raise NotImplementedError

    def _init_boundary_events(self):
        def event(planet: Planet, _t, state):
            ship_xy_pos = state[:2]
            return planet.distance(ship_xy_pos)

        self._boundary_events = []
        for planet_ in self.planets:
            event_ = partial(event, planet_)
            event_.terminal = True
            self._boundary_events.append(event_)

    def _external_force_and_torque(self, action: np.array, state: np.array):
        engine_action, thruster_action = action
        engine_force = engine_action * self.ship.max_engine_force
        thruster_torque = thruster_action * self.ship.max_thruster_torque
        angle = state[2]
        engine_force_direction = - angle_to_unit_vector(angle)
        force_xy = engine_force_direction * engine_force
        return np.array([*force_xy, thruster_torque])

    def _vector_field(self, action, _time, state: np.array):
        external_force_and_torque = self._external_force_and_torque(action, state)

        position, velocity = np.split(state, 2)
        external_force_xy, external_torque = np.split(external_force_and_torque, [2])

        force_xy = external_force_xy
        for planet in self.planets:
            force_xy += planet.gravity(position[:2], self.ship.mass)
        acceleration_xy = force_xy / self.ship.mass

        torque = external_torque
        acceleration_angle = torque / self.ship.moi

        acceleration = np.concatenate([acceleration_xy, acceleration_angle])
        return np.concatenate([velocity, acceleration])

    def _update_state(self, action):
        ode_solution = solve_ivp(
            partial(self._vector_field, action),
            t_span=(0, self.step_size),
            y0=self.state,
            events=self._boundary_events,
        )
        assert ode_solution.success, ode_solution.message
        self.state = ode_solution.y[:, -1]
        self._normalize_angle()
        done = ode_solution.status == 1
        return done

    def _normalize_angle(self):
        self.state[2] %= 2 * np.pi

    @staticmethod
    def _translate_raw_action(raw_action):
        raise NotImplementedError

    def _sample_initial_state(self):
        raise NotImplementedError

    def step(self, raw_action):
        assert self.action_space.contains(raw_action), raw_action
        action = self._translate_raw_action(raw_action)
        self.last_action = action

        done = self._update_state(action)
        reward = self.rewards.reward(self.state, action, done)
        return self.state, reward, done, {}

    def reset(self):
        self.state = self._sample_initial_state()
        return self.state

    def render(self, mode="human"):
        if self._renderer is None:
            self._renderer = Renderer(15, self.planets, self._world_min, self._world_max)

        engine_active = False
        if self.last_action is not None:
            engine_active = self.last_action[0] > 0.1
        return self._renderer.render(self.state[:3], engine_active, mode)

    @property
    def viewer(self):
        return self._renderer.viewer


class DiscreteSpaceshipEnv(SpaceshipEnv):
    def _init_action_space(self):
        # engine can be turned on or off: 2 options
        # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
        self.action_space = Discrete(2 * 3)

    @staticmethod
    def _translate_raw_action(raw_action: int):
        engine_action = float(raw_action % 2)
        thruster_action = float(raw_action // 2 - 1)
        return np.array([engine_action, thruster_action])


class ContinuousSpaceshipEnv(SpaceshipEnv):
    def _init_action_space(self):
        self.action_space = Box(low=-np.ones(2), high=np.ones(2))

    @staticmethod
    def _translate_raw_action(raw_action: np.array):
        engine_action, thruster_action = raw_action
        # [-1, 1] -> [0, 1]
        engine_action = (engine_action + 1) / 2
        return np.array([engine_action, thruster_action])
