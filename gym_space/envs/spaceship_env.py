from abc import ABC, abstractmethod

import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet, planets_min_max
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from gym_space.helpers import angle_to_unit_vector
from typing import List
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial


class SpaceshipEnv(gym.Env, ABC):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(
        self,
        *,
        ship: Ship,
        planets: List[Planet],
        rewards: Rewards,
        world_size: np.array,
        step_size: float,
        max_abs_angular_velocity: float,
        velocity_xy_std: np.array,
        max_episode_steps: int
    ):
        self.ship = ship
        self.planets = planets
        self.rewards = rewards
        # TODO: use typing with runtime check
        assert world_size.shape == (2,)
        self.world_size = world_size
        self.step_size = step_size
        self.max_abs_angular_velocity = max_abs_angular_velocity
        # TODO: use typing with runtime check
        assert velocity_xy_std.shape == (2,)
        self.velocity_xy_std = velocity_xy_std
        self.max_episode_steps = max_episode_steps

        self.observation_space = Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -1.0]
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0]
            ),
        )
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self._np_random = None
        self.seed()
        self.internal_state = self.last_action = self.elapsed_steps = self._renderer = None

    def reset(self):
        self.internal_state = self._sample_initial_state()
        self.elapsed_steps = 0
        return self.external_state

    def step(self, raw_action):
        assert (
            self.elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        assert self.action_space.contains(raw_action), raw_action
        action = self._translate_raw_action(raw_action)
        self.last_action = action

        done = self._update_state(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
        reward = self.rewards.reward(self.internal_state, action)
        return self.external_state, reward, done, {}

    def render(self, mode="human"):
        if self._renderer is None:
            from gym_space.rendering import Renderer

            self._renderer = Renderer(15, self.planets, self.world_size)

        return self._renderer.render(self.internal_state[:3], self.last_action, mode)

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _init_boundary_events(self):
        self._boundary_events = []

        def planet_event(planet: Planet, _t, state):
            ship_xy_pos = state[:2]
            return planet.distance(ship_xy_pos)

        for planet_ in self.planets:
            event = partial(planet_event, planet_)
            event.terminal = True
            self._boundary_events.append(event)

        def world_max_event(_t, state):
            return np.min(self.world_size / 2 - state[:2])

        world_max_event.terminal = True
        self._boundary_events.append(world_max_event)

        def world_min_event(_t, state):
            return np.min(state[:2] - self.world_size / 2)

        world_min_event.terminal = True
        self._boundary_events.append(world_min_event)

        def angular_velocity_event(_t, state):
            return self.max_abs_angular_velocity - np.abs(state[5])

        angular_velocity_event.terminal = True
        self._boundary_events.append(angular_velocity_event)

    def _external_force_and_torque(self, action: np.array, state: np.array):
        engine_action, thruster_action = action
        engine_force = engine_action * self.ship.max_engine_force
        thruster_torque = thruster_action * self.ship.max_thruster_torque
        angle = state[2]
        engine_force_direction = -angle_to_unit_vector(angle)
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
            y0=self.internal_state,
            events=self._boundary_events,
        )
        assert ode_solution.success, ode_solution.message
        self.internal_state = ode_solution.y[:, -1]
        self._wrap_angle()
        done = ode_solution.status == 1
        return done

    def _wrap_angle(self):
        self.internal_state[2] %= 2 * np.pi

    @property
    def external_state(self):
        external_state = self.internal_state.copy()
        # make sure that x and y positions are between -1 and 1
        external_state[:2] /= self.world_size
        # normalize translational velocity
        external_state[3:5] /= self.velocity_xy_std
        # make sure that angular velocity is between -1 and 1
        external_state[5] /= self.max_abs_angular_velocity
        # represent angle as cosine and sine
        angle = external_state[2]
        angle_repr = np.array([np.cos(angle), np.sin(angle)])
        external_state = np.concatenate([external_state[:2], angle_repr, external_state[3:]])

        return external_state

    @property
    def viewer(self):
        return self._renderer.viewer

    @abstractmethod
    def _init_action_space(self):
        pass

    @staticmethod
    @abstractmethod
    def _translate_raw_action(raw_action):
        # different for discrete and continuous environments
        pass

    @abstractmethod
    def _sample_initial_state(self):
        pass


class DiscreteSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        # engine can be turned on or off: 2 options
        # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
        self.action_space = Discrete(2 * 3)

    @staticmethod
    def _translate_raw_action(raw_action: int):
        engine_action = float(raw_action % 2)
        thruster_action = float(raw_action // 2 - 1)
        return np.array([engine_action, thruster_action])


class ContinuousSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        self.action_space = Box(low=-np.ones(2), high=np.ones(2))

    @staticmethod
    def _translate_raw_action(raw_action: np.array):
        engine_action, thruster_action = raw_action
        # [-1, 1] -> [0, 1]
        engine_action = (engine_action + 1) / 2
        return np.array([engine_action, thruster_action])
