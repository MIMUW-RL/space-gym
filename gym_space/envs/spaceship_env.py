from abc import ABC, abstractmethod

import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet
from gym_space.ship import Ship
from gym_space.helpers import angle_to_unit_vector, vector_to_angle
from typing import List, cast
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from functools import partial
from collections import deque


class SpaceshipEnv(gym.Env, ABC):
    max_episode_steps: int = None
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }
    num_prev_pos_vis: int = 25

    def __init__(
        self,
        *,
        ship: Ship,
        planets: List[Planet],
        world_size: float,
        step_size: float,
        max_abs_angular_velocity: float,
        velocity_xy_std: np.array,
        with_lidar: bool,
        with_goal: bool
    ):
        self.ship = ship
        self.planets = planets
        self.world_size = world_size
        self.step_size = step_size
        self.max_abs_angular_velocity = max_abs_angular_velocity
        assert velocity_xy_std.shape == (2,)
        self.velocity_xy_std = velocity_xy_std

        obs_high = [1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0]
        self.with_lidar = with_lidar
        self.with_goal = with_goal
        self.goal_pos = None
        if self.with_lidar:
            # (x, y) vector for each planet
            obs_high += 2 * len(self.planets) * [2 * np.sqrt(2)]
            if self.with_goal:
                obs_high += 2 * [2 * np.sqrt(2)]
        obs_high = np.array(obs_high)
        self.observation_space = Box(low=-obs_high, high=obs_high)
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self._np_random = None
        self.seed()
        self.internal_state = self.last_action = self.elapsed_steps = self._renderer = None
        self.prev_pos = deque(maxlen=self.num_prev_pos_vis)

    def reset(self):
        self._reset()
        if self._renderer is not None:
            self._renderer.move_planets()
        self.elapsed_steps = 0
        self.prev_pos.clear()
        self.prev_pos.append(self.internal_state[:2])
        return self.external_state

    def step(self, raw_action):
        assert (
            self.elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        assert self.action_space.contains(raw_action), raw_action
        action = np.array(self._translate_raw_action(raw_action))
        self.last_action = action

        done = self._update_state(action)
        self.prev_pos.append(self.internal_state[:2])
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
        reward = self._reward()
        return self.external_state, reward, done, {}

    def render(self, mode="human"):
        if self._renderer is None:
            from gym_space.rendering import Renderer

            self._renderer = Renderer(15, self.planets, self.world_size, self.with_goal)
            if self.goal_pos is not None:
                self._renderer.move_goal(self.goal_pos)

        return self._renderer.render(self.internal_state[:3], self.last_action, self.prev_pos, mode)

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
            return np.min(self.world_size / 2 + state[:2])

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
        ode_solution = cast(OdeResult, ode_solution)
        assert ode_solution.success, ode_solution.message
        self.internal_state = ode_solution.y[:, -1]
        self._wrap_angle()
        # done if any of self._boundary_events occurred
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
        external_state = [external_state[:2], angle_repr, external_state[3:]]

        def create_lidar_vector(obj_pos, planet_radius = 0.0):
            ship_center_obj_vec = obj_pos - self.internal_state[:2]
            ship_obj_angle = vector_to_angle(ship_center_obj_vec) - np.pi / 2 - self.internal_state[2]
            ship_obj_angle %= 2 * np.pi
            scale = (np.linalg.norm(ship_center_obj_vec) - planet_radius) * 2 / self.world_size
            return angle_to_unit_vector(ship_obj_angle) * scale

        if self.with_lidar:
            external_state += [create_lidar_vector(p.center_pos, p.radius) for p in self.planets]
            if self.with_goal:
                external_state += [create_lidar_vector(self.goal_pos)]
        return np.concatenate(external_state)

    @property
    def viewer(self):
        return self._renderer.viewer

    @abstractmethod
    def _init_action_space(self):
        # different for discrete and continuous environments
        pass

    @staticmethod
    @abstractmethod
    def _translate_raw_action(raw_action):
        # different for discrete and continuous environments
        pass

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _reward(self) -> float:
        pass


class DiscreteSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        # engine can be turned on or off: 2 options
        # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
        self.action_space = Discrete(2 * 3)

    @staticmethod
    def _translate_raw_action(raw_action: int):
        if raw_action == 0:
            return 0.0, 0.0
        elif raw_action == 1:
            return 1.0, 0.0
        elif raw_action == 2:
            return 0.0, -1.0
        elif raw_action == 3:
            return 0.0, 1.0
        elif raw_action <= 5:
            return 1.0, (raw_action - 4.5) * 2
        else:
            raise ValueError


class ContinuousSpaceshipEnv(SpaceshipEnv, ABC):
    def _init_action_space(self):
        self.action_space = Box(low=-np.ones(2), high=np.ones(2))

    @staticmethod
    def _translate_raw_action(raw_action: np.array):
        engine_action, thruster_action = raw_action
        # [-1, 1] -> [0, 1]
        return (engine_action + 1) / 2, thruster_action
