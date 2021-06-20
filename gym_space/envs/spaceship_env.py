from abc import ABC

import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet, planets_min_max
from gym_space.ship import Ship
from gym_space.rewards import Rewards
from gym_space.helpers import angle_to_unit_vector
from typing import List
import numpy as np
import torch
from scipy.integrate import solve_ivp
from functools import partial


class SpaceshipEnv(gym.Env):
    max_episode_steps = 300
    step_size = 1.0
    max_abs_angular_velocity = 0.3
    metadata = {
        "render.modes": ["human", "rgb_array"],
        # FIXME: it doesn't work
        "video.frames_per_second": 60 / step_size,  # one minute in one second
    }

    def __init__(
        self,
        ship: Ship,
        planets: List[Planet],
        rewards: Rewards,
        world_min: np.array = None,
        world_max: np.array = None,
        with_accelerations: bool = False
    ):
        self.ship = ship
        self.planets = planets
        self.rewards = rewards
        self.with_accelerations = with_accelerations

        world_min_max_error_message = (
            "You have to provide both world_min and world_max or none of them"
        )
        if world_min is None:
            assert world_max is None, world_min_max_error_message
            self._init_default_world_min_max()
        else:
            assert world_max is not None, world_min_max_error_message
            assert world_min.shape == world_max.shape == (2,)
            self.world_min = world_min
            self.world_max = world_max

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_world_min = torch.tensor(self.world_min, device=device)
        self._torch_world_max = torch.tensor(self.world_max, device=device)

        obs_low = [*self.world_min, 0.0, -np.inf, -np.inf, -self.max_abs_angular_velocity]
        obs_high = [*self.world_max, 2 * np.pi, np.inf, np.inf, self.max_abs_angular_velocity]
        if self.with_accelerations:
            obs_low += [-np.inf, -np.inf, -np.inf]
            obs_high += [np.inf, np.inf, np.inf]
        self.observation_space = Box(
            low=np.array(obs_low),
            high=np.array(obs_high),
        )
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self._np_random = None
        self.seed()
        self.state = self.last_action = self.elapsed_steps = self._renderer = None

    def _init_default_world_min_max(self):
        if len(self.planets) > 1:
            self.world_min, self.world_max = planets_min_max(self.planets)
        else:
            planet = self.planets[0]
            self.world_min = planet.center_pos - 4 * planet.radius
            self.world_max = planet.center_pos + 4 * planet.radius

    def _init_action_space(self):
        raise NotImplementedError

    def _init_boundary_events(self):
        self._boundary_events = []

        # Functions below use weird notation that works
        # both with numpy 1d arrays and torch 2d tensors

        def planet_event(planet: Planet, _t, state):
            # should equal 0 <=> ship center is touching planet border
            ship_center_dist2 = state[..., 0] ** 2 + state[..., 1] ** 2
            return ship_center_dist2 - planet.radius ** 2

        for planet_ in self.planets:
            event = partial(planet_event, planet_)
            event.terminal = True
            self._boundary_events.append(event)

        def world_min_event(idx: int, _t, state):
            return state[..., idx] - self.world_min[idx]

        def world_max_event(idx: int, _t, state):
            return self.world_max[idx] - state[..., idx]

        for idx_ in range(2):
            for event_fn in [world_min_event, world_max_event]:
                event = partial(event_fn, idx_)
                event.terminal = True
                self._boundary_events.append(event)

        def angular_velocity_event(_t, state):
            # should equal 0 <=> absolute angular velocity equals max_abs_angular_velocity
            return self.max_abs_angular_velocity ** 2 - state[..., 5] ** 2

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
            y0=self.state,
            events=self._boundary_events,
        )
        assert ode_solution.success, ode_solution.message
        self.state = ode_solution.y[:, -1]
        self._wrap_angle()
        done = ode_solution.status == 1
        assert done == self.termination_fn(None, torch.tensor(self.state, dtype=torch.float32)[None, :]).item()
        return done

    def _wrap_angle(self):
        self.state[2] %= 2 * np.pi
        # self.state[2] -= np.pi

    @staticmethod
    def _translate_raw_action(raw_action):
        # different for discrete and continuous environments
        raise NotImplementedError

    def _sample_initial_state(self):
        raise NotImplementedError

    def step(self, raw_action):
        assert (
            self.elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        assert self.action_space.contains(raw_action), raw_action
        action = self._translate_raw_action(raw_action)
        self.last_action = action
        reward = self.rewards.reward(self.state, action)

        done = self._update_state(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
        return self.external_state, reward, done, {}

    def reset(self):
        self.state = self._sample_initial_state()
        self.elapsed_steps = 0
        return self.external_state

    def render(self, mode="human"):
        if self._renderer is None:
            from gym_space.rendering import Renderer

            self._renderer = Renderer(15, self.planets, self.world_min, self.world_max)

        return self._renderer.render(self.state[:3], self.last_action, mode)

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def event_fn(self, next_obs: torch.Tensor):
        assert next_obs.ndim == 2
        event_fn_val = torch.full(next_obs.shape[:1], float("inf")).to(next_obs)
        for event_fn in self._boundary_events:
            # TODO: it won't work for planet as border, unless we adjust signs
            event_fn_val = torch.minimum(event_fn(_t=0, state=next_obs), event_fn_val)
        return event_fn_val

    # TODO: write tests, not just here
    # TODO: use it INSIDE DeLaN model
    def termination_fn(
        self, _act: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        assert next_obs.ndim == 2
        done = torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)
        for event_fn in self._boundary_events:
            event_fn_val = event_fn(_t=0, state=next_obs)
            # FIXME: LPO DeLaN event finding have to be accurate for this to work
            done |= is_close_to_zero(event_fn_val)

        return done[:, None]

    @property
    def external_state(self):
        state = self.state.copy()
        state[2] -= np.pi
        if self.with_accelerations:
            if self.last_action is None:
                acc = np.zeros(len(state) // 2)
            else:
                acc = self._vector_field(
                    action=self.last_action,
                    _time=0.0,
                    state=state
                )[3:]
            return np.concatenate([state, acc])
        return state

    @property
    def viewer(self):
        return self._renderer.viewer


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


def is_close_to_zero(x: torch.Tensor, **kwargs) -> torch.Tensor:
    zeros = torch.tensor(0).to(x).expand(x.shape)
    return torch.isclose(x, zeros, **kwargs)