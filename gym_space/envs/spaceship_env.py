from abc import ABC, abstractmethod

import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space.helpers import angle_to_unit_vector, vector_to_angle
import numpy as np

from gym_space.dynamic_model import ShipState


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
        ship_params: ShipParams,
        planets: list[Planet],
        world_size: float,
        max_abs_vel_angle: float,
        step_size: float,
        vel_xy_std: np.array,
        with_lidar: bool,
        with_goal: bool,
        renderer_kwargs: dict = None
    ):
        """Base class for all Spaceship environments and tasks
        
        Args:
            ship_params: parameters describing properties of the spaceship
            planets: list of parameters describing properties of the planets
            world_size: width and height of square 2D world
            max_abs_vel_angle: maximal absolute value of angular velocity
            step_size: number of seconds between consecutive observations
            vel_xy_std: approximate standard deviation of translational velocities
            with_lidar: include "lidars" for planets and goal (if present) in observations
            with_goal: if goal (point in space) is present in the environment
            renderer_kwargs: keyword args for Renderer
        """
        self.planets = planets
        self.world_size = world_size
        self.step_size = step_size
        self.max_abs_vel_angle = max_abs_vel_angle
        assert vel_xy_std.shape == (2,)
        self.velocity_xy_std = vel_xy_std

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
        self._np_random = None
        self.seed()
        self.last_action = self.elapsed_steps = self._renderer = None
        self._ship_state = ShipState(ship_params, planets, self.world_size, self.max_abs_vel_angle)
        self.observation = None
        self.renderer_kwargs = dict() if renderer_kwargs is None else renderer_kwargs

    def reset(self):
        self._reset()
        assert self._ship_state.is_defined
        self._make_observation()
        if self._renderer is not None:
            self._renderer.reset()
            if self.with_goal:
                self._renderer.move_goal(self.goal_pos)
        self.elapsed_steps = 0
        return self.observation

    def step(self, raw_action):
        assert (
                self.elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        assert self.action_space.contains(raw_action), raw_action
        action = np.array(self._translate_raw_action(raw_action))
        self.last_action = action

        done = self._ship_state.step(action, self.step_size)
        self._make_observation()
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
        reward = self._reward()
        return self.observation, reward, done, {}

    def render(self, mode="human"):
        if self._renderer is None:
            from gym_space.rendering import Renderer

            self._renderer = Renderer(15, self.planets, self.world_size, self.with_goal, **self.renderer_kwargs)
            if self.goal_pos is not None:
                self._renderer.move_goal(self.goal_pos)

        return self._renderer.render(self._ship_state.full_pos, self.last_action, mode)

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _make_observation(self):
        # make sure that x and y positions are between -1 and 1
        obs_pos_xy = self._ship_state.pos_xy / self.world_size
        # normalize translational velocity
        obs_vel_xy = self._ship_state.vel_xy / self.velocity_xy_std
        # make sure that angular velocity is between -1 and 1
        obs_vel_angle = self._ship_state.vel_angle / self.max_abs_vel_angle
        # represent angle as cosine and sine
        angle = self._ship_state.pos_angle
        angle_repr = np.array([np.cos(angle), np.sin(angle)])
        observation = [obs_pos_xy, angle_repr, obs_vel_xy, np.array([obs_vel_angle])]

        if self.with_lidar:
            observation += [self._create_lidar_vector(p.center_pos, p.radius) for p in self.planets]
            if self.with_goal:
                observation += [self._create_lidar_vector(self.goal_pos)]
        self.observation = np.concatenate(observation)

    def _create_lidar_vector(self, obj_pos: np.array, obj_radius: float = 0.0) -> np.array:
        ship_center_obj_vec = obj_pos - self._ship_state.pos_xy
        ship_obj_angle = vector_to_angle(ship_center_obj_vec) - np.pi / 2 - self._ship_state.pos_angle
        ship_obj_angle %= 2 * np.pi
        scale = (np.linalg.norm(ship_center_obj_vec) - obj_radius) * 2 / self.world_size
        return angle_to_unit_vector(ship_obj_angle) * scale

    @property
    def planets_lidars(self):
        if not self.with_lidar:
            return None
        return self.observation[-2 * len(self.planets):].reshape(-1, 2)

    @property
    def goal_lidar(self):
        if not (self.with_lidar and self.with_goal):
            return None
        return self.observation[-2:]

    @property
    def viewer(self):
        return self._renderer.viewer

    @abstractmethod
    def _init_action_space(self):
        # different for discrete and continuous environments
        pass

    @staticmethod
    @abstractmethod
    def _translate_raw_action(raw_action) -> tuple[float, float]:
        # different for discrete and continuous environments
        pass

    @abstractmethod
    def _reset(self):
        """Must call self._ship_state.set()"""
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
    def _translate_raw_action(raw_action: int) -> tuple[float, float]:
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
    def _translate_raw_action(raw_action: np.array) -> tuple[float, float]:
        engine_action, thruster_action = raw_action
        # [-1, 1] -> [0, 1]
        return (engine_action + 1) / 2, thruster_action
