from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from gym_space.helpers import angle_to_unit_vector, vector_to_angle
import numpy as np

from gym_space.dynamic_model import ShipState


@dataclass
class SpaceshipEnv(gym.Env, ABC):
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

    ship_params: ShipParams
    planets: list[Planet]
    world_size: float
    max_abs_vel_angle: float
    step_size: float
    vel_xy_std: np.array
    with_lidar: bool
    with_goal: bool
    renderer_kwargs: dict = None

    observation: np.array = field(init=False, default=None)
    last_action: np.array = field(init=False, default=None)
    goal_pos: np.array = field(init=False, default=None)

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __post_init__(self):
        if self.renderer_kwargs is None:
            self.renderer_kwargs = dict()
        self._init_observation_space()
        self._init_action_space()
        self._np_random = self._renderer = None
        self.seed()
        self._ship_state = ShipState(
            self.ship_params, self.planets, self.world_size, self.max_abs_vel_angle
        )

    def reset(self):
        self._reset()
        assert self._ship_state.is_defined
        assert self.with_goal == (self.goal_pos is not None)
        self._make_observation()
        if self._renderer is not None:
            self._renderer.reset(self.goal_pos)
        return self.observation

    # define reward function
    def reward(self, action, prev_state):
        return self.rewards.reward(self.internal_state, action)

    def step(self, raw_action):
        assert self.action_space.contains(raw_action), raw_action
        action = np.array(self._translate_raw_action(raw_action))
        self.last_action = action
        done = self._ship_state.step(action, self.step_size)
        self._make_observation()
        reward = self._reward()
        return self.observation, reward, done, {}

    def render(self, mode="human"):
        if self._renderer is None:
            from gym_space.rendering import Renderer

            self._renderer = Renderer(
                self.planets, self.world_size, self.goal_pos, **self.renderer_kwargs
            )

        return self._renderer.render(self._ship_state.full_pos, self.last_action, mode)

    def seed(self, seed=None):
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _init_observation_space(self):
        obs_high = [1.0, 1.0, 1.0, 1.0, np.inf, np.inf, 1.0]
        if self.with_lidar:
            # as normalized world is [-1, 1]^2, the highest distance between two points is 2 sqrt(2)
            # (x, y) vector for each planet
            obs_high += 2 * len(self.planets) * [2 * np.sqrt(2)]
            if self.with_goal:
                obs_high += 2 * [2 * np.sqrt(2)]
        obs_high = np.array(obs_high)
        self.observation_space = Box(low=-obs_high, high=obs_high)

    def _make_observation(self):
        # make sure that x and y positions are between -1 and 1
        obs_pos_xy = self._ship_state.pos_xy / self.world_size
        # normalize translational velocity
        obs_vel_xy = self._ship_state.vel_xy / self.vel_xy_std
        # make sure that angular velocity is between -1 and 1
        obs_vel_angle = self._ship_state.vel_angle / self.max_abs_vel_angle
        # represent angle as cosine and sine
        angle = self._ship_state.pos_angle
        angle_repr = np.array([np.cos(angle), np.sin(angle)])
        observation = [obs_pos_xy, angle_repr, obs_vel_xy, np.array([obs_vel_angle])]

        if self.with_lidar:
            observation += [
                self._create_lidar_vector(p.center_pos, p.radius) for p in self.planets
            ]
            if self.with_goal:
                observation += [self._create_lidar_vector(self.goal_pos)]
        self.observation = np.concatenate(observation)

    def _create_lidar_vector(
        self, obj_pos: np.array, obj_radius: float = 0.0
    ) -> np.array:
        """Create vector from ship to some object.

        Lidar's point of view is ship's point of view.
        It means that the returned vector is in coordinate system
        such that ship's engine exhaust is pointing downwards.
        """
        ship_center_obj_vec = obj_pos - self._ship_state.pos_xy
        ship_obj_angle = (
            vector_to_angle(ship_center_obj_vec)
            - np.pi / 2
            - self._ship_state.pos_angle
        )
        ship_obj_angle %= 2 * np.pi
        scale = (np.linalg.norm(ship_center_obj_vec) - obj_radius) * 2 / self.world_size
        return angle_to_unit_vector(ship_obj_angle) * scale

    @property
    def planets_lidars(self):
        if not self.with_lidar:
            return None
        return self.observation[-2 * len(self.planets) :].reshape(-1, 2)

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
