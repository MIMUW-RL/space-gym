from typing import Iterable
import numpy as np
from gym_space.envs.classical_mechanics import ClassicalMechanicsEnv
from collections import namedtuple

from gym_space.rendering import create_rocket_viewer_transform_exhausts

RocketPosition = namedtuple("RocketPosition", ["x", "y", "angle"])
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
MARGIN = 30


class RocketLanding(ClassicalMechanicsEnv):
    rocket_body_radius = 12.0
    engines_angles = ()
    world_width = 1.0
    world_height = 1.0

    def __init__(
        self,
        *,
        position_lows: np.array,
        position_highs: np.array,
        force_mag: np.array,
        n_actions: Iterable[int],
        gravity: float = 9.8,
        mass: float = 1.0,
        dt: float = 0.02,
        angular_pos_nums=()
    ):
        self.gravity = gravity
        self.mass = mass
        self.force_mag = force_mag
        self._viewer = None
        self._exhausts = None
        self._transform = None

        super().__init__(
            position_lows=position_lows,
            position_highs=position_highs,
            n_actions=n_actions,
            dt=dt,
            angular_pos_nums=angular_pos_nums,
        )

    def external_force(self, action, state):
        raise NotImplementedError

    def acceleration(self, x, v, external_force):
        raise NotImplementedError

    def final_reward(self):
        raise NotImplementedError

    def step_reward(self, action):
        raise NotImplementedError

    def reward(self, action):
        reward = self.step_reward(action)
        if self.done:
            reward += self.final_reward()
        return reward

    def render(self, mode="human"):
        if self._viewer is None:
            assert self._transform is None
            (
                self._viewer,
                self._exhausts,
                self._transform,
            ) = create_rocket_viewer_transform_exhausts(
                self.rocket_body_radius,
                self.engines_angles,
            )

        if self.state is None:
            return

        rocket_position = self.rocket_position
        scale_x = SCREEN_WIDTH / self.world_width
        scale_y = SCREEN_HEIGHT / self.world_height

        self._transform.set_translation(
            rocket_position.x * scale_x + MARGIN + SCREEN_WIDTH / 2,
            rocket_position.y * scale_y + MARGIN,
        )
        self._transform.set_rotation(rocket_position.angle)
        if self._action is not None:
            self.render_exhausts(self._action)

        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    @property
    def rocket_position(self) -> RocketPosition:
        raise NotImplementedError

    @staticmethod
    def raw_action_to_action(raw_action):
        return NotImplementedError

    def render_exhausts(self, action):
        return
