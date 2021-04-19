from typing import Iterable
import numpy as np
from gym.envs.classic_control import rendering
from gym_space.envs.classical_mechanics import ClassicalMechanicsEnv
from collections import namedtuple

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
        fuel_penalty: float = 1.0,
        dt: float = 0.02,
        angular_pos_nums=()
    ):
        self.gravity = gravity
        self.mass = mass
        self.force_mag = force_mag
        self.fuel_penalty = fuel_penalty
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

    def reward(self, action):
        reward = 0
        if np.any(action):
            reward -= self.fuel_penalty
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


def create_rocket_viewer_transform_exhausts(
    rocket_body_radius: float,
    engines_angles: Iterable[float],
):
    viewer = rendering.Viewer(SCREEN_WIDTH + 2 * MARGIN, SCREEN_HEIGHT + 2 * MARGIN)
    rocket_transform = rendering.Transform()

    engine_edge_length = rocket_body_radius * 1.7
    engine_width_angle = np.pi / 4

    exhausts = []
    for engine_angle in engines_angles:
        engine_left_bottom_angle = engine_angle - engine_width_angle / 2
        engine_right_bottom_angle = engine_angle + engine_width_angle / 2
        engine_left_bottom_pos = engine_edge_length * np.array(
            [np.sin(engine_left_bottom_angle), -np.cos(engine_left_bottom_angle)]
        )
        engine_right_bottom_pos = engine_edge_length * np.array(
            [np.sin(engine_right_bottom_angle), -np.cos(engine_right_bottom_angle)]
        )
        engine = rendering.FilledPolygon(
            [(0.0, 0.0), engine_left_bottom_pos, engine_right_bottom_pos]
        )
        engine.add_attr(rocket_transform)
        viewer.add_geom(engine)

        exhausts_begin_radius = rocket_body_radius * 1.9
        exhausts_end_radius = rocket_body_radius * 2.2

        flames = []
        for flame_angle in np.linspace(
            engine_angle - engine_width_angle / 4,
            engine_angle + engine_width_angle / 4,
            3,
        ):
            tmp = np.array([np.sin(flame_angle), -np.cos(flame_angle)])
            flame = rendering.Line(
                exhausts_begin_radius * tmp, exhausts_end_radius * tmp
            )
            flames.append(flame)

        exhaust = rendering.Compound(flames)
        exhaust.add_attr(rocket_transform)
        viewer.add_geom(exhaust)
        exhausts.append(exhaust)

    rocket_body = rendering.make_circle(rocket_body_radius, filled=True)
    rocket_body.set_color(1.0, 1.0, 1.0)
    rocket_body.add_attr(rocket_transform)
    viewer.add_geom(rocket_body)

    rocket_body_outline = rendering.make_circle(rocket_body_radius, filled=False)
    rocket_body_outline.add_attr(rocket_transform)
    viewer.add_geom(rocket_body_outline)

    rocket_body_middle = rendering.Point()
    rocket_body_middle.add_attr(rocket_transform)
    rocket_body_middle.set_color(.5, .5, .5)
    viewer.add_geom(rocket_body_middle)

    world_border = rendering.PolyLine(
        [
            (MARGIN, MARGIN),
            (MARGIN, SCREEN_HEIGHT + MARGIN),
            (SCREEN_WIDTH + MARGIN, SCREEN_HEIGHT + MARGIN),
            (SCREEN_WIDTH + MARGIN, MARGIN),
        ],
        close=True,
    )
    world_border.set_color(.5, .5, .5)
    viewer.add_geom(world_border)

    return viewer, exhausts, rocket_transform
