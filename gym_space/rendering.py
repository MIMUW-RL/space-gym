from typing import List
import os
import numpy as np
from gym.envs.classic_control import rendering
from collections import deque

from .planet import Planet
from .helpers import angle_to_unit_vector

MAX_SCREEN_SIZE = 600


class Renderer:
    def __init__(
        self,
        ship_body_radius: float,
        planets: List[Planet],
        world_size: float,
        with_goal: bool,
        num_prev_pos_vis: int = 30,
        prev_pos_color_decay: float = 0.85
    ):
        self.world_translation = np.full(2, -world_size/2)
        self.world_scale = MAX_SCREEN_SIZE / world_size
        screen_size = np.full(2, world_size * self.world_scale, dtype=np.int64)
        self.viewer = rendering.Viewer(*screen_size)
        self.ship_transform = rendering.Transform()
        self.planets = planets
        self._init_planets()
        self._init_engine(ship_body_radius)
        self.exhaust = None
        self._init_exhaust(ship_body_radius)
        self._init_ship(ship_body_radius)
        self.with_goal = with_goal
        self.goal_transform = None
        if self.with_goal:
            self._init_goal()
        self.prev_ship_pos = deque(maxlen=num_prev_pos_vis)
        self.prev_pos_color_decay = prev_pos_color_decay

    def _move_planets(self):
        for planet, transform in zip(self.planets, self._planets_transforms):
            transform.set_translation(*self._world_to_screen(planet.center_pos))

    def reset(self):
        self._move_planets()
        self.prev_ship_pos.clear()

    def move_goal(self, goal_pos):
        assert self.with_goal
        self.goal_transform.set_translation(*self._world_to_screen(goal_pos))

    def _init_planets(self):
        self._planets_transforms = []
        for planet in self.planets:
            planet_geom = rendering.make_circle(
                planet.radius * self.world_scale, filled=False
            )
            transform = rendering.Transform()
            self._planets_transforms.append(transform)
            planet_geom.add_attr(transform)
            self.viewer.add_geom(planet_geom)

    def _init_goal(self):
        line1 = rendering.Line((-10, -10), (10, 10))
        line2 = rendering.Line((-10, 10), (10, -10))
        goal = rendering.Compound([line1, line2])
        self.goal_transform = rendering.Transform()
        goal.add_attr(self.goal_transform)
        self.viewer.add_geom(goal)

    def _init_engine(self, ship_body_radius: float):
        engine_edge_length = ship_body_radius * 1.7
        engine_width_angle = np.pi / 4

        engine_left_bottom_angle = -engine_width_angle / 2
        engine_right_bottom_angle = engine_width_angle / 2
        engine_left_bottom_pos = engine_edge_length * angle_to_unit_vector(
            engine_left_bottom_angle
        )
        engine_right_bottom_pos = engine_edge_length * angle_to_unit_vector(
            engine_right_bottom_angle
        )
        engine = rendering.FilledPolygon(
            [(0.0, 0.0), engine_left_bottom_pos, engine_right_bottom_pos]
        )
        engine.add_attr(self.ship_transform)
        self.viewer.add_geom(engine)

    def _init_exhaust(self, ship_body_radius: float):
        engine_width_angle = np.pi / 4
        exhaust_begin_radius = ship_body_radius * 1.9
        exhaust_end_radius = ship_body_radius * 2.2

        flames = []
        for flame_angle in np.linspace(
            -engine_width_angle / 4,
            engine_width_angle / 4,
            3,
        ):
            vec = angle_to_unit_vector(flame_angle)
            flame = rendering.Line(exhaust_begin_radius * vec, exhaust_end_radius * vec)
            flames.append(flame)

        self.exhaust = rendering.Compound(flames)
        self.exhaust.add_attr(self.ship_transform)
        self.viewer.add_geom(self.exhaust)

    def _init_ship(self, ship_body_radius: float):
        ship_body = rendering.make_circle(ship_body_radius, filled=True)
        ship_body.set_color(1.0, 1.0, 1.0)
        ship_body.add_attr(self.ship_transform)
        self.viewer.add_geom(ship_body)

        ship_body_outline = rendering.make_circle(ship_body_radius, filled=False)
        ship_body_outline.add_attr(self.ship_transform)
        self.viewer.add_geom(ship_body_outline)

        ship_body_middle = rendering.Point()
        ship_body_middle.add_attr(self.ship_transform)
        ship_body_middle.set_color(0.5, 0.5, 0.5)
        self.viewer.add_geom(ship_body_middle)

        torque_img_filename = os.path.join(os.path.dirname(__file__), "assets/torque_img.png")
        self._torque_img = rendering.Image(torque_img_filename, 20., 20.)
        self._torque_img_transform = rendering.Transform()
        self._torque_img.add_attr(self._torque_img_transform)
        self._torque_img.add_attr(self.ship_transform)

    def _draw_ship_trace(self):
        color_diff = 1.0
        for i in range(1, len(self.prev_ship_pos)):
            line = rendering.Line(self.prev_ship_pos[- i], self.prev_ship_pos[- i - 1])
            color = 3 * [1 - color_diff]
            line.set_color(*color)
            color_diff *= self.prev_pos_color_decay
            self.viewer.add_onetime(line)

    def _world_to_screen(self, world_pos: np.array):
        return self.world_scale * (world_pos - self.world_translation)

    def render(self, ship_world_position: np.array, action: np.array, mode: str):
        self.viewer.add_onetime(self._torque_img)
        self._torque_img_transform.set_rotation(4)
        ship_screen_position = self._world_to_screen(ship_world_position[:2])
        self.prev_ship_pos.append(ship_screen_position)
        self.ship_transform.set_translation(*ship_screen_position)
        self.ship_transform.set_rotation(ship_world_position[2])
        if action is not None:
            thrust_action, torque_action = action
        else:
            thrust_action = torque_action = 0
        exhaust_color = 3 * [1 - thrust_action]
        self.exhaust.set_color(*exhaust_color)
        self._torque_img_transform.scale = (-torque_action, np.abs(torque_action))
        self._draw_ship_trace()
        return self.viewer.render(mode == "rgb_array")
