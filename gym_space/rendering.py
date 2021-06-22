from typing import List
import os
import numpy as np
from gym.envs.classic_control import rendering


from .planet import Planet
from .helpers import angle_to_unit_vector

MAX_SCREEN_SIZE = np.array([800, 500])


class Renderer:
    def __init__(
        self,
        ship_body_radius: float,
        planets: List[Planet],
        world_size: np.array,
    ):
        self.world_translation = -world_size/2
        self.world_scale = np.min(MAX_SCREEN_SIZE / world_size)
        screen_size = np.array(world_size * self.world_scale, dtype=np.int64)
        self.viewer = rendering.Viewer(*screen_size)
        self.ship_transform = rendering.Transform()
        self._init_planets(planets)
        self._init_engine(ship_body_radius)
        self.exhaust = None
        self._init_exhaust(ship_body_radius)
        self._init_ship(ship_body_radius)

    def _init_planets(self, planets: List[Planet]):
        for planet in planets:
            # FIXME: translate to good position
            planet_geom = rendering.make_circle(
                planet.radius * self.world_scale, filled=False
            )
            planet_geom.add_attr(
                rendering.Transform(
                    translation=self._world_to_screen(planet.center_pos)
                )
            )
            self.viewer.add_geom(planet_geom)

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

    def _world_to_screen(self, world_pos: np.array):
        return self.world_scale * (world_pos - self.world_translation)

    def render(self, ship_world_position: np.array, action: np.array, mode: str):
        self.viewer.add_onetime(self._torque_img)
        self._torque_img_transform.set_rotation(4)
        ship_screen_position = self._world_to_screen(ship_world_position[:2])
        self.ship_transform.set_translation(*ship_screen_position)
        self.ship_transform.set_rotation(ship_world_position[2])
        if action is not None:
            thrust_action, torque_action = action
        else:
            thrust_action = torque_action = 0
        exhaust_color = 3 * [1 - thrust_action]
        self.exhaust.set_color(*exhaust_color)
        self._torque_img_transform.scale = (-torque_action, np.abs(torque_action))
        return self.viewer.render(mode == "rgb_array")
