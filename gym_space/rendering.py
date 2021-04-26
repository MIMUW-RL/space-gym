from typing import List

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
        world_min: np.array,
        world_max: np.array,
    ):
        self.world_translation = world_min
        world_size = world_max - world_min
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
            # TODO: translate to good position
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

    def _world_to_screen(self, world_pos: np.array):
        return self.world_scale * (world_pos - self.world_translation)

    def render(self, ship_world_position: np.array, engine_active: bool, mode: str):
        ship_screen_position = self._world_to_screen(ship_world_position[:2])
        self.ship_transform.set_translation(*ship_screen_position)
        self.ship_transform.set_rotation(ship_world_position[2])
        if engine_active:
            self.exhaust.set_color(0.0, 0.0, 0.0)
        else:
            self.exhaust.set_color(1.0, 1.0, 1.0)
        return self.viewer.render(mode == "rgb_array")
