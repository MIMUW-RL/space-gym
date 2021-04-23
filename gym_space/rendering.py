from typing import Iterable

import numpy as np
from gym.envs.classic_control import rendering

from gym_space.envs.rocket_landing import SCREEN_WIDTH, MARGIN, SCREEN_HEIGHT


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

    world_border = rendering.Line(
        (0, MARGIN),
        (SCREEN_WIDTH + 2 * MARGIN, MARGIN)
    )
    world_border.set_color(.5, .5, .5)
    viewer.add_geom(world_border)

    return viewer, exhausts, rocket_transform