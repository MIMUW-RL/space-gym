from dataclasses import dataclass, field, InitVar
from typing import Callable, cast
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from functools import partial

from . import helpers
from .ship_params import ShipParams
from .planet import Planet

# gravitational constant
G = 6.6743e-11

def require_ship_state(method: Callable):
    def decorated_method(self, *args, **kwargs):
        assert self._ship_state is not None, "ship state is not initialized"
        return method(self, *args, **kwargs)
    return decorated_method

@dataclass
class DynamicState:
    ship_params: ShipParams
    planets: list[Planet]
    _ship_state: np.array = field(init=False, default=None)
    _boundary_events: list[Callable] = field(init=False)
    world_size: InitVar[float]
    max_abs_vel_angle: InitVar[float]

    def __post_init__(self, world_size, max_abs_vel_angle):
        self._boundary_events = make_boundary_events(world_size, max_abs_vel_angle, self.planets)

    def set_ship_state(self, pos_xy: np.array, pos_angle: float, vel_xy: np.array, vel_angle: float):
        assert pos_xy.shape == vel_xy.shape == (2,)
        self._ship_state = np.array([*pos_xy, pos_angle, *vel_xy, vel_angle])

    def step(self, action: np.array, step_size: float) -> bool:
        self._ship_state, done = make_step(
            self.ship_params,
            self.planets,
            self._ship_state,
            action,
            step_size,
            self._boundary_events
        )
        return done

    @property
    @require_ship_state
    def ship_pos(self):
        return get_ship_pos(self._ship_state)

    @property
    @require_ship_state
    def ship_pos_xy(self):
        return get_ship_pos_xy(self._ship_state)

    @property
    @require_ship_state
    def ship_pos_angle(self):
        return get_ship_pos_angle(self._ship_state)

    @property
    @require_ship_state
    def ship_vel(self):
        return get_ship_vel(self._ship_state)

    @property
    @require_ship_state
    def ship_vel_xy(self):
        return get_ship_vel_xy(self._ship_state)

    @property
    @require_ship_state
    def ship_vel_angle(self):
        return get_ship_vel_angle(self._ship_state)


def get_ship_pos(ship_state: np.array) -> np.array:
    return ship_state[:3]

def get_ship_pos_xy(ship_state: np.array) -> np.array:
    return ship_state[:2]

def get_ship_pos_angle(ship_state: np.array) -> float:
    return ship_state[2]

def get_ship_vel(ship_state: np.array) -> np.array:
    return ship_state[3:6]

def get_ship_vel_xy(ship_state: np.array) -> np.array:
    return ship_state[3:5]

def get_ship_vel_angle(ship_state: np.array) -> float:
    return ship_state[5]

def make_step(ship_params: ShipParams, planets: list[Planet], ship_state: np.array, action: np.array, step_size: float, events: list[Callable]):
    # wrong events type
    # noinspection PyTypeChecker
    ode_solution = solve_ivp(
        partial(vector_field, ship_params, planets, action),
        t_span=(0, step_size),
        y0=ship_state,
        events=events
    )
    ode_solution = cast(OdeResult, ode_solution)
    assert ode_solution.success, ode_solution.message
    ship_state = ode_solution.y[:, -1]
    wrap_angle(ship_state)
    # done if any of self._boundary_events occurred
    done = ode_solution.status == 1
    return ship_state, done

def external_force_and_torque(ship_params: ShipParams, action: np.array, state: np.array):
    engine_action, thruster_action = action
    engine_force = engine_action * ship_params.max_engine_force
    thruster_torque = thruster_action * ship_params.max_thruster_torque
    angle = get_ship_pos_angle(state)
    engine_force_direction = -helpers.angle_to_unit_vector(angle)
    force_xy = engine_force_direction * engine_force
    return np.array([*force_xy, thruster_torque])

def vector_field(ship_params: ShipParams, planets: list[Planet], action: np.array, _time, ship_state: np.array):
    external_force_and_torque_ = external_force_and_torque(ship_params, action, ship_state)
    external_force_xy, external_torque = np.split(external_force_and_torque_, [2])

    force_xy = external_force_xy
    for planet in planets:
        force_xy += gravity(get_ship_pos_xy(ship_state), planet.center_pos, ship_params.mass, planet.mass)
    acceleration_xy = force_xy / ship_params.mass

    torque = external_torque
    acceleration_angle = torque / ship_params.moi

    acceleration = np.concatenate([acceleration_xy, acceleration_angle])
    return np.concatenate([get_ship_vel(ship_state), acceleration])

def gravity(from_pos: np.array, toward_pos: np.array, from_mass: float, toward_mass: float):
    assert from_pos.shape == toward_pos.shape
    pos_diff = toward_pos - from_pos
    center_distance = np.linalg.norm(pos_diff)
    force_direction = pos_diff / center_distance
    scalar_force = G * from_mass * toward_mass / center_distance ** 2
    return force_direction * scalar_force

def wrap_angle(ship_state: np.array):
    ship_state[2] %= 2 * np.pi

def make_boundary_events(world_size: float, max_abs_angular_velocity: float, planets: list[Planet]) -> list[Callable]:
    events = []

    def planet_event(planet: Planet, _t, ship_state: np.array):
        return np.linalg.norm(planet.center_pos - get_ship_pos_xy(ship_state)) - planet.radius

    for planet_ in planets:
        event = partial(planet_event, planet_)
        event.terminal = True
        events.append(event)

    def world_max_event(_t, ship_state: np.array):
        return np.min(world_size / 2 - get_ship_pos_xy(ship_state))

    world_max_event.terminal = True
    events.append(world_max_event)

    def world_min_event(_t, ship_state: np.array):
        return np.min(world_size / 2 + get_ship_pos_xy(ship_state))

    world_min_event.terminal = True
    events.append(world_min_event)

    def angular_velocity_event(_t, ship_state: np.array):
        return max_abs_angular_velocity - np.abs(get_ship_vel_angle(ship_state))

    angular_velocity_event.terminal = True
    events.append(angular_velocity_event)

    return events