from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import Callable, cast
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from functools import partial

from . import helpers
from .ship_params import ShipParams, Steering
from .planet import Planet


def must_be_defined(method: Callable):
    """Property decorator that checks if ship state is defined"""

    def decorated_method(self, *args, **kwargs):
        assert self.is_defined, "ShipState set() method has to be called before accessing ship state"
        return method(self, *args, **kwargs)

    return decorated_method


@dataclass
class ShipState:
    """Holds state of the system and provides method to advance its evolution"""

    ship_params: ShipParams
    planets: list[Planet]
    _state_vec: np.array = field(init=False, default=None)
    _termination_events: list[Callable] = field(init=False)
    world_size: InitVar[float]
    max_abs_vel_angle: InitVar[float]

    def __post_init__(self, world_size: float, max_abs_vel_angle: float):
        self._termination_events = make_termination_events(world_size, max_abs_vel_angle, self.planets)

    def set(self, pos_xy: np.array, pos_angle: float, vel_xy: np.array, vel_angle: float):
        assert pos_xy.shape == vel_xy.shape == (2,)
        self._state_vec = np.array([*pos_xy, pos_angle, *vel_xy, vel_angle])

    def step(self, action: np.array, step_size: float) -> bool:
        """Advance evolution of the system under action by step_size seconds

        Returns:
            True if the episode ended (any of termination events occurred),
            False otherwise
        """
        self._state_vec, done = make_step(
            self.ship_params,
            self.planets,
            self._state_vec,
            action,
            step_size,
            self._termination_events,
        )
        return done

    @property
    def is_defined(self):
        return self._state_vec is not None

    @property
    @must_be_defined
    def full_pos(self) -> np.array:
        return get_ship_full_pos(self._state_vec)

    @property
    @must_be_defined
    def pos_xy(self) -> np.array:
        return get_ship_pos_xy(self._state_vec)

    @property
    @must_be_defined
    def pos_angle(self) -> float:
        return get_ship_pos_angle(self._state_vec)

    @property
    @must_be_defined
    def full_vel(self) -> np.array:
        return get_ship_full_vel(self._state_vec)

    @property
    @must_be_defined
    def vel_xy(self) -> np.array:
        return get_ship_vel_xy(self._state_vec)

    @property
    @must_be_defined
    def vel_angle(self) -> float:
        return get_ship_vel_angle(self._state_vec)


def make_step(
    ship_params: ShipParams,
    planets: list[Planet],
    state_vec: np.array,
    action: np.array,
    step_size: float,
    termination_events: list[Callable],
):
    """Advance evolution of the system by at most step_size seconds.

    Stop earlier if any of termination_events occurred.

    Returns:
        New state_vec and
        True if a termination event occurred, False otherwise.
    """
    # wrong events type
    # noinspection PyTypeChecker
    ode_solution = solve_ivp(
        partial(ship_vector_field, ship_params, planets, action),
        method="RK45",
        t_span=(0, step_size),
        y0=state_vec,
        events=termination_events,
    )
    ode_solution = cast(OdeResult, ode_solution)
    assert ode_solution.success, ode_solution.message
    state_vec = ode_solution.y[:, -1]
    wrap_ship_angle(state_vec)
    # done if any of termination_events occurred
    done = ode_solution.status == 1
    return state_vec, done


# modify such that different controls are allowed
def ship_vector_field(
    ship_params: ShipParams,
    planets: list[Planet],
    action: np.array,
    _time,
    state_vec: np.array,
):
    """Compute RHS of differential equation for ship dynamic"""
    acceleration = ship_acceleration(ship_params, planets, action, state_vec)
    if ship_params.steering == Steering.velocity:
        engine_action, thruster_action = action
        force_angle = thruster_action * 5.0  # 4 is some fixed scaling of thruster action appl. to vel.
        set_ship_vel_angle(state_vec, force_angle)
    return np.concatenate([get_ship_full_vel(state_vec), acceleration])


def ship_acceleration(
    ship_params: ShipParams,
    planets: list[Planet],
    action: np.array,
    state_vec: np.array,
):
    """Compute ship acceleration"""
    external_force = ship_external_force(ship_params, action, state_vec)
    external_force_xy, external_force_angle = np.split(external_force, [2])

    force_xy = external_force_xy
    for planet in planets:
        force_xy += helpers.gravity(get_ship_pos_xy(state_vec), planet.center_pos, ship_params.mass, planet.mass)
    acceleration_xy = force_xy / ship_params.mass

    if ship_params.steering == Steering.acceleration:
        acceleration_angle = external_force_angle / ship_params.moi
    else:
        acceleration_angle = np.zeros_like(external_force_angle)

    return np.concatenate([acceleration_xy, acceleration_angle])


def ship_external_force(ship_params: ShipParams, action: np.array, state: np.array):
    """Compute force acting on the ship due to action taken"""
    engine_action, thruster_action = action
    engine_force_scalar = engine_action * ship_params.max_engine_force
    angle = get_ship_pos_angle(state)
    engine_force_direction = -helpers.angle_to_unit_vector(angle)
    force_xy = engine_force_direction * engine_force_scalar
    force_angle = thruster_action * ship_params.max_thruster_force
    return np.array([*force_xy, force_angle])


def wrap_ship_angle(state_vec: np.array):
    state_vec[2] %= 2 * np.pi


def make_termination_events(world_size: float, max_abs_vel_angle: float, planets: list[Planet]) -> list[Callable]:
    """Create continuous function that are positive iff state is not terminal"""
    events = []

    def planet_event(planet: Planet, _t, state_vec: np.array):
        """Crashing into a planet"""
        return np.linalg.norm(planet.center_pos - get_ship_pos_xy(state_vec)) - planet.radius

    for planet_ in planets:
        event = partial(planet_event, planet_)
        event.terminal = True
        events.append(event)

    def world_max_event(_t, state_vec: np.array):
        """Going over top or right world boundary"""
        return np.min(world_size / 2 - get_ship_pos_xy(state_vec))

    world_max_event.terminal = True
    events.append(world_max_event)

    def world_min_event(_t, state_vec: np.array):
        """Going over bottom or left world boundary"""
        return np.min(world_size / 2 + get_ship_pos_xy(state_vec))

    world_min_event.terminal = True
    events.append(world_min_event)

    def angular_velocity_event(_t, state_vec: np.array):
        """Exceeding maximal angular velocity"""
        return max_abs_vel_angle - np.abs(get_ship_vel_angle(state_vec))

    angular_velocity_event.terminal = True
    events.append(angular_velocity_event)

    return events


# ------------------------ #
#  Getters for ship state  #
# ------------------------ #


def get_ship_full_pos(state_vec: np.array) -> np.array:
    return state_vec[:3]


def get_ship_pos_xy(state_vec: np.array) -> np.array:
    return state_vec[:2]


def get_ship_pos_angle(state_vec: np.array) -> float:
    return state_vec[2]


def get_ship_full_vel(state_vec: np.array) -> np.array:
    return state_vec[3:6]


def get_ship_vel_xy(state_vec: np.array) -> np.array:
    return state_vec[3:5]


def get_ship_vel_angle(state_vec: np.array) -> float:
    return state_vec[5]


def set_ship_vel_angle(state_vec: np.array, vel: float):
    state_vec[5] = vel
