from dataclasses import dataclass
from enum import Enum


class Steering(Enum):
    acceleration = 0
    velocity = 1
    angle = 2


@dataclass
class ShipParams:
    steering: Steering
    mass: float
    moi: float  # moment of inertia
    max_engine_force: float  # maximal force of main engine
    max_thruster_force: float  # maximal absolute torque of reaction control thruster
