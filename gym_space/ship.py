from dataclasses import dataclass


@dataclass
class Ship:
    mass: float
    moi: float  # moment of inertia
    max_engine_force: float  # maximal force of main engine
    max_thruster_torque: float  # maximal absolute torque of reaction control thruster
