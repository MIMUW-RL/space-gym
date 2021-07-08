from dataclasses import dataclass


@dataclass
class ShipParams:
    mass: float
    moi: float  # moment of inertia
    max_engine_force: float  # maximal force of main engine
    max_thruster_force: float  # maximal absolute torque of reaction control thruster
