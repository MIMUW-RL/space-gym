from dataclasses import dataclass
import numpy as np


@dataclass
class Planet:
    center_pos: float
    mass: float
    radius: float
