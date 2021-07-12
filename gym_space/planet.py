from dataclasses import dataclass
import numpy as np

# gravitational constant
G = 6.6743e-11


@dataclass
class Planet:
    mass: float
    radius: float
    center_pos: np.array = None
