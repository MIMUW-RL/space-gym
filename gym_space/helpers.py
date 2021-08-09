import numpy as np


def angle_to_unit_vector(angle: float) -> np.array:
    return np.array([np.cos(angle), np.sin(angle)])


def vector_to_angle(vector: np.array) -> float:
    return np.arctan2(vector[1], vector[0])


def orthogonal_proj(a: np.array, v: np.array) -> np.array:
    """orthogonal projection of v onto a"""
    c = (a[0] * v[0] + a[1] * v[1]) / np.linalg.norm(a)
    return a * c


# gravitational constant
G = 6.6743e-11


def gravity(
    from_pos: np.array, toward_pos: np.array, from_mass: float, toward_mass: float
) -> np.array:
    """Compute gravitational force between two bodies

    Returns:
        Vector of gravitational force going from from_pos to toward_pos
    """
    assert from_pos.shape == toward_pos.shape
    pos_diff = toward_pos - from_pos
    center_distance = np.linalg.norm(pos_diff)
    force_direction = pos_diff / center_distance
    scalar_force = G * from_mass * toward_mass / center_distance ** 2
    return force_direction * scalar_force
