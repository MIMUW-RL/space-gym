import numpy as np
from typing import Union

def angle_to_unit_vector(angle: Union[float, np.ndarray]) -> np.array:
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def vector_to_angle(vector: np.array) -> float:
    return np.arctan2(vector[..., 1], vector[..., 0])


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


def get_max_dist_in_direction(max_pos_, obj_pos, direction_unit_vec):
    candidate_max_dist = (
        (max_pos_ - obj_pos[0]) / direction_unit_vec[0],
        (- max_pos_ - obj_pos[0]) / direction_unit_vec[0],
        (max_pos_ - obj_pos[1]) / direction_unit_vec[1],
        (- max_pos_ - obj_pos[1]) / direction_unit_vec[1],
    )
    candidate_max_dist = filter(lambda x: x > 0, candidate_max_dist)
    return min(candidate_max_dist)

def uniform_disk_distribution(np_random, radius: Union[float, np.ndarray] = 1.0, size: int = None):
    if size is None:
        size = radius.shape[0] if isinstance(radius, np.ndarray) else 1
    angle = np_random.uniform(0, 2 * np.pi, size=size)
    r = np.sqrt(np_random.uniform(size=size) * radius**2)
    return np.squeeze(r[:, np.newaxis] * angle_to_unit_vector(angle))