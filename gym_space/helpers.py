import numpy as np


def bounded_linear(x: np.array, upper_bound: float):
    return x / (1 + (1 / upper_bound) * x)


def bounded_square(x: np.array, upper_bound: float):
    return bounded_linear(x ** 2, upper_bound)


def angle_to_unit_vector(angle: float) -> np.array:
    return np.array([np.cos(angle), np.sin(angle)])


def vector_to_angle(vector: np.array) -> float:
    return np.arctan2(vector[1], vector[0])

def orthogonal_proj(a: np.array, v: np.array) -> np.array:
    """orthogonal projection of v onto a"""
    c = (a[0]*v[0] + a[1]*v[1]) / np.linalg.norm(a)
    return  a * c
    