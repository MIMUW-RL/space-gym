import numpy as np
import torch

def bounded_linear(x: np.array, upper_bound: float):
    return x / (1 + (1 / upper_bound) * x)


def bounded_square(x: np.array, upper_bound: float):
    return bounded_linear(x ** 2, upper_bound)


def angle_to_unit_vector(angle: float) -> np.array:
    if isinstance(angle, torch.Tensor):
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    return np.array([np.cos(angle), np.sin(angle)])


def vector_to_angle(vector: np.array) -> float:
    return np.arctan2(vector[1], vector[0])
