import numpy as np


def square_linear_const(error_arr: np.array, const: float, const_from: float):
    error = np.linalg.norm(error_arr)
    if error < square_to:
        return error ** 2
    if error < linear_to:
        return square_to ** 2 + 2 * square_to * (error - square_to)
    return square_to ** 2 + 2 * square_to * (linear_to - square_to)