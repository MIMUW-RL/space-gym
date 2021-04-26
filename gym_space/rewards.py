import numpy as np

class Rewards:
    def reward(self, state: np.array, action: np.array, done: bool):
        raise NotImplementedError


class ConstantRewards(Rewards):
    def reward(self, state: np.array, action: np.array, done: bool):
        return 1.0
