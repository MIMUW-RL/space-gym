from dataclasses import dataclass
import numpy as np

class Rewards:
    def reward(self, state: np.array, action: np.array):
        raise NotImplementedError


@dataclass
class ConstantRewards(Rewards):
    reward_value: float

    def reward(self, state: np.array, action: np.array):
        return self.reward_value
