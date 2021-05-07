from dataclasses import dataclass
import numpy as np

class Rewards:
    def reward(self, state: np.array, action: np.array, done: bool):
        raise NotImplementedError


@dataclass
class ConstantRewards(Rewards):
    reward_value: float

    def reward(self, state: np.array, action: np.array, done: bool):
        return self.reward_value
