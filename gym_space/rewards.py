from dataclasses import dataclass
import numpy as np
import abc

class Rewards(abc.ABC):
    @abc.abstractmethod
    def reward(self, state: np.array, action: np.array):
        pass


@dataclass
class ConstantRewards(Rewards):
    reward_value: float

    def reward(self, state: np.array, action: np.array):
        return self.reward_value
