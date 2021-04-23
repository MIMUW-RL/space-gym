import numpy as np
from .rocket_landing import RocketLanding, RocketPosition


class Rocket1D(RocketLanding):
    engines_angles = (0.,)
    world_width = 1
    world_height = 10

    def __init__(self):
        self.low, self.high = 0, np.inf
        self.fuel_penalty = 1.0
        super().__init__(
            position_lows=self.low,
            position_highs=self.high,
            force_mag=np.array([11.0]),
            n_actions=(2,)
        )

    def external_force(self, action, _state):
        return action * self.force_mag

    def acceleration(self, x, v, external_force):
        force = external_force - self.gravity
        return force / self.mass

    def step_reward(self, action):
        x = self.state[0]
        reward = -x * 1e-3
        if np.any(action):
            reward -= self.fuel_penalty
        return reward

    def final_reward(self):
        x, v = self.state
        assert x <= self.low or np.isclose(x, self.low)
        reward = 1_000
        v = abs(v)
        if v > 10:
            penalty = min(10**2 + (v - 10), 900)
        else:
            penalty = v**2
        return reward - penalty

    def sample_initial_state(self):
        return np.array([8.0, 5 * self.np_random.normal()])

    @property
    def rocket_position(self) -> RocketPosition:
        return RocketPosition(
            x=0.,
            y=self.state[0],
            angle=0.
        )

    @staticmethod
    def raw_action_to_action(raw_action):
        return raw_action

    def render_exhausts(self, action):
        exhaust = self._exhausts[0]
        if action[0]:
            exhaust.set_color(0., 0., 0.)
        else:
            exhaust.set_color(1., 1., 1.)
