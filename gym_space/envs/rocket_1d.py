import numpy as np
from .rocket_landing import RocketLanding, RocketPosition


class Rocket1D(RocketLanding):
    engines_angles = (0.,)
    world_width = 1
    world_height = 10

    def __init__(self):
        self.low, self.high = 0, 10
        super().__init__(
            position_lows=self.low,
            position_highs=self.high,
            force_mag=np.array([19.4]),
            n_actions=(2,)
        )

    def external_force(self, action, _state):
        return action * self.force_mag

    def acceleration(self, x, v, external_force):
        force = external_force - self.gravity
        return force / self.mass

    def final_reward(self):
        x, v = self.state
        if x <= self.low or np.isclose(x, self.low):
            v = min(v, 0)
            # no penalty if not going downward, exponential otherwise
            return -min(np.expm1(-v), 1e4)
        elif x >= self.high or np.isclose(x, self.high):
            return -1e5
        else:
            raise ValueError(x)

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
