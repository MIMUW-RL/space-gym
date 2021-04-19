import numpy as np
from .rocket_landing import RocketLanding, RocketPosition


class Rocket2D2DoF(RocketLanding):
    engines_angles = (-np.pi / 2, 0., np.pi / 2)
    world_width = 10
    world_height = 10

    def __init__(self):
        self.x_low, self.x_high = -5.0, 5.0
        self.y_low, self.y_high = 0.0, 10.0
        super().__init__(
            position_lows=np.array([self.x_low, self.y_low]),
            position_highs=np.array([self.x_high, self.y_high]),
            force_mag=np.array([5., 19.4]),
            n_actions=(3, 2),
        )

    def external_force(self, action: np.array, _state):
        return action * self.force_mag

    def acceleration(self, x, v, external_force):
        force = external_force.copy()
        force[1] -= self.gravity
        return force / self.mass

    def final_reward(self):
        x, y, v_x, v_y = self.state
        if y <= self.y_low or np.isclose(y, self.y_low):
            reward = np.abs(x) * 100  # rocket should land on x = 0
            # no penalty for vertical speed if not going downward
            v_y = min(v_y, 0)
            reward -= min(np.expm1(-v_y + np.abs(v_x)), 1e4)
        else:
            reward = 1e5
        return reward

    @property
    def rocket_position(self) -> RocketPosition:
        x, y = self.state[:2]
        return RocketPosition(
            x=x,
            y=y,
            angle=0.
        )

    @staticmethod
    def raw_action_to_action(raw_action):
        raw_action[0] -= 1
        return raw_action

    def render_exhausts(self, action):
        left, middle, right = self._exhausts
        if action[0] == -1:
            right.set_color(0, 0, 0)
            left.set_color(1, 1, 1)
        else:
            left.set_color(0, 0, 0)
            right.set_color(1, 1, 1)
        if action[1]:
            middle.set_color(0, 0, 0)
        else:
            middle.set_color(1, 1, 1)