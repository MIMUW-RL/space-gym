import numpy as np
from .rocket_landing import RocketLanding, RocketPosition


# TODO: visualize applied torque
class Rocket2D3DoF(RocketLanding):
    engines_angles = (0.,)
    world_width = 10
    world_height = 10

    def __init__(self):
        self.moment_of_inertia = 1
        self.x_low, self.x_high = -5.0, 5.0
        self.y_low, self.y_high = 0.0, 10.0
        self.alpha_low, self.alpha_high = 0.0, 2 * np.pi
        super().__init__(
            position_lows=np.array([self.x_low, self.y_low, self.alpha_low]),
            position_highs=np.array([self.x_high, self.y_high, self.alpha_high]),
            force_mag=np.array([5., 19.4]),
            n_actions=(3, 2),
            angular_pos_nums=(2,),
        )

    def external_force(self, action: np.array, state: np.array):
        force_ = action.astype(np.float32)
        force_ *= self.force_mag
        alpha = state[2]
        engine_force_dir = np.array([-np.sin(alpha), np.cos(alpha)])
        force_xy = engine_force_dir * force_[1]
        return np.concatenate([force_xy, force_[:1]])

    def acceleration(self, x, v, external_force):
        force = external_force.copy()
        force[1] -= self.gravity
        acc = np.concatenate(
            [force[:2] / self.mass, force[2:] / self.moment_of_inertia]
        )
        return acc

    def final_reward(self):
        reward = 0.0
        x, y, alpha, v_x, v_y, v_alpha = self.state
        if y <= self.y_low or np.isclose(y, self.y_low):
            # rocket should land on x = 0
            reward -= np.abs(x) * 100
            # it have to point upward
            reward -= min(np.tan(alpha) ** 2, 1e4)
            # no penalty for vertical speed if not going downward
            v_y = min(v_y, 0)
            reward -= min(np.expm1(-v_y + np.abs(v_x) + np.abs(v_alpha)), 1e4)
        else:
            reward -= 1e5

        return reward

    @property
    def rocket_position(self) -> RocketPosition:
        x, y, angle = self.state[:3]
        return RocketPosition(
            x=x,
            y=y,
            angle=angle
        )

    @staticmethod
    def raw_action_to_action(raw_action):
        raw_action[0] -= 1
        return raw_action

    def render_exhausts(self, action):
        exhaust = self._exhausts[0]
        if action[0]:
            exhaust.set_color(0., 0., 0.)
        else:
            exhaust.set_color(1., 1., 1.)

    def sample_initial_state(self):
        return (0., 5., 0., 0., 0., 0.)