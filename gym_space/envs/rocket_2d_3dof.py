import numpy as np
from .classical_mechanics import ClassicalMechanicsEnv


class Rocket2D3DoF(ClassicalMechanicsEnv):
    def __init__(self):
        self.gravity = 9.8
        self.mass = 1
        self.moment_of_inertia = 1
        # torque and engine
        self.force_mag = np.array([5, 19.4])
        self.fuel_penalty = 1.0
        self.viewer = None
        self.rocket_trans = None
        self.x_low, self.x_high = -5.0, 5.0
        self.y_low, self.y_high = 0.0, 10.0
        self.alpha_low, self.alpha_high = 0.0, 2 * np.pi

        super().__init__(
            position_lows=np.array([self.x_low, self.y_low, self.alpha_low]),
            position_highs=np.array([self.x_high, self.y_high, self.alpha_high]),
            # rotation and engine
            n_actions=(3, 2),
            dt=0.02,
            angular_pos_nums=(2,)
        )

    @property
    def engine_active(self):
        if self._action is None:
            return False
        return bool(self._action[1])

    @staticmethod
    def action_natural_to_integer(action: np.array):
        integer_action = action.copy()
        integer_action[0] -= 1
        return integer_action

    def external_force(self, action: np.array, state: np.array):
        integer_action = self.action_natural_to_integer(action)
        force_ = integer_action.astype(np.float32)
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

    def reward(self, action):
        integer_action = self.action_natural_to_integer(action)
        reward = 0.0
        if np.any(integer_action):
            reward -= self.fuel_penalty

        if self.done:
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

    # def sample_initial_state(self):
    #     return np.array([0., 5., 0., 0., 0., 0.])

    def render(self, mode="human"):
        screen_width = 400
        screen_height = 600

        world_height = self.x_high - self.x_low
        world_width = self.y_high - self.y_low
        scale_x = screen_width / world_width
        scale_y = screen_height / world_height
        rocket_width = 30.0
        rocket_height = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = (
                -rocket_width / 2,
                rocket_width / 2,
                rocket_height / 2,
                -rocket_height / 2,
            )
            rocket = rendering.FilledPolygon([(l, b), (0, t), (r, b)])
            self.rocket_trans = rendering.Transform()
            rocket.add_attr(self.rocket_trans)

            self._exhaust = rendering.FilledPolygon([
                (-rocket_width / 4, -rocket_height/2),
                (0, -rocket_height),
                (rocket_width / 4, -rocket_height/2)
            ])
            self._exhaust.set_color(1., 0.54, 0.)
            self.exhaust_trans = rendering.Transform()
            self._exhaust.add_attr(self.exhaust_trans)

            self.viewer.add_geom(rocket)
            self.viewer.add_geom(self._exhaust)

        if self.state is None:
            return

        pos_x, pos_y, pos_alpha = self.state[:3]
        pos_x = (pos_x + world_width / 2) * scale_x
        # we want rocket's bottom, not middle, to touch the surface
        pos_y = pos_y * scale_y + rocket_height / 2
        self.rocket_trans.set_translation(pos_x, pos_y)
        self.rocket_trans.set_rotation(pos_alpha)

        self.exhaust_trans.set_translation(pos_x, pos_y)
        self.exhaust_trans.set_rotation(pos_alpha)
        if self.engine_active:
            self._exhaust.set_color(1., 0.54, 0.)
        else:
            self._exhaust.set_color(1., 1., 1.)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
