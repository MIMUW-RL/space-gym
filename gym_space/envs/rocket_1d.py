import numpy as np
from .classical_mechanics import ClassicalMechanicsEnv


class Rocket1D(ClassicalMechanicsEnv):
    def __init__(self):
        self.gravity = 9.8
        self.mass = 1
        self.force_mag = 19.4
        self.fuel_penalty = 1.0
        self.viewer = None
        self.rocket_trans = None
        self.low, self.high = 0, 10
        super().__init__(
            position_lows=self.low, position_highs=self.high, n_actions=2, dt=0.02
        )

    def action_to_forces(self, action):
        return np.array([action * self.force_mag])

    def acceleration(self, x, v, control):
        force = control - self.gravity
        return force / self.mass

    def reward(self, action):
        reward = -action * self.fuel_penalty
        if self.done:
            x, v = self.state
            if x < self.low:
                v = min(v, 0)
                # no penalty if not going downward, exponential otherwise
                reward -= min(np.expm1(-v), 1e4)
            elif x > self.high:
                reward -= 1e5
            else:
                raise ValueError

        return reward

    def sample_initial_state(self):
        return np.array([5, 0])

    def render(self, mode="human"):
        screen_width = 200
        screen_height = 600

        world_height = self.high - self.low
        scale = screen_height / world_height
        rocket_width = 20.0
        rocket_height = 50.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = (
                -rocket_width / 2,
                rocket_width / 2,
                rocket_height / 2,
                -rocket_height / 2,
            )
            rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.rocket_trans = rendering.Transform()
            rocket.add_attr(self.rocket_trans)
            self.viewer.add_geom(rocket)

        if self.state is None:
            return

        rocket_bottom_pos = self.state[0] * scale + rocket_height / 2
        self.rocket_trans.set_translation(screen_width / 2, rocket_bottom_pos)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
