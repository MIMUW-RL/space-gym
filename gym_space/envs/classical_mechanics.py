import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial


class ClassicalMechanicsEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self, position_lows, position_highs, n_actions, dt, angular_pos_nums=()
    ):
        if np.isscalar(position_lows):
            assert np.isscalar(position_highs)
            position_lows = np.array([position_lows])
            position_highs = np.array([position_highs])
        self.dof = len(position_lows)
        velocities_lows = np.full_like(position_lows, -np.inf, dtype=np.float32)
        velocities_highs = np.full_like(position_lows, np.inf, dtype=np.float32)
        lows = np.concatenate([position_lows, velocities_lows], dtype=np.float32)
        highs = np.concatenate([position_highs, velocities_highs], dtype=np.float32)
        self.state_space = spaces.Box(lows, highs)
        self.state_space_dim = self.state_space.shape[0]
        assert self.state_space_dim >= 1
        self.angular_pos_nums = angular_pos_nums

        def event(threshold, sign, ind, _t, state):
            return sign * (state[ind] - threshold)

        self.terminal_events = []
        for i, (low, high) in enumerate(zip(position_lows, position_highs)):
            if i not in self.angular_pos_nums:
                self.terminal_events.append(partial(event, low, 1, i))
                self.terminal_events.append(partial(event, high, -1, i))
        for e in self.terminal_events:
            e.terminal = True

        if not np.isscalar(n_actions):
            assert len(n_actions) <= self.state_space_dim

        self.action_space = spaces.MultiDiscrete(n_actions)
        self.dt = dt  # seconds between observations
        self.state = None
        self.done = False
        self.np_random = None
        self.seed()
        self._action = None

    def normalize_angular(self):
        for i in self.angular_pos_nums:
            self.state[i] -= self.state_space.low[i]
            self.state[i] %= self.state_space.high[i] - self.state_space.low[i]
            self.state[i] += self.state_space.low[i]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        self.state = self.sample_initial_state()
        self.done = False
        return self.state

    def sample_initial_state(self):
        return self.state_space.sample()

    def step(self, action):
        assert self.action_space.contains(action), f"{action=} invalid"
        self._action = action

        def vector_field_(_t, state):
            return self.vector_field(action, state)

        ode_solution = solve_ivp(
            vector_field_,
            t_span=(0, self.dt),
            y0=self.state,
            events=self.terminal_events,
        )
        assert ode_solution.success
        self.state = ode_solution.y[:, -1]
        self.normalize_angular()
        # we were out of state space
        done = ode_solution.status == 1

        if done:
            if not self.done:
                # we just finished
                self.done = True
            else:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

        return self.state, self.reward(action), done, {}

    def vector_field(self, action: np.array, state: np.array) -> np.array:
        # in classical mechanics we can't control position directly
        x, v = np.split(state, 2)
        x_dot = v
        external_force = self.external_force(action, state)
        v_dot = self.acceleration(x, v, external_force)
        return np.concatenate([x_dot, v_dot])

    def external_force(self, action, state):
        raise NotImplementedError

    def acceleration(self, x, v, external_force):
        raise NotImplementedError

    def reward(self, action):
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError
