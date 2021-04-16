import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC
from functools import partial


class ODEEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, lows, highs, n_actions, dt):
        # TODO: How should we define state_space, to make sure we are always inside?
        #   Depending on dt and vector field, the last observation can be outside.
        if np.isscalar(lows):
            assert np.isscalar(highs)
            self.state_space = spaces.Box(lows, highs, shape=(1,))
        else:
            assert lows.shape == highs.shape
            assert len(lows.shape) == 1
            self.state_space = spaces.Box(lows, highs)

        self.state_space_dim = self.state_space.shape[0]
        assert self.state_space_dim >= 1
        self.terminal_events = []
        for i, (low, high) in enumerate(
            zip(self.state_space.low, self.state_space.high)
        ):
            def low_event(low, i, state):
                return state[i] - low
            def high_event(high, i, state):
                return high - state[i]
            self.add_terminal_event(partial(low_event, low, i))
            self.add_terminal_event(partial(high_event, high, i))

        if not np.isscalar(n_actions):
            assert len(n_actions) <= self.state_space_dim

        self.action_space = spaces.MultiDiscrete(n_actions)
        self.dt = dt  # seconds between observations
        self.state = None
        self.done = False
        self.np_random = None
        self.seed()

    def add_terminal_event(self, event_function):
        def event(_t, state):
            return event_function(state)

        # we want to solve ODE for the entire duration of the last step
        event.terminal = False
        self.terminal_events.append(event)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action=} invalid"

        control = self.action_to_control(action)
        assert control.shape == (self.state_space_dim,)

        def vector_field_(_t, state):
            return self.vector_field(state, control)

        ode_solution = solve_ivp(
            vector_field_,
            t_span=(0, self.dt),
            y0=self.state,
            # events=self.terminal_events,
        )
        assert ode_solution.success
        self.state = ode_solution.y[:, -1]
        # we were out of state space
        done = not self.state_space.contains(self.state)

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

    def reset(self):
        self.state = self.sample_initial_state()
        self.done = False
        return self.state

    def sample_initial_state(self):
        return self.state_space.sample()

    def action_to_control(self, action):
        raise NotImplementedError

    def vector_field(self, state: np.array, control: np.array) -> np.array:
        raise NotImplementedError

    def reward(self, action):
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError


class ClassicalMechanicsEnv(ODEEnv, ABC):
    def __init__(self, position_lows, position_highs, n_actions, dt):
        if np.isscalar(position_lows):
            assert np.isscalar(position_highs)
            position_lows = np.array([position_lows])
            position_highs = np.array([position_highs])
        self.dof = len(position_lows)
        velocities_lows = np.full_like(position_lows, -np.inf, dtype=np.float32)
        velocities_highs = np.full_like(position_lows, np.inf, dtype=np.float32)
        lows = np.concatenate([position_lows, velocities_lows], dtype=np.float32)
        highs = np.concatenate([position_highs, velocities_highs], dtype=np.float32)
        super().__init__(lows, highs, n_actions, dt)

    def vector_field(self, state: np.array, control: np.array) -> np.array:
        # in classical mechanics we can't control position directly
        _, control = np.split(control, 2)
        x, v = np.split(state, 2)
        x_dot = v
        v_dot = self.acceleration(x, v, control)
        return np.concatenate([x_dot, v_dot])

    def action_to_control(self, action):
        action_forces = self.action_to_forces(action)
        assert action_forces.shape == (self.dof,)
        # in classical mechanics we can't control velocity directly
        return np.concatenate(
            [
                np.zeros(self.dof),
                action_forces,
            ]
        )

    def action_to_forces(self, action):
        raise NotImplementedError

    def acceleration(self, x, v, control):
        raise NotImplementedError

