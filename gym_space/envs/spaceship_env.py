import gym
from gym.spaces import Discrete, Box
from gym_space.planet import Planet, planets_min_max
from gym_space.ship import Ship
from gym_space.action_space import ActionSpaceType
from gym_space.rewards import Rewards, LandOnPlanetRewards, OrbitPlanetRewards, NoRewards
from gym_space.rendering import Renderer
from gym_space.helpers import angle_to_unit_vector
from typing import List
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

# TODO: seed()
class SpaceshipEnv(gym.Env):
    """Spaceship starts away from any planet, it's goal is to land on planet nr 0"""

    def __init__(
        self,
        ship: Ship,
        planets: List[Planet],
        action_space_type: ActionSpaceType,
        rewards: Rewards,
        step_size: float,
        control_thruster: bool = True,
    ):
        self.ship = ship
        self.planets = planets
        self._world_min, self._world_max = None, None
        self._init_world_min_max()
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf, 0.0, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, 2 * np.pi, np.inf, np.inf, np.inf])
        )
        self.action_space_type = action_space_type
        self.rewards = rewards
        self.rewards.destination_planet = planets[0]
        self.control_thruster = control_thruster
        self.step_size = step_size
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 60 / step_size  # one minute in one second
        }
        self._init_action_space()
        self._boundary_events = None
        self._init_boundary_events()
        self.state = None
        self.last_action = None
        self._renderer = None

    def _init_world_min_max(self):
        if len(self.planets) > 1:
            self._world_min, self._world_max = planets_min_max(self.planets)
        else:
            planet = self.planets[0]
            self._world_min = planet.center_pos - 4 * planet.radius
            self._world_max = planet.center_pos + 4 * planet.radius

    def _init_action_space(self):
        if self.action_space_type is ActionSpaceType.DISCRETE:
            # engine can be turned on or off: 2 options
            if self.control_thruster:
                # thruster can act clockwise, doesn't act or act counter-clockwise: 3 options
                self.action_space = Discrete(2 * 3)
            else:
                self.action_space = Discrete(2)
        elif self.action_space_type is ActionSpaceType.CONTINUOUS:
            # engine action is in [0, 1], thruster action in [-1, 1]
            if self.control_thruster:
                self.action_space = Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]))
            else:
                self.action_space = Box(low=np.array([0.0]), high=np.array([1.0]))

    def _init_boundary_events(self):
        def event(planet: Planet, _t, state):
            ship_xy_pos = state[:2]
            return planet.distance(ship_xy_pos)

        self._boundary_events = []
        for planet_ in self.planets:
            event_ = partial(event, planet_)
            event_.terminal = True
            self._boundary_events.append(event_)

    def _external_force_and_torque(self, action: np.array, state: np.array):
        engine_action, thruster_action = action
        engine_force = engine_action * self.ship.max_engine_force
        thruster_torque = thruster_action * self.ship.max_thruster_torque
        angle = state[2]
        engine_force_direction = - angle_to_unit_vector(angle)
        force_xy = engine_force_direction * engine_force
        return np.array([*force_xy, thruster_torque])

    def _vector_field(self, action, _time, state: np.array):
        external_force_and_torque = self._external_force_and_torque(action, state)

        position, velocity = np.split(state, 2)
        external_force_xy, external_torque = np.split(external_force_and_torque, [2])

        force_xy = external_force_xy
        for planet in self.planets:
            force_xy += planet.gravity(position[:2], self.ship.mass)
        acceleration_xy = force_xy / self.ship.mass

        torque = external_torque
        acceleration_angle = torque / self.ship.moi

        acceleration = np.concatenate([acceleration_xy, acceleration_angle])
        return np.concatenate([velocity, acceleration])

    def _update_state(self, action):
        ode_solution = solve_ivp(
            partial(self._vector_field, action),
            t_span=(0, self.step_size),
            y0=self.state,
            events=self._boundary_events,
        )
        assert ode_solution.success, ode_solution.message
        self.state = ode_solution.y[:, -1]
        self._normalize_angle()
        done = ode_solution.status == 1
        return done

    def _normalize_angle(self):
        self.state[2] %= 2 * np.pi

    @staticmethod
    def _discrete_to_continuous_action(discrete_action: int):
        engine_action = float(discrete_action % 2)
        thruster_action = float(discrete_action // 2 - 1)
        return np.array([engine_action, thruster_action])

    def _sample_initial_state(self):
        try_nr = 0
        while True:
            if try_nr > 100:
                raise ValueError("Could not find correct initial state")
            try_nr += 1
            pos_xy = np.random.uniform(low=self._world_min, high=self._world_max)
            for planet in self.planets:
                if planet.distance(pos_xy) < 0:
                    break
                if np.linalg.norm(planet.gravity(pos_xy, self.ship.mass)) > self.ship.max_engine_force:
                    break
            else:
                break
        pos_angle = np.random.uniform(0, 2 * np.pi)
        velocities_xy = np.random.normal(size=2) * 10
        velocity_angle = 0.0
        return np.array([*pos_xy, pos_angle, *velocities_xy, velocity_angle])

    def step(self, action_):
        assert self.action_space.contains(action_)
        if self.action_space_type is ActionSpaceType.DISCRETE:
            action = self._discrete_to_continuous_action(action_)
        else:
            action = action_
        self.last_action = action

        done = self._update_state(action)
        reward = self.rewards.reward(self.state, action, done)
        return self.state, reward, done, {}

    def reset(self):
        self.state = self._sample_initial_state()
        return self.state

    def render(self, mode="human"):
        if self._renderer is None:
            self._renderer = Renderer(15, self.planets, self._world_min, self._world_max)

        engine_active = False
        if self.last_action is not None:
            engine_active = self.last_action[0] > 0.1
        return self._renderer.render(self.state[:3], engine_active, mode)


class SpaceshipLandV0(SpaceshipEnv):
    def __init__(self):
        MAX_EPISODE_SECONDS = 3_600
        STEP_SIZE = 1.0
        MAX_EPISODE_STEPS = int(MAX_EPISODE_SECONDS / STEP_SIZE)
        PLANETS = [Planet(center_pos=np.zeros(2), mass=5.972e24, radius=6.371e6)]
        SHIP = Ship(mass=5.5e4, moi=1, max_engine_force=7.607e6, max_thruster_torque=1e-2)
        REWARDS = LandOnPlanetRewards(
            destination_reward=10_000,
            max_destination_distance_penalty=400,
            max_reasonable_distance=5e6,
            max_fuel_penalty=100,
            max_landing_velocity_penalty=2_500,
            max_landing_angle_penalty=2_500,
            max_episode_steps=MAX_EPISODE_STEPS
        )

        super().__init__(
            ship=SHIP,
            planets=PLANETS,
            action_space_type=ActionSpaceType.DISCRETE,
            rewards=REWARDS,
            step_size=STEP_SIZE
        )


class SpaceshipOrbitEnv(SpaceshipEnv):
    def __init__(self, action_space_type: ActionSpaceType):
        STEP_SIZE = 10.0
        PLANETS = [Planet(center_pos=np.zeros(2), mass=5.972e24, radius=6.371e6)]
        SHIP = Ship(mass=1e4, moi=1, max_engine_force=1e5, max_thruster_torque=1e-5)
        REWARDS = OrbitPlanetRewards(PLANETS[0])

        super().__init__(
            ship=SHIP,
            planets=PLANETS,
            action_space_type=action_space_type,
            rewards=REWARDS,
            step_size=STEP_SIZE
        )

    def _sample_initial_state(self):
        planet_angle = np.random.uniform(0, 2 * np.pi)
        ship_angle = (planet_angle - np.pi/2) % (2 * np.pi)
        pos_xy = angle_to_unit_vector(planet_angle) * self.planets[0].radius * 1.3
        velocities_xy = - angle_to_unit_vector(ship_angle) * 5e3
        return np.array([*pos_xy, ship_angle, *velocities_xy, 0.0])


class SpaceshipOrbitDiscreteV0(SpaceshipOrbitEnv):
    def __init__(self):
        super().__init__(ActionSpaceType.DISCRETE)


class SpaceshipOrbitContinuousV0(SpaceshipOrbitEnv):
    def __init__(self):
        super().__init__(ActionSpaceType.CONTINUOUS)


# no rewards, just to show env with two planets
class SpaceshipTwoPlanetsV0(SpaceshipEnv):
    def __init__(self):
        STEP_SIZE = 10.0
        PLANETS = [
            Planet(center_pos=np.ones(2) * 2e7, mass=5.972e24, radius=6.371e6),
            Planet(center_pos=-np.ones(2) * 2e7, mass=2e24, radius=4e6),
        ]
        SHIP = Ship(mass=1e4, moi=1, max_engine_force=1e5, max_thruster_torque=1e-5)
        REWARDS = NoRewards()

        super().__init__(
            ship=SHIP,
            planets=PLANETS,
            action_space_type=ActionSpaceType.DISCRETE,
            rewards=REWARDS,
            step_size=STEP_SIZE
        )