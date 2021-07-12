from abc import ABC
import numpy as np
from scipy.stats import multivariate_normal

from gym_space.planet import Planet
from gym_space.ship_params import ShipParams
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv

PLANETS_WORLD_AREA_RATIO = 0.5
MAX_PLANETS = 5

class GoalEnv(SpaceshipEnv, ABC):
    _total_planets_mass = 1e9
    _max_position_sample_tries = 30

    def __init__(
        self,
        n_planets: int = 3,
        survival_reward_scale: float = 0.5,
        goal_dist_reward_scale: float = 0.4,
        economy_reward_scale: float = 0.1,
        goal_dist_reward_std: float = 0.1,
        test_env: bool = False,
        renderer_kwargs: dict = None,
    ):
        assert n_planets <= MAX_PLANETS, f"Current sampling algorithm can't accommodate more that {MAX_PLANETS} planets"
        self._n_planets = n_planets
        world_size = 3.0
        """
        S is PLANETS_WORLD_AREA_RATIO
        n is n_planets + 2
        W is world_size
        r is planets_radius
        
        S = n pi (1.5 r)^2 / (W - r)^2
        (W-r)^2 S = n pi 2.25 r^2
        SW^2 - 2SWr + Sr^2 = n pi 2.25 r^2
        
        (S - n pi 2.25) r^2 - 2SWr + SW^2 = 0
        
        delta = (-2SW)^2 - 4 (S - n pi 2.25) SW^2 =
                4 S^2 W^2 - 4 S^2 W^2 + n pi 9 S W^2 =
                n pi 9 S W^2
        
        sqrt(delta) = 3 sqrt(n pi S) W
        r = ( 2SW +- 3 sqrt(n pi S) W ) / (2S - n pi 4.5)
          = W (2S +- 3 sqrt(n pi S)) / (2S - n pi 4.5)
        
        For 0 < S <= 1 r will be positive with - and negative with + sign, so
        r = W (2S - 3 sqrt(n pi S)) / (2S - n pi 4.5)
        """
        n_objects = n_planets + 2
        self.planets_radius = (
            world_size *
            (
                2 * PLANETS_WORLD_AREA_RATIO -
                3 * np.sqrt(n_objects * np.pi * PLANETS_WORLD_AREA_RATIO)
            ) /
            (2 * PLANETS_WORLD_AREA_RATIO - n_objects * np.pi * 4.5)
        )
        s = n_objects * np.pi * (1.5 * self.planets_radius) ** 2 / (world_size - self.planets_radius) ** 2
        assert np.isclose(s, PLANETS_WORLD_AREA_RATIO)
        planets_mass = self._total_planets_mass / n_planets

        planets = [
            Planet(mass=planets_mass, radius=self.planets_radius)
            for _ in range(self._n_planets)
        ]
        ship = ShipParams(
            mass=1, moi=0.05, max_engine_force=0.3, max_thruster_force=0.05
        )

        assert (
            survival_reward_scale >= 0
            and goal_dist_reward_scale >= 0
            and economy_reward_scale >= 0
        ), "Reward scales have to be positive"
        assert (
            survival_reward_scale + goal_dist_reward_scale + economy_reward_scale == 1
        ), "Reward scales have to sum up to 1.0"

        self.survival_reward_scale = survival_reward_scale
        self.goal_dist_reward_scale = goal_dist_reward_scale
        self.economy_reward_scale = economy_reward_scale
        self.goal_dist_reward_std = goal_dist_reward_std

        self.test_env = test_env
        self._normal_pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * self.goal_dist_reward_std).pdf
        self._max_normal_pdf = self._normal_pdf(np.zeros(2))

        super().__init__(
            ship_params=ship,
            planets=planets,
            world_size=world_size,
            step_size=0.07,
            max_abs_vel_angle=5.0,
            vel_xy_std=np.ones(2),
            with_lidar=True,
            with_goal=True,
            renderer_kwargs=renderer_kwargs,
        )

    def _sample_positions(self):
        positions = []
        n_total_tries = 0
        n_tries = 0
        while len(positions) < self._n_planets + 2:
            n_tries += 1
            n_total_tries += 1
            assert n_total_tries < 1e4
            if n_tries > self._max_position_sample_tries:
                # let's start again
                n_tries = 0
                positions = []
                continue
            new_pos = self._np_random.uniform(-1.0, 1.0, 2) * (
                self.world_size / 2 - 1.5 * self.planets_radius
            )
            for other_pos in positions:
                if np.linalg.norm(other_pos - new_pos) < 3 * self.planets_radius:
                    break
            else:
                positions.append(new_pos)
                if self.test_env and len(positions) == self._n_planets + 2:
                    ship_pos, goal_pos = positions[-2:]
                    if np.linalg.norm(ship_pos - goal_pos) < 0.7 * (self.world_size - self.planets_radius):
                        positions = positions[:-2]
                        continue
        print(n_total_tries)
        return positions

    def _reset(self):
        *planets_pos, ship_pos, self.goal_pos = self._sample_positions()
        for pos, planet in zip(planets_pos, self.planets):
            planet.center_pos = pos
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_vel_angle
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self._ship_state.set(ship_pos, ship_angle, velocities_xy, angular_velocity)

    def _reward(self) -> float:
        reward = (
            self.survival_reward_scale
            + self.goal_dist_reward_scale * self._goal_dist_reward()
            + self.economy_reward_scale * self._economy_reward()
        )
        assert 0.0 <= reward <= 1
        return reward

    def _goal_dist_reward(self) -> float:
        r = self._normal_pdf(self._ship_state.pos_xy - self.goal_pos) / self._max_normal_pdf
        assert 0.0 <= r <= 1
        return r

    def _economy_reward(self) -> float:
        action_norm = np.linalg.norm(self.last_action)
        max_action_norm = np.sqrt(2)
        normalized_action_norm = action_norm / max_action_norm
        # function of action norm decreasing from 1 (no action) to 0 (max action)
        r = 1 - normalized_action_norm
        assert 0 <= r <= 1
        return r

class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass
