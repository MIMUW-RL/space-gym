from abc import ABC
import numpy as np

from gym_space.planet import Planet
from gym_space.ship import Ship
from .spaceship_env import SpaceshipEnv, DiscreteSpaceshipEnv, ContinuousSpaceshipEnv


class GoalEnv(SpaceshipEnv, ABC):
    max_episode_steps = 300
    _n_planets = 4
    _planets_radius = 0.3
    _planets_mass = 2e8
    _max_position_sample_tries = 30


    def __init__(
        self,
        survival_reward_scale: float = 0.5,
        goal_dist_reward_scale: float = 0.4,
        economy_reward_scale: float = 0.1,
        test_env: bool = False
    ):
        planets = [Planet(mass=self._planets_mass, radius=self._planets_radius) for _ in range(self._n_planets)]
        ship = Ship(mass=1, moi=0.05, max_engine_force=0.3, max_thruster_torque=0.05)

        assert survival_reward_scale >= 0 and goal_dist_reward_scale >= 0 and economy_reward_scale >= 0, "Reward scales have to be positive"
        assert survival_reward_scale + goal_dist_reward_scale + economy_reward_scale == 1, "Reward scales have to sum up to 1.0"

        self.survival_reward_scale = survival_reward_scale
        self.goal_dist_reward_scale = goal_dist_reward_scale
        self.economy_reward_scale = economy_reward_scale

        self.test_env = test_env

        super().__init__(
            ship=ship,
            planets=planets,
            world_size=3.0,
            step_size=0.07,
            max_abs_angular_velocity=5.0,
            velocity_xy_std=np.ones(2),
            with_lidar=True,
            with_goal=True
        )

    def _sample_positions(self):
        positions = []
        n_tries = 0
        while len(positions) < self._n_planets + 2:
            n_tries += 1
            if n_tries > self._max_position_sample_tries:
                # let's start again
                n_tries = 0
                positions = []
                continue
            new_pos = self._np_random.uniform(-1.0, 1.0, 2) * (self.world_size / 2 - self._planets_radius)
            for other_pos in positions:
                if np.linalg.norm(other_pos - new_pos) < 3 * self._planets_radius:
                    break
            else:
                positions.append(new_pos)
                if self.test_env and len(positions) == self._n_planets + 2:
                    ship_pos, goal_pos = positions[-2:]
                    if np.linalg.norm(ship_pos - goal_pos) < 0.7 * self.world_size:
                        positions = positions[:-2]
                        continue
        return positions

    def _reset(self):
        *planets_pos, ship_pos, self.goal_pos = self._sample_positions()
        for pos, planet in zip(planets_pos, self.planets):
            planet.center_pos = pos
        if self._renderer is not None:
            self._renderer.move_goal(self.goal_pos)
        ship_angle = self._np_random.uniform(0, 2 * np.pi)
        velocities_xy = self._np_random.standard_normal(2) * 0.07
        max_abs_ang_vel = 0.7 * self.max_abs_angular_velocity
        angular_velocity = self._np_random.standard_normal() * max_abs_ang_vel / 3
        angular_velocity = np.clip(angular_velocity, -max_abs_ang_vel, max_abs_ang_vel)
        self.internal_state = np.array([*ship_pos, ship_angle, *velocities_xy, angular_velocity])

    def _reward(self) -> float:
        survival_reward = 1.0

        true_goal_dist = np.linalg.norm(self.internal_state[:2] - self.goal_pos)
        max_goal_dist = self.world_size * np.sqrt(2)
        normalized_goal_dist = true_goal_dist / max_goal_dist
        # function of distance decreasing from 1 (no distance) to 0 (max distance)
        goal_dist_reward = (normalized_goal_dist - 1)**2
        assert 0.0 <= goal_dist_reward <= 1.0

        action_norm = np.linalg.norm(self.last_action)
        max_action_norm = np.sqrt(2)
        normalized_action_norm = action_norm / max_action_norm
        # function of action norm decreasing from 1 (no action) to 0 (max action)
        economy_reward = 1 - normalized_action_norm
        assert 0.0 <= economy_reward <= 1.0

        reward_unscaled = (
            self.survival_reward_scale * survival_reward +
            self.goal_dist_reward_scale * goal_dist_reward +
            self.economy_reward_scale * economy_reward
        )
        assert 0.0 <= reward_unscaled <= 1

        # sum of rewards should be <= 100.0
        return 100 * reward_unscaled / self.max_episode_steps


class GoalDiscreteEnv(GoalEnv, DiscreteSpaceshipEnv):
    pass


class GoalContinuousEnv(GoalEnv, ContinuousSpaceshipEnv):
    pass