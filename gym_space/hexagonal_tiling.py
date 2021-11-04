import numpy as np
from typing import Union
import gym
import warnings

from gym_space import helpers

MAX_OBJ_TILES_RATIO = 0.6
PLANET_TILE_RATIO = 0.75
MAX_GOAL_CANDIDATES = 3

class HexagonalTiling:
    _debug = False

    def __init__(self, n_planets: int, world_size: float):
        if self._debug:
            warnings.warn("HexagonalTiling in DEBUG mode", UserWarning)
        self._np_random = None
        self.seed()

        self.n_planets = n_planets
        self.n_objects = self.n_planets + 2
        self.world_size = world_size

        # minimum number of tiles
        if self.n_planets == 2 or self._debug:
            min_tiles = self.n_objects
        else:
            min_tiles = int(np.ceil(self.n_objects / MAX_OBJ_TILES_RATIO))

        r, c, a = compute_tiling_rows_cols_a(min_tiles, world_size)

        self._rows = r
        self._cols = c
        self._n_tiles = r * c
        self._a = a
        self._hex_height = a * np.sqrt(3)
        self._hex_width = 2 * a
        self._tiling_width = 3 * a * (c - 1) / 2 + 2 * a
        self._tiles_coord = np.array([(row, col) for row in range(r) for col in range(c)])

        self._case_b = self._flip_xy = self._col_shift = None
        self._free_tiles_nrs = self._ship_tile_nr = self._goal_tile_nr = None

        self.planets_radius = self._hex_height / 2
        # if not self._debug:
        self.planets_radius *= PLANET_TILE_RATIO
        self.goal_radius = self.ship_radius = self.planets_radius / 2

    def seed(self, seed=None):
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self):
        # Case A: row starts with top hexagon (tile zero)
        # X X X
        #  X X X ...

        # Case B: row starts with bottom hexagon (tile zero)
        #  X X X
        # X X X ...

        self._goal_tile_nr = None

        if self._debug:
            self._case_b = self._flip_xy = False
            self._col_shift = np.zeros(self._cols)

        else:
            self._case_b, self._flip_xy = self._np_random.uniform(size=2) < 0.5
            self._col_shift = np.cumsum(self._np_random.uniform(size=self._cols))
            free_x_space = self.world_size - self._tiling_width
            self._col_shift *= free_x_space / self._col_shift[-1]

        # ship and planets
        if self.n_planets == 2 and self._np_random.uniform() < 0.25:
            # In case of 2 planets 2/3 of possibilities are dull.
            # This is a way to ensure that on average only half
            # of the cases are boring.
            diagonal_cases = [
                # diagonal
                (1, 0, 3),
                (2, 0, 3),
                # anti-diagonal
                (0, 1, 2),
                (3, 1, 2),
            ]
            tiles_nrs = np.array(diagonal_cases[self._np_random.randint(4)])
        else:
            tiles_nrs = self._np_random.choice(self._n_tiles, size=self.n_objects - 1, replace=False)
        self._ship_tile_nr = tiles_nrs[0]
        self._free_tiles_nrs = [i for i in range(self._n_tiles) if i not in tiles_nrs]
        radii = np.array([self.ship_radius] + self.n_planets * [self.planets_radius])
        return self._sample_disc_from_tile(tiles_nrs, radii)

    def find_new_goal(self):
        self._reset_goal_tile_nr()
        return self._sample_disc_from_tile(self._goal_tile_nr, self.goal_radius)

    def _reset_goal_tile_nr(self):
        # first goal was achieved and we're generating some subsequent goal
        if self._goal_tile_nr is not None:
            previous_ship_tile_nr = self._ship_tile_nr
            # ship no longer occupies it's previous tile
            self._free_tiles_nrs.append(previous_ship_tile_nr)
            # ship now occupies the same tile as current goal
            self._ship_tile_nr = self._goal_tile_nr

        if self._np_random.uniform() < 0.25:
            self._goal_tile_nr = self._ship_tile_nr
            return 

        n_candidates = min(MAX_GOAL_CANDIDATES, len(self._free_tiles_nrs))
        tile_nr_idx_candidates = self._np_random.choice(len(self._free_tiles_nrs), size=n_candidates, replace=False)

        # choose most distant
        max_ship_taxi_dist = -np.inf
        max_ship_taxi_dist_idx = None
        for idx in tile_nr_idx_candidates:
            tile_row, tile_col = self._tiles_coord[self._free_tiles_nrs[idx]]
            ship_tile_row, ship_tile_col = self._tiles_coord[self._ship_tile_nr]
            ship_taxi_dist = abs(tile_row - ship_tile_row) + abs(tile_col - ship_tile_col)
            if ship_taxi_dist > max_ship_taxi_dist:
                max_ship_taxi_dist = ship_taxi_dist
                max_ship_taxi_dist_idx = idx

        new_goal_tile_nr = self._free_tiles_nrs.pop(max_ship_taxi_dist_idx)

        self._goal_tile_nr = new_goal_tile_nr

    def _sample_disc_from_tile(self, tile_nr: Union[int, np.ndarray], radius: Union[float, np.ndarray]):
        center_pos = self._tile_center_pos(tile_nr)
        noise_radius = self._hex_height / 2 - radius
        noise = helpers.uniform_disk_distribution(self._np_random, noise_radius)
        return center_pos + noise

    def _tile_center_pos(self, tile_nr: Union[int, np.ndarray]):
        # we consider hexagons of the following shape
        #    ____
        #  /     \
        #  \____/
        #
        tiles = self._tiles_coord[tile_nr]
        row_nrs = tiles[..., 0]
        col_nrs = tiles[..., 1]
        tile_zero_pos_x = - self.world_size / 2 + self._hex_width / 2
        tile_zero_pos_y = self.world_size / 2 - self._hex_height / 2
        if self._case_b:
            tile_zero_pos_y -= self._hex_height / 2
        x_shifts = col_nrs * 1.5 * self._a + self._col_shift[col_nrs]
        y_shifts_due_rows = - row_nrs * self._hex_height
        y_shifts_due_cols = - (col_nrs % 2) * self._hex_height / 2
        if self._case_b:
            y_shifts_due_cols *= -1
        y_shifts = y_shifts_due_rows + y_shifts_due_cols
        center_pos = np.stack([tile_zero_pos_x + x_shifts, tile_zero_pos_y + y_shifts], axis=-1)
        if self._flip_xy:
            return center_pos[..., ::-1]
        return center_pos


def compute_tiling_rows_cols_a(min_tiles: int, world_size: float):
    m = min_tiles
    r_ = (
            np.sqrt(72 * np.sqrt(3) * m - 6 * np.sqrt(3) + 12) / 12 -
            1 / 4 + np.sqrt(3) / 12
    )
    r = int(np.ceil(r_))
    while True:
        c = int(np.floor(2 * np.sqrt(3) * r / 3 - 1 / 3 + np.sqrt(3) / 3))
        if r * c >= m:
            break
        r += 1
    a = 2 * np.sqrt(3) * world_size / (3 * (2 * r + 1))
    return r, c, a