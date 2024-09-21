import json
from enum import Enum
from threading import RLock
from typing import Callable, Generator

import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm

from harvest.helpers import KERNEL, PROBABILITY_FUNCS, HarvestMap, HarvestTile
from harvest.player import HarvestPlayer


class HarvestGame:
    """
    A mixed-motive, multi-agent game where agents coexist in a 2D grid world. There are orchards in
    the world which spawn apples at a rate roughly proportional to the number of apples already in
    the region. Agents must balance between harvesting apples and gaining points for themselves / 
    in the short term, and waiting for apples to respawn and gaining more points in the long term / 
    for the social collective.

    State Space: 2D Grid World, Agent Locations, Apples.
    Action Space: Movement in Ordinal Directions.

    An instance of [HarvestGame] is only meant to be used once, since it keeps track of game 
    history. It must be instantiated before all other Harvest objects. If the seed is set, all other
    calls to np.random should be deterministic at call-site, to avoid unreproducible behavior.
    """

    def __init__(
        self,
        map: HarvestMap,
        spawn_points: list[tuple[int, int]],
        seed: int = None,
    ) -> None:
        self._lock = RLock()

        self.history = []
        self.players = {}
        self.players: dict[str, HarvestPlayer]

        assert map.ndim == 2, "Map must be a 2D numpy array."
        assert np.all(map != HarvestTile.APPLE), "Initial map must not contain apples."
        self.map = map

        assert len(spawn_points) > 0, "At least one spawn point must be provided."
        assert len(spawn_points) <= map.size, "Too many spawn points for map size."
        assert all(self.in_bounds(point) for point in spawn_points), "Spawn points must be in bounds."
        assert all(map[point] not in [HarvestTile.ORCHARD, HarvestTile.WALL] for point in spawn_points), "Spawn points must not be orchards or walls."
        self._spawn_points = spawn_points

        if seed is not None: 
            np.random.seed(seed)
        
        self.history.append({ "type": "init", "map": np.copy(self.map).tolist(), "seed": seed })

    def __getitem__(self, key: tuple[int, int]) -> HarvestTile:
        """Returns the tile at the given coordinates (in (x, y) form) in [self.map]."""
        with self._lock: return self.map[key[::-1]]
    
    def __setitem__(self, key: tuple[int, int], value: HarvestTile) -> None:
        """Sets the tile at the given coordinates (in (x, y) form) in [self.map]."""
        with self._lock: self.map[key[::-1]] = value

    def __iter__(self) -> Generator[tuple[tuple[int, int], str], None, None]:
        """
        Iterates over the map, left-to-right, top-to-bottom, with coordinates. This method takes a
        copy of the map at call for thread safety, so may diverge from the original map!
        """
        with self._lock:
            snapshot = np.copy(self.map)

        for y in range(snapshot.shape[0]):
            for x in range(snapshot.shape[1]):
                yield (x, y), snapshot[y][x]

    @staticmethod
    def parse_map(map_string: list[str]) -> HarvestMap:
        """Parses a list of strings into a 2D numpy array of HarvestTiles."""

        return np.array([
            [HarvestTile(char) for char in row]
            for row in map_string
        ], dtype = np.dtype(HarvestTile))
    
    @property
    def width(self) -> int:
        """Returns the width of the map."""
        with self._lock: return self.map.shape[1]
    
    @property
    def height(self) -> int:
        """Returns the height of the map."""
        with self._lock: return self.map.shape[0]
    
    def in_bounds(self, key: tuple[int, int]) -> bool:
        """Returns whether the given coordinates (in (x, y) form) are within bounds."""
        with self._lock: return 0 <= key[0] < self.width and 0 <= key[1] < self.height
    
    def setup(self, desired_apple_num: int | None = None) -> None:
        """Sets up the game by spawning apples in the orchards.

        Args:
            desired_apple_num (int, optional): The number of apples to spawn. Defaults to 
                `width * height / 2`.
        """
        if desired_apple_num is None:
            desired_apple_num = self.width * self.height // 2

        assert desired_apple_num <= self.map.size, "Too many apples for map size."

        with self._lock:
            orchards = np.argwhere(self.map == HarvestTile.ORCHARD.value)
            selected_indices = np.random.choice(len(orchards), size = desired_apple_num, replace = False)
            apples = orchards[selected_indices]
            self.map[apples[:, 0], apples[:, 1]] = HarvestTile.APPLE

            self.history.append({ "type": "setup_map", "apples": apples.tolist(), "map": np.copy(self.map).tolist() })

    def advance(
        self,
        probability_func: Callable[[np.ndarray[int]], np.ndarray[float]] = PROBABILITY_FUNCS['original'],
    ) -> np.ndarray[tuple[int, int]]:
        """Advances the state of the game by one step. In human-speak, this means apples spawn.

        Args:
            probability_func (callable[[np.ndarray], np.ndarray[float]]): A function which, given
                a 2D array representing the number of apples in the L2-norm neighborhood of radius
                2 around each cell, returns a 2D array representing the probability of an apple
                spawning in each cell. Defaults to the original probability function stated in 
                `Melting Pot 2.0, arXiv:2211.13746`.

        Returns:
            np.ndarray[tuple[int, int]]: The coordinates of new apples.
        """
        
        with self._lock:
            neighborhood = convolve(
                (self.map == 'A').astype(int), KERNEL,
                mode = 'constant', cval = 0
            )

            probabilities = probability_func(neighborhood)

            spawns = np.random.random(self.map.shape) < probabilities
            spawns = spawns & (self.map == 'O')

            self.map[spawns] = HarvestTile.APPLE
            spawn_coords = np.argwhere(spawns).tolist()
            self.history.append({ "type": "advance", "new_apples": spawn_coords, "map": np.copy(self.map).tolist() })
            return spawn_coords
        
    def view_map(self, mark: tuple[int, int] | None = None) -> str:
        """Returns a string representation of the map."""

        if mark is not None:
            assert self.in_bounds(mark)

        with self._lock:
            return "\n".join(
                "".join(
                    '#' if mark is not None and (row_num, col_num) == mark else repr(tile)
                    for col_num, tile in enumerate(row)
                )
                for row_num, row in enumerate(self.map)
            )
    
    def view_partial_map(
            self, center: tuple[int, int], radius: int,
            mark: tuple[int, int] | None = None
    ) -> str:
        """Returns a string representation of a partial map centered around a point."""

        x, y = center
        if mark is not None:
            assert self.in_bounds(mark)

        with self._lock:
            start_row, start_col = max(0, y - radius), max(0, x - radius)
            end_row, end_col = min(self.height, y + radius + 1), min(self.width, x + radius + 1)
            
            return "\n".join(
                "".join(
                    '#' if mark is not None and (j, i) == mark else repr(self.map[i, j])
                    for j in range(start_col, end_col)
                )
                for i in range(start_row, end_row)
            )
    
    def add_player(self, name: str, goal: str, sight: int = -1) -> HarvestPlayer:
        """
        Generates a new [HarvestPlayer] and returns it.

        Args:
            name (str): The name of the player.
            goal (str): The goal for this player.
            sight (int, optional): How far this player can see. Defaults to -1, which means infinite
                sight.

        Returns:
            HarvestPlayer: The newly created player.
        """

        assert name not in self.players, "Player name must be unique."
        assert len(self.players) < len(self._spawn_points), "No more spawn points available."

        with self._lock:
            spawn_point = self._spawn_points[len(self.players)]
            player = HarvestPlayer(name, self, spawn_point, sight = sight, goal = goal)
            self.players[name] = player
            return player

    def experiment(
            self, output_file: str | None = None,
            probability_func: Callable[[np.ndarray[int]], np.ndarray[float]] = PROBABILITY_FUNCS["original"],
            desired_apple_num: int | None = None, limit: int = 50
    ) -> None:
        """Runs this [HarvestGame] as an experiment."""

        self.setup(desired_apple_num)
        self.history.append({
            "type": "round",
            "num": 0,
            "map": np.copy(self.map).tolist(),
            "players": [it.describe() for it in self.players.values()],
        })

        for round_num in tqdm(range(1, limit + 1), desc = "Experimenting..."):
            for player in self.players.values():
                player: HarvestPlayer
                player.run()

            self.advance(probability_func)

            self.history.append({
                "type": "round",
                "num": round_num,
                "map": np.copy(self.map).tolist(),
                "players": [it.describe() for it in self.players.values()],
            })

        with open(output_file, 'w') as f:
            json.dump(self.history, f)
