from enum import Enum
import numpy as np
from os.path import dirname, join


class HarvestTile(str, Enum):
    """
    Enumerates the different types of tiles in the Harvest Game. The [value] of each enum is the
    character that represents the tile in the map.
    """

    WALL = '@' 
    """Represents a wall, which is impassable."""

    EMPTY = '.'
    """Represents an empty space, which is passable."""

    ORCHARD = 'O'
    """Represents an orchard, which is passable and can spawn apples."""

    APPLE = 'A'
    """Represents an apple, which can be harvested by agents."""

    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return self.value


HarvestMap = np.ndarray[HarvestTile]
"""Type alias for a 2D numpy array of [HarvestTile]s."""

KERNEL = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

PROBABILITY_FUNCS = {
    "original": lambda neighbourhood: np.select(
        [
            neighbourhood >= 3, neighbourhood == 2,
            neighbourhood == 1, neighbourhood == 0,
        ],
        [0.025, 0.005, 0.001, 0.0]
    ),
    "reduced": lambda neighbourhood: np.select(
        [
            neighbourhood >= 3, neighbourhood == 2,
            neighbourhood == 1, neighbourhood == 0,
        ],
        [0.015, 0.003, 0.0006, 0.0]
    ),
    "increased": lambda neighbourhood: np.select(
        [
            neighbourhood >= 3, neighbourhood == 2,
            neighbourhood == 1, neighbourhood == 0,
        ],
        [0.015, 0.003, 0.0006, 0.0]
    ),
    "zero": lambda neighbourhood: np.zeros_like(neighbourhood),
    "uniform": lambda neighbourhood: np.random.rand(*neighbourhood.shape),
}

OPENAI_API_KEY = open(join(dirname(__file__), "../.openai_key")).read().strip()