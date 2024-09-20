from typing import Callable

import numpy as np
from harvest.game import HarvestGame
from harvest.helpers import PROBABILITY_FUNCS, HarvestMap


def execute_experiment(
    map: HarvestMap,
    spawn_points: list[tuple[int, int]],
    seed: int,
    player_num: int,
    player_goal: str,
    output_file: str,
    probability_func: Callable[[np.ndarray[int]], np.ndarray[float]],
    desired_apple_num: int,
    round_limit: int,
) -> None:
    """Executes a simple [HarvestGame] experiment.

    Args:
        map (list[str]): The map to be used. See [HarvestTile] for corresponding characters.
        spawn_points (list[tuple[int, int]]): The list of spawn locations for agents.
        seed (int): The RNG seed for [numpy]. *Does not influence LLMs.*
        player_num (int): The numer of players.
        player_goal (str): The goal of each player.
        output_file (str): The file to output logs to.
        probability_func (Callable[[np.ndarray[int]], np.ndarray[float]]): The probability function
            for advancing the map between rounds.
        desired_apple_num (int): The desired number of apples at the beginning.
        round_limit (int): The number of rounds to run the experiment for.
    """

    game = HarvestGame(
        map = map,
        spawn_points = spawn_points,
        seed = seed,
    )

    for i in range(player_num):
        game.add_player(str(i), player_goal)

    game.experiment(
        output_file = output_file, 
        probability_func = probability_func, 
        desired_apple_num = desired_apple_num, 
        limit = round_limit
    )
    

if __name__ == "__main__":
    execute_experiment(
        map = HarvestGame.parse_map([
            "@@@@@@@@@@",
            "@.OO.OOOO@",
            "@OOOOOOOO@",
            "@OOOOOOOO@",
            "@.OO.OOOO@",
            "@@@@@@@@@@",
        ]),
        spawn_points = [(1, 1), (1, 4), (4, 1), (4, 4)],
        seed = ord('B'),
        player_num = 4,
        player_goal = "To maximise your own points.",
        output_file = "expts/4-4x8MO-Original-12-B-4.json", 
        probability_func = PROBABILITY_FUNCS["original"], 
        desired_apple_num = 12, 
        round_limit = 50
    )
