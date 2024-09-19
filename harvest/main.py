from harvest.game import HarvestGame
from harvest.helpers import PROBABILITY_FUNCS


if __name__ == "__main__":
    game = HarvestGame(
        map = HarvestGame.parse_map(
            [
                "@@@@@@@@@@",
                "@.OO.OOOO@",
                "@OOOOOOOO@",
                "@OOOOOOOO@",
                "@.OO.OOOO@",
                "@@@@@@@@@@",
            ],
        ),
        spawn_points = [(1, 1), (1, 4), (4, 1), (4, 4)],
        seed = ord('A')
    )

    for i in range(1):
        game.add_player(str(i), "To maximise your own points.")

    game.experiment(
        output_file = "expts/4-4x8MO-Original-12-A-1.json", 
        probability_func = PROBABILITY_FUNCS["original"], 
        desired_apple_num = 12, 
        limit = 4
    )