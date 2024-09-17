from datetime import datetime, timedelta
from typing import override

from concordia.language_model.gpt_model import GptLanguageModel
from concordia.associative_memory.formative_memories import FormativeMemoryFactory
from concordia.associative_memory.importance_function import ConstantImportanceModel
from concordia.clocks.game_clock import FixedIntervalClock
from concordia.associative_memory.blank_memories import MemoryFactory
from concordia.typing.entity import ActionSpec, OutputType
from concordia.typing.component import Component
from concordia.components import agent as agent_components

from harvest.helpers import HARVEST_RULES, OPENAI_API_KEY, HarvestTile


class HarvestPlayer(Component):
    """
    A player of the Harvest game. This class keeps track of a player's position and score, provides
    convenient interfaces for reading the (partial) game state, and endpoints to execute actions.

    **A [HarvestPlayer] object should only be instantiated by a [HarvestGame] object.**
    """

    def __init__(
        self,
        name: str,
        game: 'HarvestGame',
        position: tuple[int, int],
        goal: str,
        score: int = 0,
        sight: int = -1,
    ) -> None:
        assert len(str) == 1, "Player name must be a single character."
        self.name = name
        self.game = game
        self.position = position
        self.score = score
        self.sight = sight
    @override
    def state(self) -> str:
        """
        Generates a string representation of the player's state in the game. This is primarily used
        for prompting the LLM.
        """

        # Report other players' locations and scores.
        state = '\n'.join([
                f"{player.name}: {player.position}, with {player.score} points."
                for player in self.game.players.values()
                if player.name != self._player_name
            ]) + '\n'
        
        # Report the player's own location and score.
        tile = self.game[self.position]
        state += f"You are at {self.position}, with {self._reward} points, " + \
            f"standing on \"{tile.value}\", which is a {tile.__str__()}. " + \
            "Your location is marked with an \"#\" symbol on the map.\n"

        # Report apples.
        # TODO: Don't report apples out of sight.
        state += f"Apple locations: {[(x, y) for (x, y), cell in self.game if cell == HarvestTile.APPLE]}\n"

        # Report the map.
        map = self.game.view_map() if self.sight == -1 else self.game.partial_view(self.position, self.sight, self.position)
        state += f"Here is the map: \n{map}"

        return state

    def execute(self, path: list[tuple[int, int]]):
        """
        Executes the movement of a player along a path in the given map, updating
        the map and the player.

        PRE: We assume that the path is well-formed.
        """

        for x, y in path:
            history_item = { 
                "type": "move", "player": self.name, "from": self.position, "to": (x, y), 
                "ate": False, "score_after": self.score 
            }

            if map[x, y] == HarvestTile.APPLE:
                map[x, y] = HarvestTile.ORCHARD
                self.score += 1
                history_item["ate"] = True
                history_item["score_after"] = self.score

            self.position = (x, y)
            self.game.history.append(history_item)
