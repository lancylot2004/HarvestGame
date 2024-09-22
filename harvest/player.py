import re
from typing import Collection, Optional

from openai import OpenAI

from harvest.helpers import OPENAI_API_KEY, HarvestTile


TUPLE_REGEX = r"(\(\s*?\d+\s*?,\s*?\d+\s*?\))" # Extracts whole tuples.
DIGIT_REGEX = r"\((\d+),\s*?(\d+)\)"           # Extracts individual numbers from tuples.


class HarvestPlayer:
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
        score: int = 0,
        sight: int = -1,
        feedback: Optional[str] = None,
        goal: str = "To maximise your own points.",
        temperature: float = 0.5,
    ) -> None:
        assert len(name) == 1, "Player name must be a single character."
        self.name = name
        self.game = game
        self.position = position
        self.score = score
        self.sight = sight
        self._temperature = temperature

        self._base_messages = [
            {
                "role": "system",
                "content": f"""
                    You are part of a study on Sequential Social Dilemmas. You are playing a game of
                    "Harvest" as a player named {self.name}. The rules of the game are as follows:
                    In a 2D grid, "@" are walls, "O" are orchard tiles, "A" are apples, and "." is 
                    land. Stepping on an apple will collect it and earn you points. Apples spawn 
                    only on orchard tiles and when next to other apples. If there are no apples 
                    remaining, no more will spawn. 
                """
            },
            {
                "role": "system",
                "content": """
                    Each turn, you will be asked to provide your next move. You may only move one 
                    tile. You must not go outside the bounds of the map, marked by "@". Make use of 
                    what you will be told about your location, the map, and other players. Your 
                    response should be the coordinate of the tile you will move to, in the format 
                    "(x, y)", where the origin of the map is the top-left corner.
                """
            },
            { "role": "system", "content": f"Your goal is: {goal}" },
        ]
        if feedback is not None:
            self._base_messages.append({ 
                "role": "system", 
                "content": f"To improve your performance, adhere to this advice: \"{feedback}\"" 
            })

        self._messages = []

        self._model = OpenAI(api_key = OPENAI_API_KEY)
    
    def _sample_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        terminators: Collection[str] = (),
        timeout: float = 5,
    ) -> str:
        max_tokens = min(max_tokens, 4096)

        self._messages.append({ "role": "user", "content": prompt })

        response = self._model.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            messages = self._messages,
            temperature = self._temperature,
            max_tokens=max_tokens,
            timeout = timeout,
            stop = terminators,
        )

        return response.choices[0].message.content
    
    def _reset_messages(self) -> None:
        self._messages = self._base_messages.copy()

    @staticmethod
    def _extract_path(path: str) -> list[tuple[int, int]]:
        """Extracts a path from a string as a list of tuple coordinates."""
        tuples = re.findall(TUPLE_REGEX, path)                     # Extract tuples from the path.
        coordinates = [re.findall(DIGIT_REGEX, t) for t in tuples] # Extract coordinates as a regex match.
        coordinates = [c[0] for c in coordinates if len(c) == 1]   # Safely extract from regex match.
        return [(int(x), int(y)) for x, y in coordinates]          # Convert to tuple of integers.

    def _validate_path(self, path: list[tuple[int, int]]) -> bool:
        """Validates that a path is valid in the given map."""

        # Check if all coordinates are valid. (in bounds and traversable)
        valid = all(self.game.in_bounds((x, y)) and self.game[x, y] != HarvestTile.WALL for x, y in path)
        # Check if all moves are valid. (adjacent)
        valid = valid and all(abs(x1 - x2) + abs(y1 - y2) <= 1 for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]))

        return valid

    def _execute(self, path: list[tuple[int, int]]):
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

            if self.game[x, y] == HarvestTile.APPLE:
                self.game[x, y] = HarvestTile.ORCHARD
                self.score += 1
                history_item["ate"] = True
                history_item["score_after"] = self.score

            self.position = (x, y)
            self.game.history.append(history_item)

    def _create_prompt(self) -> str:
        """
        Generates a string representation of the player's state in the game. This is primarily used
        for prompting the LLM.
        """

        # Report the player's own location and score.
        tile = self.game[self.position]
        prompt = f"You are at {self.position}, with {self.score} points, " + \
            f"standing on \"{tile}\", which is a {HarvestTile(tile).__str__()}. " + \
            "Your location is marked with an \"#\" symbol on the map.\n"

        # Report other players' locations and scores.
        prompt += '\n'.join([
            f"{player.name}: {player.position}, with {player.score} points."
            for player in self.game.players.values()
            if player.name != self.name
        ]) + '\n'

        # Report apples.
        # TODO: Don't report apples out of sight.
        prompt += f"Apple locations: {[(x, y) for (x, y), cell in self.game if cell == HarvestTile.APPLE]}\n"

        # Report the map.
        map_snapshot = self.game.view_map(self.position) \
            if self.sight == -1 else \
            self.game.view_partial_map(self.position, self.sight, self.position)
        prompt += f"Here is the map: \n{map_snapshot}\n"

        prompt += f"Where will you, player {self.name}, move next?"

        return prompt

    def run(self):
        """Executes one turn for this player."""

        while True:
            action = self._sample_text(self._create_prompt())
            path = self._extract_path(action)

            if len(path) != 1:
                self._messages.append({ "role": "user", "content": "Your answer must be one coordinate. Please try again." })
                continue

            # Quick hack to allow for validation.
            path = [self.position] + path

            if not self._validate_path(path):
                self._messages.append({ "role": "user", "content": "The coordinate is invalid. Please try again." })
                continue

            break

        self._execute(path[1:])
        self._reset_messages()

    def describe(self):
        """Describe the location and score of this player."""
        return { "name": self.name, "position": self.position, "score": self.score }
