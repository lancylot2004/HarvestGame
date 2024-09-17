import functools
from datetime import datetime, timedelta
import re
from typing import override

from concordia.agents.deprecated_agent import BasicAgent
from concordia.components.agent.question_of_query_associated_memories import Identity
from concordia.components.agent.to_be_deprecated.identity import SimIdentity
from concordia.components.agent.to_be_deprecated.plan import SimPlan
from concordia.components.constant import ConstantComponent
from concordia.language_model.gpt_model import GptLanguageModel
from concordia.associative_memory.formative_memories import FormativeMemoryFactory, AgentConfig
from concordia.associative_memory.importance_function import ConstantImportanceModel
from concordia.clocks.game_clock import FixedIntervalClock
from concordia.associative_memory.blank_memories import MemoryFactory
from concordia.typing.entity import ActionSpec, OutputType
from concordia.typing.component import Component
from concordia.components import agent as agent_components
from concordia.agents.simple_llm_agent import SimpleLLMAgent
from sentence_transformers import SentenceTransformer

from harvest.helpers import HARVEST_RULES, OPENAI_API_KEY, HarvestTile


TUPLE_REGEX = r"(\(\s*?\d+\s*?,\s*?\d+\s*?\))" # Extracts whole tuples.
DIGIT_REGEX = r"\((\d+),\s*?(\d+)\)"           # Extracts individual numbers from tuples.


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
        assert len(name) == 1, "Player name must be a single character."
        self._name = name
        self.game = game
        self.position = position
        self.score = score
        self.sight = sight

        embedder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embedder = lambda x: embedder_model.encode(x, show_progress_bar = False)
 
        model = GptLanguageModel(api_key = OPENAI_API_KEY, model_name = "gpt-3.5-turbo-0125")

        clock = FixedIntervalClock(
            start = datetime.now(),
            step_size = timedelta(minutes = 1)
        )

        fixed_importance = ConstantImportanceModel()
        blank_factory = MemoryFactory(
            model = model,
            embedder = embedder,
            importance = fixed_importance.importance,
            clock_now = clock.now,
        )

        formative_factory = FormativeMemoryFactory(
            model = model,
            shared_memories = HARVEST_RULES,
            blank_memory_factory_call = blank_factory.make_blank_memory
        )

        memory = formative_factory.make_memories(agent_config = AgentConfig(
            name = name,
            goal = goal,
            context = ''.join(HARVEST_RULES),
        ))

        idComp = SimIdentity(model, memory, name)
        planComp = SimPlan(
            model = model, memory = memory, agent_name = name,
            clock_now = clock.now, components = [idComp], verbose = False,
            goal = ConstantComponent(state = goal),
        )

        self.call_to_action = ActionSpec(
            output_type = OutputType.FREE,
            call_to_action = """
                What is your strategy? Give the next move you will make. You may only move one tile. You must not go 
                outside the bounds of the map, marked by "@". Make use of what you\'ve been told about your location, 
                the map, and other players. Your goal is to maximise your own points by picking up apples, but 
                remember, apples spawn more frequently next to other apples, and do not spawn at all if there are no 
                apples nearby. Your response should be the coordinate of the tile you will move to, in the format "(x, y)".
            """.strip()
        )

        self.agent = BasicAgent(
            agent_name = name, model = model, clock = clock, verbose = False,
            components = [idComp, planComp, self]
        )

    @override
    @functools.cache
    def name(self) -> str: return "Harvest Agent Component"
    
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
                if player.name != self.name
            ]) + '\n'
        
        # Report the player's own location and score.
        tile = self.game[self.position]
        state += f"You are at {self.position}, with {self.score} points, " + \
            f"standing on \"{tile}\", which is a {HarvestTile(tile).__str__()}. " + \
            "Your location is marked with an \"#\" symbol on the map.\n"

        # Report apples.
        # TODO: Don't report apples out of sight.
        state += f"Apple locations: {[(x, y) for (x, y), cell in self.game if cell == HarvestTile.APPLE]}\n"

        # Report the map.
        map_snapshot = self.game.view_map(self.position) \
            if self.sight == -1 else (
            self.game.partial_view(self.position, self.sight, self.position))
        state += f"Here is the map: \n{map_snapshot}"

        return state

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
                "type": "move", "player": self._name, "from": self.position, "to": (x, y),
                "ate": False, "score_after": self.score 
            }

            if self.game[x, y] == HarvestTile.APPLE:
                self.game[x, y] = HarvestTile.ORCHARD
                self.score += 1
                history_item["ate"] = True
                history_item["score_after"] = self.score

            self.position = (x, y)
            self.game.history.append(history_item)

    def run(self):
        """Executes one turn for this player."""

        while True:
            action = self.agent.act(self.call_to_action)
            path = self._extract_path(action)

            if len(path) != 1:
                self.agent.observe("The action must be one coordinate. Please try again.")
                continue

            # Quick hack to allow for validation.
            path = [self.position] + path

            if not self._validate_path(path):
                self.agent.observe("The coordinate is invalid. Please try again.")
                continue

            break

        self._execute(path[1:])

    def describe(self):
        """Describe the location and score of this player."""
        return { "name": self._name, "position": self.position, "score": self.score }
