from textwrap import dedent
from typing import Callable, Optional

import numpy as np
from openai import OpenAI
from harvest.game import HarvestGame
from harvest.helpers import OPENAI_API_KEY, PROBABILITY_FUNCS, HarvestMap
import json


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
    feedback_file: Optional[str] = None,
    temperature: float = 0.5,
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
        feedback_file (Optional[str]): The feedback file to use. If specified, the feedback content
            will be given to all agents as part of their system instructions.
    """

    game = HarvestGame(
        map = map,
        spawn_points = spawn_points,
        seed = seed,
    )

    feedback = None
    if feedback_file is not None:
        with open(feedback_file, "r") as file:
            data = json.load(file)
            analysis = data.get("analysis")
            if isinstance(analysis, list) and len(analysis) == player_num:
                feedback = analysis
            else:
                raise ValueError(f"Invalid feedback file: analysis must be a list of length {player_num}")

    for i in range(player_num):
        player_feedback = feedback[i] if feedback else None
        game.add_player(str(i), goal = player_goal, feedback = player_feedback, temperature = temperature)

    game.experiment(
        output_file = output_file, 
        probability_func = probability_func, 
        desired_apple_num = desired_apple_num, 
        limit = round_limit
    )

def extract_history(log_file: str) -> dict:
    """Extracts the history from a log file.

    Args:
        log_file (str): The file to read. This should be in the form of a list of JSON objects.

    Returns:
        dict: The list of quadruples (AGENT, OBSERVATION, ACTION, REWARD) extracted (property 
            "history"), the number of agents (property "num_agents"), and the final total score
            (property "total_score").
    """
    trajectory = []
    num_agents = 0
    
    with open(log_file, 'r') as file:
        data = json.load(file)
        
        # Determine `num_agents`. We assume that it would be correct to just read the first "round"
        # object from the file.
        for obj in data:
            if obj["type"] == "round":
                num_agents = len(obj["players"])
                break
        
        assert num_agents > 0, "Could not determine the number of agents"

        # Extract all agent actions into the desired format.
        round_map = None
        for obj in data:
            if obj["type"] == "round":
                round_map = obj["map"]
            elif obj["type"] == "move":
                trajectory.append({
                    "AGENT": obj["player"],
                    "OBSERVATION": [(j, i) for i, row in enumerate(round_map) for j, tile in enumerate(row) if tile == "A"],
                    "ACTION": f"{obj['from']} -> {obj['to']}",
                    "REWARD": 1 if obj["ate"] else 0
                })

        # Calculate `total_score` from the last "round" object
        for obj in reversed(data):
            if obj["type"] == "round":
                assert len(obj["players"]) == num_agents, "Mismatch in number of agents"
                total_score = sum(player["score"] for player in obj["players"])
                break
        
        assert total_score is not None, "Could not determine the total score."

        return {
            "trajectory": trajectory,
            "num_agents": num_agents,
            "total_score": total_score
        }

def execute_analysis(
    log_files_set_1: list[str],
    log_files_set_2: list[str],
    prompt: str,
    output_file: str,
) -> None:
    """
    Executes an analysis experiment.

    Args:
        log_files_set_1 (list[str]): The list of log files for the high reward history.
        log_files_set_2 (list[str]): The list of log files for the low reward history.
        prompt (str): The prompt to be used for the analysis. Variables {num_agents}, 
            {trajectories_set_1}, and {trajectories_set_2} will be provided. The prompt instruct 
            the LLM to use "-INSTRUCTION-" to separate instructions.
        output_file (str): The file to output the analysis to.
    """

    assert len(log_files_set_1) > 0, "Must have at least one HIGH REWARD history!"
    assert len(log_files_set_2) > 0, "Must have at least one LOW REWARD history!"

    histories_set_1 = list(map(extract_history, log_files_set_1))
    histories_set_2 = list(map(extract_history, log_files_set_2))

    num_agents = set([history["num_agents"] for history in [*histories_set_1, *histories_set_2]])
    assert len(num_agents) == 1, "Histories have differing agent numbers!"
    num_agents = list(num_agents)[0]

    _backslash = "\n"
    _prompt = dedent(prompt).format(
        num_agents = num_agents,
        trajectories_set_1 = {_backslash.join(f"Number {index}, TOTAL REWARD {trajectory["total_score"]}, {trajectory}" for index, trajectory in enumerate(histories_set_1))},
        trajectories_set_2 = {_backslash.join(f"Number {index}, TOTAL REWARD {trajectory["total_score"]}, {trajectory}" for index, trajectory in enumerate(histories_set_2))}
    )

    _model = OpenAI(api_key = OPENAI_API_KEY)

    _messages = [
        {
            "role": "system",
            "content": dedent(f"""
                You are in a 2D grid with walls ("@"), orchards ("O"), apples ("A"), and land 
                ("."). Collecting apples earns points, and apples only regrow on orchard tiles if 
                nearby apples remain. The more apples near an empty orchard, the faster new apples 
                grow. If too few apples are left, the chance of new ones appearing decreases 
                significantly. Players can move freely without affecting the respawn rate or 
                blocking each other.
            """.strip())
        },
        { "role": "system", "content": "You are a helpful assistant. Answer the user's question." },
        { "role": "user", "content": _prompt }
    ]

    response = _model.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = _messages,
        temperature = 0.5,
        max_tokens = 1024,
        timeout = 10,
        stop = (),
    ).choices[0].message.content

    # Split the response into individual instructions
    instructions = response.split("-INSTRUCTION-")
    instructions = [instr.strip() for instr in instructions if instr.strip()]

    # Ensure we have the correct number of instructions
    if len(instructions) != num_agents:
        raise ValueError(f"Expected {num_agents} instructions, but got {len(instructions)}")
    
    # LLM should know about its previous response
    _messages.append({ "role": "assistant", "content": response })
    # _messages.append({
    #     "role": "user",
    #     "content": f"""
    #         For each of the instructions you provided, explain why you came up with that 
    #         instruction. Remember, that you were asked to provide strategies to achieve HIGH TOTAL
    #         REWARD, not LOW TOTAL REWARD, especially considering the differences in TOTAL REWARD 
    #         between the RESPAWNING trajectories and the NON-RESPAWNING trajectories, and analyse 
    #         the effect of how, when, and if apples respawn on TOTAL REWARD.
            
    #         Use the following format, with each instruction separated by '-EXPLANATION-':
    #         AGENT 1 should follow my instruction because [explanation 1]
    #         -EXPLANATION-
    #         ...
    #         -EXPLANATION-
    #         AGENT {num_agents} should follow my instruction because [explanation {num_agents}]
    #     """.strip()
    # })

    # response = _model.chat.completions.create(
    #     model = "gpt-4o-mini-2024-07-18",
    #     messages = _messages,
    #     temperature = 0.5,
    #     max_tokens = 1024,
    #     timeout = 10,
    #     stop = (),
    # ).choices[0].message.content

    # # Split the response into individual explanations
    # explanations = response.split("-EXPLANATION-")
    # explanations = [expl.strip() for expl in explanations if expl.strip()]

    # # Ensure we have the correct number of explanations
    # if len(explanations) != num_agents:
    #     raise ValueError(f"Expected {num_agents} explanations, but got {len(explanations)}")

    with open(output_file, "w") as file:
        json.dump(
            { 
                "log_files_set_1": log_files_set_1,
                "log_files_set_2": log_files_set_2,
                "prompt": prompt,
                "analysis": instructions,
                # "explanations": explanations,
                "messages": _messages # + [{ "role": "assistant", "content": explanations }]
            },
            file
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
        seed = ord('A'),
        player_num = 4,
        player_goal = "To maximise your own points.",
        output_file = "expts/4-4x8MO-Increased-12-A-F12-2.json", 
        probability_func = PROBABILITY_FUNCS["increased"], 
        desired_apple_num = 12, 
        round_limit = 50,
        feedback_file = "expts/Analysis-12.json"
    )

    # Do another one
    # Do same without feedback
    # Repeat all four with "increased" functions

    # execute_analysis(
    #     log_files_set_1 = ["expts/4-4x8MO-Original-12-A-F0.9-4.json", "expts/4-4x8MO-Original-12-A-F0.9-3.json"],
    #     log_files_set_2 = ["expts/4-4x8MO-Zero-12-A-1.json"],
    #     prompt = """
    #         You will receive a list of (AGENT, OBSERVATION, ACTION, REWARD) sets from {num_agents} 
    #         agents playing the Harvest game. Each OBSERVATION shows the coordinates of apples 
    #         visible to the agent, and each ACTION represents the movement the agent made, 
    #         written as '(x, y) -> (u, v)'. The data is split into two types: RESPAWNING, where 
    #         apples regenerate based on proximity to other apples, as described above in the rules; 
    #         and NON-RESPAWNING, where apples never grow back.

    #         In RESPAWNING trajectories: {trajectories_set_1}

    #         In NON-RESPAWNING trajectories: {trajectories_set_2}

    #         TOTAL REWARD is the combined reward of all agents over the entire game. Your task is to 
    #         create {num_agents} language instructions summarising the best strategies for each 
    #         agent to achieve a HIGH TOTAL REWARD. Compare how TOTAL REWARD differs between 
    #         RESPAWNING and NON-RESPAWNING scenarios. Assume the agents are in a RESPAWNING setting 
    #         where apples grow back as described. The optimal strategy is to balance the rate at 
    #         which agents consume apples with the rate at which apples grow back. Generally, agents
    #         should consume apples at a slower rate, to allow for more apples to grow back.

    #         Use this format, with each instruction separated by '-INSTRUCTION-': 
    #         AGENT 1 should [instruction 1] 
    #         -INSTRUCTION- 
    #         AGENT 2 should [instruction 2] 
    #         ... 
    #         -INSTRUCTION- 
    #         AGENT {num_agents} should [instruction {num_agents}]
    #     """.strip(),
    #     output_file = "expts/Analysis-13.json",
    # )
