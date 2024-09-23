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
            {trajectories_set_1}, and {trajectories_set_2} will be provided.
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
            "content": f"""
                You are part of a study on Sequential Social Dilemmas. The rules of the game are as 
                follows: In a 2D grid, "@" are walls, "O" are orchard tiles, "A" are apples, and 
                "." is land. Stepping on an apple will collect it and earn you points. Apples spawn 
                only on orchard tiles and when next to other apples. If there are no apples 
                remaining, no more will spawn. 
            """.strip()
        },
        { "role": "system", "content": "Answer the user's question." },
        { "role": "user", "content": _prompt }
    ]

    response = _model.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = _messages,
        temperature = 0.5,
        max_tokens = 1024,
        timeout = 5,
        stop = (),
    ).choices[0].message.content

    # Split the response into individual instructions
    instructions = response.split("---INSTRUCTION---")
    instructions = [instr.strip() for instr in instructions if instr.strip()]

    # Ensure we have the correct number of instructions
    if len(instructions) != num_agents:
        raise ValueError(f"Expected {num_agents} instructions, but got {len(instructions)}")
    
    # LLM should know about its previous response
    _messages.append({ "role": "assistant", "content": response })
    _messages.append({
        "role": "user",
        "content": f"""
            For each of the instructions you provided, explain why you came up with that instruction.
            
            Use the following format, with each instruction separated by '---EXPLANATION---':
            AGENT 1 should follow my instruction because [explanation 1]
            ---EXPLANATION---
            ...
            ---EXPLANATION---
            AGENT {num_agents} should follow my instruction because [explanation {num_agents}]
        """.strip()
    })

    response = _model.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = _messages,
        temperature = 0.5,
        max_tokens = 1024,
        timeout = 10,
        stop = (),
    ).choices[0].message.content

    # Split the response into individual explanations
    explanations = response.split("---EXPLANATION---")
    explanations = [expl.strip() for expl in explanations if expl.strip()]

    # Ensure we have the correct number of explanations
    if len(explanations) != num_agents:
        raise ValueError(f"Expected {num_agents} explanations, but got {len(explanations)}")

    with open(output_file, "w") as file:
        json.dump(
            { 
                "log_files_set_1": log_files_set_1,
                "log_files_set_2": log_files_set_2,
                "prompt": prompt,
                "analysis": instructions,
                "explanations": explanations,
                "messages": _messages + [{ "role": "assistant", "content": explanations }]
            },
            file
        )


if __name__ == "__main__":
    # execute_experiment(
    #     map = HarvestGame.parse_map([
    #         "@@@@@@@@@@",
    #         "@.OO.OOOO@",
    #         "@OOOOOOOO@",
    #         "@OOOOOOOO@",
    #         "@.OO.OOOO@",
    #         "@@@@@@@@@@",
    #     ]),
    #     spawn_points = [(1, 1), (1, 4), (4, 1), (4, 4)],
    #     seed = ord('A'),
    #     player_num = 4,
    #     player_goal = "To maximise your own points.",
    #     output_file = "expts/4-4x8MO-Original-12-A-F4.json", 
    #     probability_func = PROBABILITY_FUNCS["original"], 
    #     desired_apple_num = 12, 
    #     round_limit = 50,
    #     feedback_file = "expts/Analysis-4.json"
    # )

    execute_analysis(
        log_files_set_1 = ["expts/4-4x8MO-Original-12-A-2.json"],
        log_files_set_2 = ["expts/4-4x8MO-Original-12-A-F0.9-2.json"],
        prompt = """
            You will be given a list of (AGENT, OBSERVATION, ACTION, REWARD) quadruples collected 
            from {num_agents} agents playing the Harvest game. Each OBSERVATION is a list of 
            coordinates of apples that the AGENT can see, and each ACTION describes the ACTION of 
            that AGENT performed given OBSERVATION. ACTIONs are in the form '(x, y) -> (u, v)', 
            indicating movement from '(x,y)' to '(u,v)'. The trajectories are separated into 
            RESPAWNING and NON-RESPAWNING examples, where apples respawn and do not respawn 
            respectively.    
        
            RESPAWNING trajectories
            {trajectories_set_1}

            NON-RESPAWNING trajectories
            {trajectories_set_2}

            Define TOTAL REWARD as the summation of all the agents' reward. Output exactly 
            {num_agents} language instructions that best summarise the strategies that each AGENT 
            should follow to receive HIGH TOTAL REWARD in the RESPAWNING setting, not LOW TOTAL 
            REWARD. Agents should prioritise HIGH TOTAL REWARD and NOT HIGH INDIVIDUAL REWARD. You 
            should compare the TOTAL REWARD differences between the RESPAWN trajectories and 
            NON-RESPAWNING trajectories, and analyzing the effect of respawning rule to the TOTAL 
            REWARD. Your instructions should assume agents are in a RESPAWNING situation, where 
            apples respawn according to the rules of the game.

            Use the following format, with each instruction separated by '---INSTRUCTION---':
            AGENT 1 should [instruction 1]
            ---INSTRUCTION---
            ...
            ---INSTRUCTION---
            AGENT {num_agents} should [instruction {num_agents}]
        """.strip(),
        output_file = "expts/Analysis-6.json",
    )
