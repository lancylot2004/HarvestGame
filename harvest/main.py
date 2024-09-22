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

def extract_history(log_file: str) -> list[tuple[str, str, str, int]]:
    history = []
    with open(log_file, 'r') as file:
        data = json.load(file)

        for obj in data:
            match obj["type"]:
                case "round": round_map = obj["map"]
                case "move":
                    history.append({
                        "AGENT": obj["player"], 
                        "OBSERVATION": [(j, i) for i, row in enumerate(round_map) for j, tile in enumerate(row) if tile == "A"], 
                        "ACTION": f"{obj['from']} -> {obj['to']}", 
                        "REWARD": 1 if obj["ate"] else 0
                    })
                case _: continue

        return history

def execute_analysis(
    high_reward_log_files: list[str],
    low_reward_log_files: list[str],
    output_file: str,
    num_agents: int,
) -> None:
    assert len(high_reward_log_files) > 0, "Must have at least one HIGH REWARD history!"
    assert len(low_reward_log_files) > 0, "Must have at least one LOW REWARD history!"

    high_reward_histories = list(map(extract_history, high_reward_log_files))
    low_reward_histories = list(map(extract_history, low_reward_log_files))

    _backslash = "\n"
    _prompt = f"""
        You will be given a list of (AGENT, OBSERVATION, ACTION, REWARD) quadruples collected from 
        {num_agents} agents playing the Harvest game. Each OBSERVATION is a list of coordinates of apples that 
        the AGENT can see, and each ACTION describes the ACTION of that AGENT performed given 
        OBSERVATION. ACTIONs are in the form '(x, y) -> (u, v)', indicating movement from '(x,y)' 
        to '(u,v)'. The trajectories are separated into HIGH REWARD and LOW REWARD examples.

        HIGH REWARD trajectories
        {_backslash.join(f"Number {index}, {trajectory}" for index, trajectory in enumerate(high_reward_histories))}

        LOW REWARD trajectories
        {_backslash.join(f"Number {index}, {trajectory}" for index, trajectory in enumerate(low_reward_histories))}

        Output exactly {num_agents} language instructions that best summarise the strategies that each AGENT should 
        follow to receive HIGH REWARD, not LOW REWARD, based on the above trajectories. You should 
        output an instruction for each of the agents, each starting with the prefix 'AGENT should'.

        Use the following format, with each instruction separated by '---INSTRUCTION---':
        AGENT should [instruction 1]
        ---INSTRUCTION---
        AGENT should [instruction 2]
        ---INSTRUCTION---
        ...
        ---INSTRUCTION---
        AGENT should [instruction {num_agents}]
    """.strip()

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
            """
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

    with open(output_file, "w") as file:
        json.dump(
            { 
                "high_reward_log_files": high_reward_log_files,
                "low_reward_log_files": low_reward_log_files,
                "analysis": instructions
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
        output_file = "expts/4-4x8MO-Original-12-A-F0.7-2.json", 
        probability_func = PROBABILITY_FUNCS["original"], 
        desired_apple_num = 12, 
        round_limit = 50,
        feedback_file = "expts/Analysis-1.json",
        temperature = 0.7
    )

    # execute_analysis(
    #     ["expts/4-4x8MO-Original-12-A-2.json"],
    #     ["expts/4-4x8MO-Original-12-A-1.json"],
    #     "expts/Analysis-1.json",
    #     num_agents=4
    # )
