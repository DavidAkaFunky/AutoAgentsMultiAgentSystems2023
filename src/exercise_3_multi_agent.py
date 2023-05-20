import argparse
import numpy as np
from gym import Env
from typing import Sequence
import time
import copy

from environment import Agent
from environment.utils import compare_results
from environment.share_or_take import ShareOrTake

from basic_agent import BasicAgent

def run_multi_agent(environment: Env, agents: list[Agent], n_episodes: int, render=False) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        print("Starting episode {} in {}".format(episode + 1, n_episodes))
        steps = 0
        finished = False
        observations = environment.reset(agents)
        for agent in agents:
            agent.reset_parameters(agent.id)

        while True:
            if render:
                environment.render()
                time.sleep(0.5)
            
            steps += 1
            
            # TODO: Add statistics like fertility rate, average energy, mortality rate, etc.
            observations, finished = environment.step(observations)

            if finished:
                break

        results[episode] = steps

    if render:
        environment.render()
        environment.close()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    teams = {
        "Greedy Team": [
            BasicAgent(greedy=True),
            BasicAgent(greedy=True),
            BasicAgent(greedy=False),
            BasicAgent(greedy=False)
        ],
    }

    results = {}
    for team, agents in teams.items():
        environment = ShareOrTake(grid_shape=(15, 15), n_food=5, max_steps=10)
        # result = run_multi_agent(environment, agents, opt.episodes, render=True)
        result = run_multi_agent(environment, agents, 5, render=True)
        results[team] = result

    compare_results(
        results,
        title="Teams Comparison on 'Share or Take' Environment",
        colors=["orange",]
    )

