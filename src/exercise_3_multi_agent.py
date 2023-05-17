import argparse
import numpy as np
from gym import Env
from typing import Sequence
import time

from environment import Agent
from environment.utils import compare_results
from environment.share_or_take import ShareOrTake

from basic_agent import BasicAgent

def run_multi_agent(environment: Env, agents: list[Agent], n_episodes: int, render=False) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        finished = False
        observations = environment.reset()

        while True:
            if render:
                environment.render()
                time.sleep(0.5)
            
            steps += 1
            actions = []
            for agent_id, agent in enumerate(agents):
                agent.see(observations[agent_id])
                actions.append(agent.action())
                
            observations, rewards, finished = environment.step(actions)

            for agent_id, agent in enumerate(agents):
                agent.feedback(rewards[agent_id])
                agent.has_eaten = False # Reset

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
        environment = ShareOrTake(agents, grid_shape=(7, 7), n_food=2, max_steps=15)
        result = run_multi_agent(environment, agents, opt.episodes, render=True)
        results[team] = result

    compare_results(
        results,
        title="Teams Comparison on 'Share or Take' Environment",
        colors=["orange",]
    )

