import argparse
import numpy as np
from gym import Env
from typing import Sequence
import time

from environment import Agent
from environment.utils import compare_results
from environment.share_or_take import ShareOrTake

from basic_agent import BasicAgent

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int, render=False) -> np.ndarray:

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

    # 1 - Setup the environment
    environment = ShareOrTake(grid_shape=(7, 7), initial_n_agents=6, n_food=2, max_steps=4)

    # 2 - Setup the teams
    teams = {
        "Greedy Team": [
            BasicAgent(agent_id=0, greedy=True),
            BasicAgent(agent_id=1, greedy=True),
            BasicAgent(agent_id=2, greedy=False),
            BasicAgent(agent_id=3, greedy=False)
        ],
    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        result = run_multi_agent(environment, agents, opt.episodes, render=True)
        results[team] = result

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Share or Take' Environment",
        colors=["orange",]
    )

