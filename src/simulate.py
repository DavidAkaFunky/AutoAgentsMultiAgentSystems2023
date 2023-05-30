import argparse
import numpy as np
from gym import Env
from typing import Sequence
import time
import copy

from environment.utils import compare_results_pop, compare_results_other_metrics
from environment.share_or_take import ShareOrTake

from agents.random_agent import RandomAgent
from agents.regular_agent import RegularAgent
from agents.tribal_agent import TribalAgent
from agents.rational_agent import RationalAgent

COLOURS = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

def run_multi_agent(environment: Env, starting_agents: list[RandomAgent], n_episodes: int, render=False) -> np.ndarray:

    population = np.zeros((n_episodes, environment.max_steps + 1))
    deaths = np.zeros((n_episodes, environment.max_steps))
    births = np.zeros((n_episodes, environment.max_steps))
    population_greedy = np.zeros((n_episodes, environment.max_steps + 1))
    population_non_greedy = np.zeros((n_episodes, environment.max_steps + 1))

    for episode in range(n_episodes):
        agents = copy.deepcopy(starting_agents)

        print("Starting episode {} in {}".format(episode + 1, n_episodes))
        steps = 0
        finished = False
        observations = environment.reset(agents)
        for agent in agents:
            agent.reset_parameters(agent.id)

        while True:
            population[episode, steps] = len(environment.agents)
            greedy_agents = sum(environment.agents[agent].is_greedy for agent in environment.agents)
            population_greedy[episode, steps] = greedy_agents
            population_non_greedy[episode, steps] = len(environment.agents) - greedy_agents

            if render:
                environment.render()
                time.sleep(2.5)

            steps += 1
            
            # TODO: Add death and birth rates
            observations, deaths_ep, births_ep, finished = environment.step(observations, steps)
            deaths[episode, steps - 1] = deaths_ep
            births[episode, steps - 1] = births_ep

            if finished:
                break

        population[episode, steps] = len(environment.agents)
        greedy_agents = sum(environment.agents[agent].is_greedy for agent in environment.agents)
        population_greedy[episode, steps] = greedy_agents
        population_non_greedy[episode, steps] = len(environment.agents) - greedy_agents


    if render:
        environment.render()
        environment.close()
            

    return population, deaths, births, population_greedy, population_non_greedy

def parse_config(input_file) -> dict[str, list[RandomAgent]]:
    situations = {}
    for line in input_file.readlines():
        line = line.strip().split()
        match line[0]:
            case "g":
                grid_shape = tuple(map(int, line[1:]))
            case "f":
                n_food = int(line[1])
            case "s":
                n_steps = int(line[1])
            case "sit":
                situation_name = " ".join(line[1:])
                situations[situation_name] = []
            case "t":
                if len(line) == 1:
                    tribe_name = None
                else:
                    tribe_name = " ".join(line[1:])
            case "a":
                greedy = (line[2] == "y")
                
                energy = int(line[3])
                reproduction_threshold = int(line[4])
                quantity = int(line[-1])
                
                for _ in range(quantity):
                    if tribe_name is not None:
                        agent = TribalAgent(tribe_name, energy, reproduction_threshold)
                    else:
                        match line[1]:
                            case "random":
                                agent = RandomAgent(greedy, energy, reproduction_threshold)
                            case "regular":
                                agent = RegularAgent(greedy, energy, reproduction_threshold)
                            case "rational":
                                agent = RationalAgent(greedy, energy, reproduction_threshold)
                    situations[situation_name].append(agent)
    return situations, grid_shape, n_food, n_steps

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=argparse.FileType('r'))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    input_file = opt.input_file
    render = opt.render
    episodes = opt.episodes
    situations, grid_shape, n_food, n_steps = parse_config(input_file)
    
    population = {}
    deaths = {}
    births = {}
    greedy_population = {}
    non_greedy_population = {}

    for situation, agents in situations.items():
        environment = ShareOrTake(grid_shape=grid_shape, n_food=n_food, max_steps=n_steps, debug=False)
        population_sit, deaths_sit, births_sit, greedy_sit, n_greedy_sit = run_multi_agent(environment, agents, episodes, render=render)
        
        population[situation] = np.transpose(population_sit)
        greedy_population[situation] = np.transpose(greedy_sit)
        non_greedy_population[situation] = np.transpose(n_greedy_sit)

        deaths[situation] = np.transpose(deaths_sit)
        births[situation] = np.transpose(births_sit)

    compare_results_pop(
        population,
        title="Population Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )
    compare_results_pop(
        greedy_population,
        title="Greedy Population Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )

    compare_results_pop(
        non_greedy_population,
        title="Non Greedy Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )
    
    '''

    compare_results_other_metrics(
        deaths,
        title="Deaths Comparison on 'Share or Take' Environment",
        metric="Deaths per step",
        colors=COLOURS[:len(situations)]
    )

    compare_results_other_metrics(
        births,
        title="Births Comparison on 'Share or Take' Environment",
        metric="Births per step",
        colors=COLOURS[:len(situations)]
    )
    '''