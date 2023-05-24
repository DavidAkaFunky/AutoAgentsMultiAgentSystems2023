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

    population = np.zeros((n_episodes, environment.max_steps + 1))
    avg_population = dict()
    deaths = np.zeros((n_episodes, environment.max_steps))
    births = np.zeros((n_episodes, environment.max_steps))

    for episode in range(n_episodes):

        print("Starting episode {} in {}".format(episode + 1, n_episodes))
        steps = 0
        finished = False
        observations = environment.reset(agents)
        for agent in agents:
            agent.reset_parameters(agent.id)

        while True:
            population[episode, steps] = len(environment.agents)
            if render:
                environment.render()
                time.sleep(0.5)

            steps += 1
            
            # TODO: Add death and birth rates
            observations, deaths_ep, births_ep, finished = environment.step(observations)
            deaths[episode, steps - 1] = deaths_ep
            births[episode, steps - 1] = births_ep

            if finished:
                break

        population[episode, steps] = len(environment.agents)
        print("Population: ", population[episode])
        print("Deaths: ", deaths[episode])
        print("Births: ", births[episode])

    if render:
        environment.render()
        environment.close()
            

    return population, deaths, births

def parse_config(input_file) -> dict[str, list[Agent]]:
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
                tribe_name = " ".join(line[1:])
            case "a":
                greedy = (line[1] == "y")
                energy = int(line[2])
                reproduction_threshold = int(line[3])
                quantity = int(line[-1])
                for _ in range(quantity):
                    situations[situation_name].append(BasicAgent(tribe_name, greedy, energy, reproduction_threshold))
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
    print(situations)
    
    population = {}
    deaths = {}
    births = {}
    for situation, agents in situations.items():
        environment = ShareOrTake(grid_shape=grid_shape, n_food=n_food, max_steps=n_steps, debug=False)
        population_sit, deaths_sit, births_sit = run_multi_agent(environment, agents, episodes, render=render)
        population[situation] = population_sit
        deaths[situation] = deaths_sit
        births[situation] = births_sit

    
    for situation in situations.keys():
        dict_pop = {}
        dict_deaths = {}
        dict_births = {}
        for i in range(n_steps):
            for population_sit in population[situation]:
                dict_pop[i] = (np.mean(population_sit[i]))
            for deaths_sit in deaths[situation]:
                dict_deaths[i] = (np.mean(deaths_sit[i]))
            for births_sit in births[situation]:
                dict_births[i] = (np.mean(births_sit[i]))

        compare_results(
            dict_pop,
            title="Population Comparison on 'Share or Take' Environment",
            metric="Population per step",
            plot=True,
            colors=["orange",]
        )

        compare_results(
            dict_deaths,
            title="Deaths Comparison on 'Share or Take' Environment",
            metric="Deaths per step",
            colors=["orange",]
        )

        compare_results(
            dict_births,
            title="Births Comparison on 'Share or Take' Environment",
            metric="Births per step",
            colors=["orange",]
        )