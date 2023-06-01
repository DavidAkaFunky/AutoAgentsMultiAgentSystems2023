import argparse
import numpy as np
from gym import Env
import time
import copy

from utils import compare_results_pop, compare_results_other_metrics
from environment.share_or_take import ShareOrTake

from agents.random_agent import RandomAgent
from agents.regular_agent import RegularAgent
from agents.tribal_agent import TribalAgent
from agents.rational_agent import RationalAgent
from agents.evolutive_agent import EvolutiveAgent

COLOURS = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

def run_multi_agent(environment: Env, starting_agents: list[RandomAgent], n_episodes: int, render=False) -> np.ndarray:

    population = np.zeros((n_episodes, environment.max_steps + 1))
    avg_energy = np.zeros((n_episodes, environment.max_steps + 1))
    avg_greedy_energy = np.zeros((n_episodes, environment.max_steps + 1))
    avg_peaceful_energy = np.zeros((n_episodes, environment.max_steps + 1))
    population_greedy = np.zeros((n_episodes, environment.max_steps + 1))
    population_peaceful = np.zeros((n_episodes, environment.max_steps + 1))
    births = np.zeros((n_episodes, environment.max_steps))
    deaths = np.zeros((n_episodes, environment.max_steps))

    for episode in range(n_episodes):
        agents = copy.deepcopy(starting_agents)

        print("Starting episode {} in {}".format(episode + 1, n_episodes))
        steps = 0
        finished = False
        observations, total_agents_ep, greedy_agents_ep, peaceful_agents_ep, avg_energy_ep, greedy_avg_energy_ep, peaceful_avg_energy_ep = environment.reset(agents)

        for agent in agents:
            agent.reset_parameters(agent.id)

        while True:
            population[episode, steps] = total_agents_ep
            population_greedy[episode, steps] = greedy_agents_ep
            population_peaceful[episode, steps] = peaceful_agents_ep
            avg_energy[episode, steps] = avg_energy_ep
            avg_greedy_energy[episode, steps] = greedy_avg_energy_ep
            avg_peaceful_energy[episode, steps] = peaceful_avg_energy_ep

            if render:
                environment.render()
                time.sleep(2.5)

            steps += 1
            
            # TODO: Add death and birth rates
            observations, total_agents_ep, greedy_agents_ep, peaceful_agents_ep, avg_energy_ep, greedy_avg_energy_ep, peaceful_avg_energy_ep, deaths_ep, births_ep, finished = environment.step(observations, steps)
            deaths[episode, steps - 1] = deaths_ep
            births[episode, steps - 1] = births_ep

            if finished:
                break

        population[episode, steps] = total_agents_ep
        population_greedy[episode, steps] = greedy_agents_ep
        population_peaceful[episode, steps] = peaceful_agents_ep
        avg_energy[episode, steps] = avg_energy_ep
        avg_greedy_energy[episode, steps] = greedy_avg_energy_ep
        avg_peaceful_energy[episode, steps] = peaceful_avg_energy_ep

    if render:
        environment.render()
        environment.close()
            
    # TODO Return energy-related metrics
    return population, population_greedy, population_peaceful, avg_energy, avg_greedy_energy, avg_peaceful_energy, deaths, births

def parse_config(input_file) -> dict[str, list[RandomAgent]]:
    situations = {}
    policies = {}
    tribe_name = None
    for line in input_file.readlines():
        line = line.strip().split()
        match line[0]:
            case "g":
                grid_shape = tuple(map(int, line[1:]))
            case "p":
                policies[situation_name] = " ".join(line[1:])
            case "f":
                n_food = int(line[1])
            case "s":
                n_steps = int(line[1])
            case "sit":
                situation_name = " ".join(line[1:])
                situations[situation_name] = []
                policies[situation_name] = "RANDOM"
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
                            case "evolutive":
                                agent = EvolutiveAgent(greedy, energy, reproduction_threshold)
                    situations[situation_name].append(agent)
    return situations, grid_shape, n_food, n_steps, policies

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=argparse.FileType('r'))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    opt = parser.parse_args()

    input_file = opt.input_file
    render = opt.render
    episodes = opt.episodes
    situations, grid_shape, n_food, n_steps, policies = parse_config(input_file)
    
    population = {}
    greedy_population = {}
    peaceful_population = {}
    avg_energy = {}
    avg_greedy_energy = {}
    avg_peaceful_energy = {}
    deaths = {}
    births = {}

    for situation, agents in situations.items():
        environment = ShareOrTake(grid_shape=grid_shape, n_food=n_food, max_steps=n_steps, debug=False, policy=policies[situation])
        population_sit, greedy_sit, n_greedy_sit, avg_energy_sit, avg_greedy_energy_sit, avg_peaceful_energy_sit, deaths_sit, births_sit = run_multi_agent(environment, agents, episodes, render=render)
        
        population[situation] = np.transpose(population_sit)
        greedy_population[situation] = np.transpose(greedy_sit)
        peaceful_population[situation] = np.transpose(n_greedy_sit)
        avg_energy[situation] = np.transpose(avg_energy_sit)
        avg_greedy_energy[situation] = np.transpose(avg_greedy_energy_sit)
        avg_peaceful_energy[situation] = np.transpose(avg_peaceful_energy_sit)
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
        peaceful_population,
        title="Peaceful Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )

    compare_results_pop(
        avg_energy,
        title="Avg. Energy Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )

    compare_results_pop(
        avg_greedy_energy,
        title="Avg. Energy (Greedy) Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )

    compare_results_pop(
        avg_peaceful_energy,
        title="Avg. Energy (Peaceful) Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)]
    )

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