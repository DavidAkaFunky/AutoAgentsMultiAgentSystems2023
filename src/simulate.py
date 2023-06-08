import argparse
import numpy as np
from gym import Env
import time
import copy
import os
import multiprocessing

from utils import compare_results_pop, compare_results_other_metrics
from environment.share_or_take import ShareOrTake

from agents.random_agent import RandomAgent
from agents.regular_agent import RegularAgent
from agents.tribal_agent import TribalAgent
from agents.rational_agent import RationalAgent
from agents.evolutive_agent import EvolutiveAgent

COLOURS = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

def run_multi_agent(environment: Env, starting_agents: list[RandomAgent], n_episodes: int, return_dict: dict, situation: str, debug: bool, render: bool) -> np.ndarray:
    population = np.zeros((environment.max_steps + 1, n_episodes))
    avg_energy = np.zeros((environment.max_steps + 1, n_episodes))
    avg_greedy_energy = np.zeros((environment.max_steps + 1, n_episodes))
    avg_peaceful_energy = np.zeros((environment.max_steps + 1, n_episodes))
    population_greedy = np.zeros((environment.max_steps + 1, n_episodes))
    population_peaceful = np.zeros((environment.max_steps + 1, n_episodes))
    birth_rates = np.zeros((environment.max_steps, n_episodes))
    death_rates = np.zeros((environment.max_steps, n_episodes))

    pid = os.getpid()

    for episode in range(n_episodes):
        agents = copy.deepcopy(starting_agents)

        if debug:
            print(f"Starting episode {episode + 1} in {n_episodes} by pid {pid} for situation: {situation}.")
        step = 0
        finished = False
        observations, total_agents_ep, greedy_agents_ep, peaceful_agents_ep, avg_energy_ep, greedy_avg_energy_ep, peaceful_avg_energy_ep = environment.reset(agents)

        for agent in agents:
            agent.reset_parameters(agent.id)

        while True:
            population[step, episode] = total_agents_ep
            population_greedy[step, episode] = greedy_agents_ep
            population_peaceful[step, episode] = peaceful_agents_ep
            avg_energy[step, episode] = avg_energy_ep
            avg_greedy_energy[step, episode] = greedy_avg_energy_ep
            avg_peaceful_energy[step, episode] = peaceful_avg_energy_ep

            if render:
                environment.render()
                time.sleep(2.5)

            step += 1
            
            observations, total_agents_ep, greedy_agents_ep, peaceful_agents_ep, avg_energy_ep, greedy_avg_energy_ep, peaceful_avg_energy_ep, death_rate_ep, birth_rate_ep, finished = environment.step(observations, step)
            death_rates[step - 1, episode] = death_rate_ep
            birth_rates[step - 1, episode] = birth_rate_ep

            if finished:
                break

        population[step, episode] = total_agents_ep
        population_greedy[step, episode] = greedy_agents_ep
        population_peaceful[step, episode] = peaceful_agents_ep
        avg_energy[step, episode] = avg_energy_ep
        avg_greedy_energy[step, episode] = greedy_avg_energy_ep
        avg_peaceful_energy[step, episode] = peaceful_avg_energy_ep

    if render:
        environment.render()
        environment.close()

    return_dict["population_sit"] = population
    return_dict["greedy_sit"] = population_greedy
    return_dict["peaceful_sit"] = population_peaceful
    return_dict["avg_energy_sit"] = avg_energy
    return_dict["avg_greedy_energy_sit"] = avg_greedy_energy
    return_dict["avg_peaceful_energy_sit"] = avg_peaceful_energy
    return_dict["death_rates_sit"] = death_rates
    return_dict["birth_rates_sit"] = birth_rates

def parse_config(input_file) -> dict[str, list[RandomAgent]]:
    situations = {}
    policies = {}
    tribe_name = None
    filename = None
    for line in input_file.readlines():
        line = line.strip().split()
        match line[0]:
            case "filename":
                filename = " ".join(line[1:])
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
    return situations, grid_shape, n_food, n_steps, policies, filename

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=argparse.FileType('r'))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    opt = parser.parse_args()

    input_file = opt.input_file
    render = opt.render
    episodes = opt.episodes
    debug = opt.debug
    situations, grid_shape, n_food, n_steps, policies, filename = parse_config(input_file)
    
    population = {}
    greedy_population = {}
    peaceful_population = {}
    avg_energy = {}
    avg_greedy_energy = {}
    avg_peaceful_energy = {}
    deaths = {}
    births = {}

    jobs = []
    return_dicts_proc = []
    manager = multiprocessing.Manager() # create only 1 mgr

    for situation, agents in situations.items():
        return_dict = manager.dict()
        return_dict["situation"] = situation
        return_dicts_proc.append(return_dict)
        environment = ShareOrTake(grid_shape=grid_shape, n_food=n_food, max_steps=n_steps, debug=debug, policy=policies[situation])
        p = multiprocessing.Process(target=run_multi_agent, args=(environment, agents, episodes, return_dict, situation, debug, render))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    for return_dict in return_dicts_proc:
        situation = return_dict["situation"]
        population[situation] = return_dict["population_sit"]
        greedy_population[situation] = return_dict["greedy_sit"]
        peaceful_population[situation] = return_dict["peaceful_sit"]
        avg_energy[situation] = return_dict["avg_energy_sit"]
        avg_greedy_energy[situation] = return_dict["avg_greedy_energy_sit"]
        avg_peaceful_energy[situation] = return_dict["avg_peaceful_energy_sit"]
        deaths[situation] = return_dict["death_rates_sit"]
        births[situation] = return_dict["birth_rates_sit"]
    
    if filename is not None:
        try:
            os.mkdir("../results")
        except:
            pass

    compare_results_pop(
        population,
        title="Population Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric = "Population"
    )

    compare_results_pop(
        greedy_population,
        title="Greedy Population Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric= "Population"
    )

    compare_results_pop(
        peaceful_population,
        title="Peaceful Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric="Population"
    )

    compare_results_pop(
        avg_energy,
        title="Avg. Energy Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric="Energy"
    )

    compare_results_pop(
        avg_greedy_energy,
        title="Avg. Energy (Greedy) Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric="Energy"
    )

    compare_results_pop(
        avg_peaceful_energy,
        title="Avg. Energy (Peaceful) Comparison on 'Share or Take' Environment",
        plot=True,
        colors=COLOURS[:len(situations)],
        filename = filename,
        metric="Energy"
    )

    compare_results_other_metrics(
        deaths,
        title="Death Rate Comparison on 'Share or Take' Environment",
        metric="Death Rate",
        colors=COLOURS[:len(situations)],
        filename = filename
    )

    compare_results_other_metrics(
        births,
        title="Birth Rate Comparison on 'Share or Take' Environment",
        metric="Birth Rate",
        colors=COLOURS[:len(situations)],
        filename = filename
    )