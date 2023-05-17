import copy
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor
import gym
from gym.utils import seeding

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

class ShareOrTake(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, agents, grid_shape=(50, 50), n_food=10, max_steps=100):

        # General grid parameters
        self.grid_shape = grid_shape
        self.max_steps = max_steps
        self.step_count = 0

        # RNG seed
        self.seed()

        # Agents
        self.grid = self.create_grid()
        self.agent_id = 0 # Used to assign unique agent IDs
        self.agents = {} # Dictionary of agents
        self.add_all_agents(agents)

        # Food
        self.n_food = n_food
        self.available_n_food = 0
        self.food_pos = set()
        self.add_remaining_food()

        # Rendering
        self.draw_base_img()
        self.viewer = None

    def add_all_agents(self, agents):
        """
        Spawn a new agent into the grid.
        """
        for agent in agents:
            while True:
                pos = [self.np_random.randint(0, self.grid_shape[0] - 1),
                    self.np_random.randint(0, self.grid_shape[1] - 1)]
                if self.is_cell_vacant(pos):
                    break
            agent.set_position(pos)
            self.agents[self.agent_id] = agent
            self.update_agent_view(self.agent_id)
            self.agent_id += 1

    def add_remaining_food(self):
        """ 
        Add food to the grid until the desired number is reached.
        This is called to initialise the environment and after an agent eats food.
        """
        while self.available_n_food < self.n_food:
            while True:
                pos = (self.np_random.randint(0, self.grid_shape[0] - 1),
                       self.np_random.randint(0, self.grid_shape[1] - 1))
                if self.is_cell_vacant(pos):
                    break
            self.food_pos.add(pos)
            self.available_n_food += 1
            self.update_food_view(pos)

    def reset(self):
        return [self.observation(id) for id in self.agents]

    def observation(self, id):
        agent = self.agents[id]
        [agent_row, agent_col] = agent.get_position()

        agents_pos = []
        for other_agent_id in self.agents:
            [row, col] = self.agents[other_agent_id].get_position()
            if abs(row - agent_row) <= agent.vision_range and abs(col - agent_col) <= agent.vision_range:
                agents_pos.append((col, row))

        food_pos = []
        for (row, col) in self.food_pos:
            if abs(row - agent_row) <= agent.vision_range and abs(col - agent_col) <= agent.vision_range:
                agents_pos.append((col, row))

        return agents_pos, food_pos

    def step(self, agents_action):
        self.step_count += 1
        finished = False
        rewards = [0 for _ in self.agents]
        
        # TODO: This will receive a queue of actions, iterate over them
        # and stop once they have moved or ran out of options
        for agent_i, action in enumerate(agents_action):
            self.update_agent_pos(agent_i, action)

        if (self.step_count >= self.max_steps):
            finished = True

        return [self.observation(id) for id in self.agents], rewards, finished

    def draw_base_img(self):
        self.base_img = draw_grid(self.grid_shape[0], self.grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def create_grid(self):
        grid = [[PRE_IDS['empty'] for _ in range(self.grid_shape[1])] for row in range(self.grid_shape[0])]
        return grid

    def is_valid(self, pos):
        return (0 <= pos[0] < self.grid_shape[0]) and (0 <= pos[1] < self.grid_shape[1])

    def is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self.grid[pos[0]][pos[1]] == PRE_IDS['empty'])

    def update_agent_pos(self, agent_i, move):
        agent = self.agents[agent_i]
        curr_pos = agent.get_position()
        next_pos = None
        if move == 0:    # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self.is_cell_vacant(next_pos):
            agent.set_position(next_pos)
            self.grid[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.update_agent_view(agent_i)

    def update_agent_view(self, agent_i):
        pos = self.agents[agent_i].get_position()
        self.grid[pos[0]][pos[1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def update_food_view(self, pos):
        self.grid[pos[0]][pos[1]] = PRE_IDS['food']

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self.grid[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self.grid[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self.grid[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self.grid[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self.grid[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        img = copy.copy(self.base_img)
        for agent_i in self.agents:
            pos = self.agents[agent_i].get_position()
            for neighbour in self.get_neighbour_coordinates(pos):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, pos, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

            draw_circle(img, pos, cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=pos, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for food in self.food_pos:
            draw_circle(img, food, cell_size=CELL_SIZE, fill=FOOD_COLOR)
            write_cell_text(img, text="", pos=food, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
FOOD_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'food': 'F',
    'wall': 'W',
    'empty': '0'
}
