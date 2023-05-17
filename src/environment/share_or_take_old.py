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

    def __init__(self, grid_shape=(50, 50), initial_n_agents=20, n_food=10, step_cost=-1, max_steps=100):

        # General grid parameters
        self._grid_shape = grid_shape
        self._max_steps = max_steps
        self._step_cost = step_cost
        self._step_count = 0

        # RNG seed
        self.seed()

        # Agents
        self._agent_view_mask = (1, 1)
        self._full_obs = self.__create_grid()
        self.agent_id = 0 # Used to assign unique agent IDs
        self.n_agents = 0
        self.agent_pos = {}
        for _ in range(initial_n_agents):
            self.__add_agent()

        # Food
        self.n_food = n_food
        self.available_n_food = 0
        self.food_pos = set()
        self.__add_remaining_food()

        # Rendering
        self.__draw_base_img()
        self.viewer = None

    def reset(self):
        return [self.observation(id) for id in self.agent_pos]

    def observation(self, id):
        agent_row, agent_col = self.agent_pos[id]
        agent_pos = []
        for agent_id in self.agent_pos:
            row, col = self.agent_pos[agent_id]
            if abs(row - agent_row) <=
            agent_pos.append((col, row))

        food_pos = []
        for (row, col) in self.food_pos:

            food_pos.append((col, row))

        return agent_pos, food_pos

    def step(self, agents_action):
        self._step_count += 1
        finished = False
        rewards = [self._step_cost for _ in range(self.n_agents)]
        
        # TODO: This will receive a queue of actions, iterate over them
        # and stop once they have moved or ran out of options
        for agent_i, action in enumerate(agents_action):
            self.__update_agent_pos(agent_i, action)

        if (self._step_count >= self._max_steps):
            finished = True

        return [self.observation() for _ in range(self.n_agents)], rewards, finished

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __add_remaining_food(self):
        """ 
        Add food to the grid until the desired number is reached.
        This is called to initialise the environment and after an agent eats food.
        """
        while self.available_n_food < self.n_food:
            while True:
                pos = (self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1))
                if self._is_cell_vacant(pos):
                    self.food_pos.add(pos)
                    break
            self.available_n_food += 1
            self.__update_food_view(pos)

    def __add_agent(self):
        """
        Spawn a new agent into the grid.
        """
        while True:
            pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                   self.np_random.randint(0, self._grid_shape[1] - 1)]
            if self._is_cell_vacant(pos):
                self.agent_pos[self.agent_id] = pos
                break
        self.__update_agent_view(self.n_agents)
        self.n_agents += 1
        self.agent_id += 1


    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
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

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_food_view(self, pos):
        self._full_obs[pos[0]][pos[1]] = PRE_IDS['food']

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def __get_neighbour_coordinates(self, pos):
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
        img = copy.copy(self._base_img)
        for agent_i in self.agent_pos:
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in self.agent_pos:
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
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
