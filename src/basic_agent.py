import math
import random
import numpy as np
from scipy.spatial.distance import cityblock

from environment import Agent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class BasicAgent(Agent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, agent_id, greedy):
        super(BasicAgent, self).__init__(f"Basic Agent")
        self.agent_id = agent_id
        self.n_actions = N_ACTIONS
        self.greedy = greedy

    def action(self) -> int:
        agents_positions = self.observation[0]
        food_positions = self.observation[1]
        agent_position = agents_positions[self.agent_id]
        closest_food_position = self.closest_food(agent_position, food_positions)
        return self.direction_to_go(agent_position, closest_food_position)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, food_position):
        """
        Given the position of the agent and the position of a food,
        returns the action to take in order to close the distance
        """
        distances = np.array(food_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_food(self, agent_position, food_positions):
        """
        Given the positions of an agent and a sequence of positions of all food,
        returns the positions of the closest food.
        If there is no food, None is returned instead
        """
        min = math.inf
        closest_food_position = None
        for p in range(len(food_positions)):
            food_position = food_positions[p]
            distance = cityblock(agent_position, food_position)
            if distance < min:
                min = distance
                closest_food_position = food_position
        return closest_food_position

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[0] > 0:
            return RIGHT
        elif distances[0] < 0:
            return LEFT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[1] > 0:
            return DOWN
        elif distances[1] < 0:
            return UP
        else:
            return STAY