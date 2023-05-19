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

    def __init__(self, greedy):
        super(BasicAgent, self).__init__(f"Basic Agent")
        self.vision_range = 2
        self.living_cost = 1
        self.move_cost = 2
        self.energy = 10
        self.pos = None
        self.n_actions = N_ACTIONS
        self.is_greedy = greedy
        self.has_eaten = False
        self.agent_id = None

    def action(self) -> int:
        print(self.pos, self.observation)
        print("-------------")
        agents_positions = self.observation[0]
        food_positions = self.observation[1]
        closest_food_positions = self.closest_food(self.pos, food_positions)
        print (closest_food_positions)
        if closest_food_positions is None:
            return [STAY]
        return [self.direction_to_go(self.pos, pos) for pos in closest_food_positions] + [STAY]

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def direction_to_go(self, agent_position, food_position):
        """
        Given the position of the agent and the position of a food,
        returns the action to take in order to close the distance
        """
        if food_position is None:
            return STAY
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
            returns the positions of the closest food blocks.
            If there is no food, None is returned instead
        """
        min = math.inf
        closest_food_positions = None
        for p in range(len(food_positions)):
            food_position = food_positions[p]
            distance = cityblock(agent_position, food_position)
            if distance < min:
                min = distance
                closest_food_positions = [food_position]
            elif distance == min:
                closest_food_positions.append(food_position)
        return closest_food_positions

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