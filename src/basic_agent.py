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
        self.reproduction_threshold = 25
        self.vision_range = 2
        self.living_cost = 1
        self.move_cost = 2
        self.energy = 20
        self.pos = None
        self.n_actions = N_ACTIONS
        self.is_greedy = greedy
        self.has_eaten = False
        self.id = None

    def action(self) -> int:
        agents_positions = self.observation[0]
        food_positions = self.observation[1]
        print(food_positions)
        closest_food_positions = self.closest_food(self.pos, food_positions)
        if closest_food_positions is None:
            # Allow the agent to move randomly to eventually find some food
            all_actions = list(range(self.n_actions))
            random.shuffle(all_actions)
            return all_actions
        return [self.direction_to_go(self.pos, pos) for pos in closest_food_positions] + [STAY]
    
    def feedback(self, reward: float):
        self.energy += reward

    def reset_parameters(self, id):
        self.energy = 20
        self.has_eaten = False
        self.id = id

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
        distances = np.array(tuple(reversed(food_position)) - np.array(agent_position))
        abs_distances = np.absolute(distances)
        
        print("Agent {} is at position {} and food is at position {} with distance {}".format(
            self.id, agent_position, tuple(reversed(food_position)), distances))
        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances)
        elif abs_distances[1] < abs_distances[0]:
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
        if distances[1] > 0:
            print("Agent {} is going right".format(self.id))
            return RIGHT
        elif distances[1] < 0:
            print("Agent {} is going left".format(self.id))
            return LEFT
        else:
            print("Agent {} is staying".format(self.id))
            return STAY

    def _close_vertically(self, distances):
        if distances[0] > 0:
            print("Agent {} is going down".format(self.id))
            return DOWN
        elif distances[0] < 0:
            print("Agent {} is going up".format(self.id))
            return UP
        else:
            print("Agent {} is staying".format(self.id))
            return STAY