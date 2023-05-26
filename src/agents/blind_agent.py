import math
import random
import numpy as np
from scipy.spatial.distance import cityblock

from environment import Agent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class BlindAgent(Agent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, greedy, energy, reproduction_threshold):
        self.name = "Blind agent"
        self.reproduction_threshold = reproduction_threshold
        self.vision_range = 1
        self.living_cost = 1
        self.move_cost = 1
        self.base_energy = energy
        self.energy = energy
        self.pos = None
        self.n_actions = N_ACTIONS
        self.is_greedy = greedy
        self.has_eaten = False
        self.id = None

    def action(self) -> int:
        # Allow the agent to move randomly to eventually find some food
        all_actions = list(range(self.n_actions))[:-1]
        random.shuffle(all_actions)
        return all_actions + [STAY]
    
    def see(self, observation: np.ndarray):
        pass

    def feedback(self, reward: float):
        self.energy += reward

    def reset_parameters(self, id):
        self.energy = 20
        self.has_eaten = False
        self.id = id

    def share_or_take(self, other, food_energy):
        if self.is_greedy and other.is_greedy:
            pass # The energy earned with food is lost during the fight
        elif self.is_greedy:
            self.energy += food_energy * 0.75
        elif other.is_greedy:
            self.energy += food_energy * 0.25
        else:
            self.energy += food_energy * 0.5
        self.has_eaten = True

    def __repr__(self) -> str:
        return f"{self.name} - Energy: {self.energy} - Position: {self.pos} - Greedy: {self.is_greedy}"

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
        
        #print("Agent {} is at position {} and food is at position {} with distance {}".format(
        #    self.id, agent_position, tuple(reversed(food_position)), distances))
        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances)
        elif abs_distances[1] < abs_distances[0]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_food(self, agent_position, food_positions):
        positions = {cityblock(agent_position, food_position): [food_position] for food_position in food_positions}
        order = []
        if len(positions) == 0:
            return None
        for distance in sorted(positions.keys()):
            random.shuffle(positions[distance])
            order += positions[distance]
        return order

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[1] > 0:
            #print("Agent {} is going right".format(self.id))
            return RIGHT
        elif distances[1] < 0:
            #print("Agent {} is going left".format(self.id))
            return LEFT
        else:
            #print("Agent {} is staying".format(self.id))
            return STAY

    def _close_vertically(self, distances):
        if distances[0] > 0:
            #print("Agent {} is going down".format(self.id))
            return DOWN
        elif distances[0] < 0:
            #print("Agent {} is going up".format(self.id))
            return UP
        else:
            #print("Agent {} is staying".format(self.id))
            return STAY