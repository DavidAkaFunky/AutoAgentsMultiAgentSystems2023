import random
import numpy as np
from .random_agent import RandomAgent
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class RegularAgent(RandomAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, greedy, energy, reproduction_threshold):
        super().__init__(greedy, energy, reproduction_threshold)
        self.name = "Regular agent"
        self.observation = None
        self.vision_range = 4

    def action(self) -> list:
        food_positions = self.observation[1]
        closest_food_positions = self.closest_food(food_positions)
        # Allow the agent to move randomly to eventually find some food
        random_actions = list(range(self.n_actions))[:-1]
        random.shuffle(random_actions)
        choices = []
        if closest_food_positions is not None:
            preferred_actions, extra_actions = [], []
            actions = [self.direction_to_follow_food(pos) for pos in closest_food_positions]
            random.shuffle(actions)
            for action, extra_action in actions:
                preferred_actions.append(action)
                for x in extra_action:
                    extra_actions.append(x)
            choices += preferred_actions + extra_actions + random_actions + [STAY]
        choices += random_actions + [STAY]
        return list(dict.fromkeys(choices))
    
    def see(self, observation: np.ndarray):
        self.observation = observation

    def closest_food(self, food_positions):
        positions = {cityblock(self.pos, food_position): [food_position] for food_position in food_positions}
        order = []
        if len(positions) == 0:
            return None
        for distance in sorted(positions.keys()):
            random.shuffle(positions[distance])
            order += positions[distance]
        return order

    def direction_to_follow_food(self, food_position):
        """
        Given the position of the agent and the position of a food,
        returns the action to take in order to close the distance
        """
        distances = np.array(food_position) - np.array(self.pos)
        abs_distances = np.absolute(distances)
        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances), [UP, DOWN]
        elif abs_distances[1] < abs_distances[0]:
            return self._close_vertically(distances), [LEFT, RIGHT]
        else:
            roll = random.uniform(0, 1)
            if roll > 0.5:
                return self._close_horizontally(distances), [UP, DOWN]
            return self._close_vertically(distances), [LEFT, RIGHT]

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