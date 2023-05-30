import random
import numpy as np
from scipy.spatial.distance import cityblock
from .blind_agent import BlindAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class BasicAgent(BlindAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, greedy, energy, reproduction_threshold):
        super().__init__(greedy, energy, reproduction_threshold)
        self.name = "Basic agent"
        self.observation = None
        self.vision_range = 5

    def action(self) -> int:
        # agents_positions = self.observation[0]
        food_positions = self.observation[1]
        # print(food_positions)
        closest_food_positions = self.closest_food(self.pos, food_positions)
        if closest_food_positions is None:
            # Allow the agent to move randomly to eventually find some food
            all_actions = list(range(self.n_actions))[:-1]
        else:
            all_actions = [self.direction_to_go(self.pos, pos) for pos in closest_food_positions]
        random.shuffle(all_actions)
        return all_actions + [STAY]
    
    def see(self, observation: np.ndarray):
        self.observation = observation