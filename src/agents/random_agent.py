import random
import numpy as np

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class RandomAgent():

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
        self.energy = self.base_energy
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