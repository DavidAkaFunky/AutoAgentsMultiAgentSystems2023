import random
import numpy as np
from scipy.spatial.distance import cityblock
from .basic_agent import BasicAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class TribalAgent(BasicAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, tribe, greedy, energy, reproduction_threshold):
        super().__init__(greedy, energy, reproduction_threshold)
        self.name = "Tribal agent"
        self.tribe = tribe

    def share_or_take(self, other, food_energy):
        try:
            tribe = other.tribe
        except AttributeError:
            tribe = None
        if self.tribe == tribe:
            self.energy += food_energy * 0.5
        else:
            pass # The energy earned with food is lost during the fight
        self.has_eaten = True

    def __repr__(self) -> str:
        return f"{self.name} ({self.tribe}) - Energy: {self.energy} - Position: {self.pos} - Greedy: {self.is_greedy}"