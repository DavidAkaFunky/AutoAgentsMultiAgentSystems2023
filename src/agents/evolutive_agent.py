import random
from .rational_agent import RationalAgent
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class EvolutiveAgent(RationalAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, greedy, energy, reproduction_threshold):
        super().__init__(greedy, energy, reproduction_threshold)
        self.name = "Evolutive agent"
        self.adaptation_rate = 0.5
        self.adapt_to = None

    def share_or_take(self, other, food_energy):
        self.has_eaten = True
        if self.is_greedy and other.is_greedy:
            # The energy earned with food is lost during the fight
            self.adapt_to = "PEACEFUL"
        elif self.is_greedy:
            self.energy += food_energy * 0.75
        elif other.is_greedy:
            self.energy += food_energy * 0.25
            self.adapt_to = "GREEDY"
        else:
            self.energy += food_energy * 0.5

    def adapt(self):
        value = random.uniform(0, 1)
        if value < self.adaptation_rate:
            if self.adapt_to == "GREEDY":
                self.is_greedy = True
            elif self.adapt_to == "PEACEFUL":
                self.is_greedy = False
        self.adapt_to = None
