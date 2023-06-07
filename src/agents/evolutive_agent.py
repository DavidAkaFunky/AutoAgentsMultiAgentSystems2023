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

    def share_or_take(self, other, food_energy, debug):
        self.has_eaten = True
        if self.is_greedy and other.is_greedy:
            # The energy earned with food is lost during the fight
            self.print_if_debug(f"Agents {self.id} and {other.id} are both greedy, none earned energy.", debug)
            self.print_if_debug(f"Evolutive greedy agent {self.id} has now a 50% chance of becoming peaceful.", debug)
            self.adapt_to = "PEACEFUL"
        elif self.is_greedy:
            self.print_if_debug(f"Greedy agent {self.id} is stealing 75% of the food, peaceful agent {other.id} only got 25%.", debug)
            self.energy += food_energy * 0.75
        elif other.is_greedy:
            self.print_if_debug(f"Greedy agent {other.id} is stealing 75% of the food, peaceful agent {self.id} only got 25%.", debug)
            self.print_if_debug(f"Evolutive peaceful agent {self.id} has now a 50% chance of becoming greedy.", debug)
            self.energy += food_energy * 0.25
            self.adapt_to = "GREEDY"
        else:
            self.print_if_debug(f"Agents {self.id} and {other.id} are both peaceful, splitting the food evenly.", debug)
            self.energy += food_energy * 0.5

    def adapt(self, debug):
        value = random.uniform(0, 1)
        if value < self.adaptation_rate:
            if self.adapt_to == "GREEDY":
                self.print_if_debug(f"Evolutive peaceful agent {self.id} has just become greedy.", debug)
                self.is_greedy = True
            elif self.adapt_to == "PEACEFUL":
                self.print_if_debug(f"Evolutive greedy agent {self.id} has just become peaceful.", debug)
                self.is_greedy = False
        self.adapt_to = None
