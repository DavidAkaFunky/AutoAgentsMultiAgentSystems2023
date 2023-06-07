from .regular_agent import RegularAgent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class TribalAgent(RegularAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, tribe, energy, reproduction_threshold):
        super().__init__(False, energy, reproduction_threshold)
        self.name = "Tribal agent"
        self.tribe = tribe

    def share_or_take(self, other, food_energy, debug):
        self.has_eaten = True
        try:
            tribe = other.tribe
        except AttributeError:
            tribe = None
        if self.tribe == tribe:
            self.print_if_debug(f"Agents {self.id} and {other.id} belong to the same tribe ({self.tribe}), splitting the food evenly.", debug)
            self.energy += food_energy * 0.5
        else:
            self.print_if_debug(f"Agents {self.id} (from {self.tribe if self.tribe is not None else 'no tribe'}) and {other.id} (from {other.tribe if other.tribe is not None else 'no tribe'}) fought for food, none earned energy.", debug)
            pass # The energy earned with food is lost during the fight

    def __repr__(self) -> str:
        return f"{self.name} ({self.tribe}) - Energy: {self.energy} - Position: {self.pos}"