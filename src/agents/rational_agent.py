import random
from .regular_agent import RegularAgent
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class RationalAgent(RegularAgent):

    """
    A baseline agent for the ShareOrTake environment.
    """

    def __init__(self, greedy, energy, reproduction_threshold):
        super().__init__(greedy, energy, reproduction_threshold)
        self.name = "Rational agent"
        self.observation = None
        self.vision_range = 4

    def action(self) -> int:
        agents_positions = self.observation[0]
        food_positions = self.observation[1]
        reachable_food_positions = []
        for food_pos in food_positions:
            is_reachable = True
            for agent_pos in agents_positions:
                if cityblock(self.pos, food_pos) > cityblock(agent_pos, food_pos):
                    is_reachable = False
                    break
            if is_reachable:
                reachable_food_positions.append(food_pos)
        closest_food_positions = self.closest_food(reachable_food_positions)
        if closest_food_positions is None:
            # Allow the agent to move randomly to eventually find some food
            all_actions = list(range(self.n_actions))[:-1]
        else:
            all_actions = [self.direction_to_follow_food(pos) for pos in closest_food_positions]
        random.shuffle(all_actions)
        return all_actions + [STAY]