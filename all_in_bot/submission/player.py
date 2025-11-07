import random

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["street"] == 0:
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")

        if observation["valid_actions"][action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            raise_amount = observation["max_raise"]
            if raise_amount > 20:
                self.logger.info(f"Going all-in for {raise_amount}")
        elif observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
            raise_amount = 0
        else:
            action_type = action_types.CHECK.value
            raise_amount = 0

        card_to_discard = -1
        return action_type, raise_amount, card_to_discard

