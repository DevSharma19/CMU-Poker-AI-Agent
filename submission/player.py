from collections import defaultdict
import os
import pickle

import numpy as np
from agents.agent import Agent
from gym_env import PokerEnv, WrappedEval
import random

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
int_card_to_str = PokerEnv.int_card_to_str

class PlayerAgent(Agent):
    def __name__(self):
        return "CFRPlayerAgent"

    def __init__(self, stream: bool = True, strategy_path="submission/strategies/poker_strategy.pkl"):
        super().__init__(stream)
        
        # Load pre-trained strategy
        try:
            with open(strategy_path, 'rb') as f:
                self.strategy = pickle.load(f)
        except FileNotFoundError:
            self.logger.warning(f"No strategy file found at {strategy_path}. Using random strategy.")
            self.strategy = None
        
        # Tracking variables
        self.hand_number = 0
        self.won_hands = 0
        self.last_state_key = None
        self.last_action = None
        
        # Fallback exploration parameter
        self.exploration_rate = 0.1
        
        # Evaluator
        self.evaluator = WrappedEval()
        
        # Community cards by street (matching CFR trainer)
        self.community_cards_by_street = {
            0: [],         # Preflop: no community cards
            1: [0, 1, 2],  # Flop: first three cards
            2: [0, 1, 2, 3],  # Turn: first four cards
            3: [0, 1, 2, 3, 4]   # River: all five cards
        }
    
    def get_hand_strength(self, my_cards, community_cards):
        """Calculate relative hand strength bucket (0-9, higher is better)"""
        if not my_cards or len(my_cards) == 0:
            return 0
            
        # Filter out -1 values from community cards
        valid_community = [c for c in community_cards if c != -1]
        
        # If we have no community cards, evaluate hand strength differently
        if len(valid_community) == 0:
            # Check for pairs
            if len(my_cards) >= 2 and my_cards[0] % 9 == my_cards[1] % 9:
                if (my_cards[0] % 9 >= 5):
                    return 9  # Hi Pocket Pair, i.e. 7s 8s 9s As
                else:
                    return 8  # Lo Pocket Pair
            
            # Check for suited cards
            if len(my_cards) >= 2 and my_cards[0] // 9 == my_cards[1] // 9:
                return 6  # Suited cards
            
            # Check for connected cards (adjacent ranks)
            if len(my_cards) >= 2:
                ranks = [card % 9 for card in my_cards]
                if abs(ranks[0] - ranks[1]) == 1 or abs(ranks[0] - ranks[1]) == 8:  # 8 for A-2 connection
                    return 5  # Connected cards
                
                # Check for high cards
                high_cards = sum(1 for rank in ranks if rank >= 7)  # High Cards, i.e. 9 or A
                if high_cards > 0:
                    return 4 + high_cards  # 1 or 2 high cards
                
                # Otherwise, map remaining hand strength linearly from 0-3
                total_rank = sum(ranks)
                return min(3, total_rank // 3)
        
        # With community cards, use evaluator
        try:
            # Convert card ints to card format expected by evaluator
            my_board = list(map(int_to_card, valid_community))
            my_hand = list(map(int_to_card, [c for c in my_cards if c != -1]))
            
            if len(my_hand) == 0 or len(my_board) == 0:
                return 0
                
            hand_rank = self.evaluator.evaluate(
                my_hand,
                my_board,
            )
                
            # Convert to 0-9 scale (higher is better)
            # Hand_rank is lower for better hands
            # Range is approximately 1 (royal flush) to 7462 (high card)
            norm_rank = min(9, max(0, 9 - int(hand_rank / 830)))
            return norm_rank
        except Exception as e:
            # Fallback if evaluation fails
            return 0
    
    def get_state_key(self, observation):
        """Create a simplified state key similar to CFR trainer"""
        street = observation["street"]
        my_cards = observation["my_cards"]
        community_cards = observation["community_cards"]
        my_bet = observation["my_bet"] 
        opp_bet = observation["opp_bet"]
        position = 0 if observation["acting_agent"] == 0 else 1
        opp_discarded = 1 if observation.get("opp_discarded_card", -1) != -1 else 0
        i_discarded = 1 if observation.get("opp_discarded_card", -1) != -1 else 0
        
        # Calculate pot size and bet level
        pot_size = my_bet + opp_bet
        bet_level = 0
        if opp_bet > my_bet:
            bet_level = 1  # Facing bet
        elif my_bet > 0 and opp_bet == my_bet:
            bet_level = 2  # After raise
        
        # Calculate hand strength
        hand_strength = self.get_hand_strength(
            my_cards, 
            community_cards
        )
        
        # Simplify pot size to small/medium/large
        if pot_size < 10:
            pot_level = 0  # Small
        elif pot_size < 30:
            pot_level = 1  # Medium
        else:
            pot_level = 2  # Large
        
        # Create state key
        state_key = f"{street}:{hand_strength}:{position}:{bet_level}:{pot_level}:{opp_discarded}:0"
        print(state_key)
        return state_key
    
    def choose_action_with_strategy(self, observation, valid_actions):
        """Choose action based on learned strategy or fallback to exploration"""
        state_key = self.get_state_key(observation)
        self.last_state_key = state_key
        
        # If no pre-trained strategy, use random strategy
        if self.strategy is None or state_key not in self.strategy:
            print("No strategy for current game state, defaulting to random")
            # Fallback to random valid action
            valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            return random.choice(valid_action_indices), 0, -1
        
        # Get the learned strategy for this state
        learned_strategy = self.strategy[state_key]
        
        # Add exploration
        if random.random() < self.exploration_rate:
            valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            return random.choice(valid_action_indices), 0, -1
        
        # Normal strategy selection with probability matching
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Extract probabilities for valid actions
        valid_action_probs = [learned_strategy[i] for i in valid_action_indices]
        
        # Normalize probabilities
        total_prob = sum(valid_action_probs)
        valid_action_probs = [p/total_prob for p in valid_action_probs]
        
        # Choose action based on strategy
        print(f"Current strategy : {valid_action_indices, valid_action_probs}")
        action_index = np.random.choice(valid_action_indices, p=valid_action_probs)
        
        # For raises, determine appropriate raise amount
        raise_amount = 0
        if action_index == action_types.RAISE.value:
            action_index = action_types.FOLD.value # TODO : FIX THIS
        
        # For discard, implement strategic discard
        card_to_discard = -1
        if action_index == action_types.DISCARD.value:
            card_to_discard = self.choose_discard_card(observation)
        
        return action_index, raise_amount, card_to_discard
    
    def choose_discard_card(self, observation):
        """Strategic card discard based on hand strength"""
        my_cards = observation["my_cards"]
        
        # If only one card, can't discard
        if len(my_cards) <= 1:
            return -1
        
        # Get card ranks
        ranks = [card % 9 for card in my_cards]
        
        # If pair, keep the highest pair card
        if ranks[0] == ranks[1]:
            return -1
        
        # Prefer keeping aces
        if 8 in ranks:  # Ace is 8 in this representation
            return 0 if ranks[0] != 8 else 1
        
        # Discard the lower-ranked card
        return 0 if ranks[0] < ranks[1] else 1
    
    def act(self, observation, reward, terminated, truncated, info):
        # First, get the list of valid actions we can take
        valid_actions = observation["valid_actions"]
        
        # Choose action based on strategy
        action_type, raise_amount, card_to_discard = self.choose_action_with_strategy(
            observation, valid_actions
        )
        
        # Save last action for potential learning
        self.last_action = (action_type, raise_amount, card_to_discard)
        
        return action_type, raise_amount, card_to_discard
    
    def observe(self, observation, reward, terminated, truncated, info):
        """Update stats after each hand"""
        # Increment hand count when game terminates
        if terminated:
            self.hand_number += 1
            
            # Track win rate
            if reward > 0:
                self.won_hands += 1
            
            # Occasionally log performance
            if self.hand_number % 100 == 0:
                win_rate = self.won_hands / self.hand_number
                self.logger.info(f"Hand #{self.hand_number}, Win Rate: {win_rate:.2%}")

class RandomAgent(Agent):
    def __name__(self):
        return "RandomAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0

    def act(self, observation, reward, terminated, truncated, info):
        # Example of using the logger
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']}")

        # First, get the list of valid actions we can take
        valid_actions = observation["valid_actions"]
        
        # Get indices of valid actions (where value is 1)
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Randomly choose one of the valid action indices
        action_type = random.choice(valid_action_indices)
        
        # Set up our response values
        raise_amount = 0
        card_to_discard = -1  # -1 means no discard
        
        # If we chose to raise, pick a random amount between min and max
        if action_type == action_types.RAISE.value:
            if observation["min_raise"] == observation["max_raise"]:
                raise_amount = observation["min_raise"]
            else:
                raise_amount = random.randint(
                    observation["min_raise"],
                    observation["max_raise"]
                )
        
        # If we chose to discard, randomly pick one of our two cards (0 or 1)
        if action_type == action_types.DISCARD.value:
            card_to_discard = random.randint(0, 1)
        
        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")