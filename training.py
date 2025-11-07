import random

from gym_env import PokerEnv, WrappedEval

import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

# The proper action types as specified
action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
int_card_to_str = PokerEnv.int_card_to_str

class CFRTrainer:
    def __init__(self):
        # CFR strategy variables
        self.regrets = defaultdict(lambda: np.zeros(6))  # 6 possible actions
        self.strategy = defaultdict(lambda: np.ones(6)/6)  # Start with uniform strategy
        self.strategy_sum = defaultdict(lambda: np.zeros(6))
        self.iterations = 0
        
        # Hand strength buckets - simplify the state space
        self.strength_buckets = 10  # Number of buckets for hand strength
        
        # Evaluator for hand strength
        self.evaluator = WrappedEval()
        
        # For tracking training progress
        self.utility_history = []
        
        # Define available community card positions by street
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
    
    def get_state_key(self, state):
        """Create a simplified state key from the game state"""
        # Unpack state information
        street = state["street"]
        my_cards = state["my_cards"]
        community_cards = state["community_cards"]
        my_bet = state["my_bet"] 
        opp_bet = state["opp_bet"]
        position = state["position"]  # 0 if first to act, 1 otherwise
        opp_discarded = 1 if state.get("opp_discarded_card", -1) != -1 else 0
        i_discarded = 1 if state.get("discarded_card", -1) != -1 else 0
        
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
        state_key = f"{street}:{hand_strength}:{position}:{bet_level}:{pot_level}:{opp_discarded}:{i_discarded}"
        return state_key
    
    def get_strategy(self, state_key, valid_actions):
        """Get current strategy for this state"""
        # Ensure regrets are properly initialized
        if state_key not in self.regrets:
            self.regrets[state_key] = np.zeros(6)
        
        # Get regrets for this state
        regrets = self.regrets[state_key]
        
        # Update strategy using regret matching
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Compute positive regrets
        pos_regrets = np.maximum(regrets, 0)
        
        # Create a strategy based on regret matching
        strategy = np.zeros(6)
        total_pos_regret = np.sum([pos_regrets[i] for i in valid_action_indices])

        if total_pos_regret > 0:
            # Normalize only among valid actions
            for i in valid_action_indices:
                strategy[i] = max(pos_regrets[i] / total_pos_regret, 0.01)  # Ensure minimum probability
            
            # Normalize to ensure sum is 1
            strategy /= np.sum(strategy)
        else:
            # Uniform distribution among valid actions
            for i in valid_action_indices:
                strategy[i] = 1.0 / len(valid_action_indices)
        
        # Update strategy sum for this state (for average strategy)
        self.strategy_sum[state_key] += strategy
        
        return strategy
    
    def get_average_strategy(self):
        """Get the average strategy across all iterations"""
        avg_strategy = {}
        
        for state_key, strategy_sum in self.strategy_sum.items():
            # Compute average strategy for this state
            total = sum(strategy_sum)
            if total > 0:
                avg_strategy[state_key] = strategy_sum / total
            else:
                # If no accumulated strategy, use uniform
                num_actions = len(strategy_sum)
                avg_strategy[state_key] = np.ones(num_actions) / num_actions
                
        return avg_strategy
    
    def determine_winner(self, p1_state, p2_state):
        """Determine the winner based on hand strength"""
        # Get current community cards based on street
        street = max(p1_state["street"], p2_state["street"])
        
        # Ensure we have valid community cards
        community_cards = p1_state.get("community_cards", [])
        if not community_cards or all(c == -1 for c in community_cards):
            community_cards = p2_state.get("community_cards", [])
            
        # If we have folded or no community cards, compare hole cards
        if street == 0 or not community_cards or all(c == -1 for c in community_cards):
            p1_strength = self.get_hand_strength(p1_state["my_cards"], [])
            p2_strength = self.get_hand_strength(p2_state["my_cards"], [])
        else:
            p1_strength = self.get_hand_strength(p1_state["my_cards"], community_cards)
            p2_strength = self.get_hand_strength(p2_state["my_cards"], community_cards)
            
        # Compare hand strengths to determine winner
        if p1_strength > p2_strength:
            return 1  # Player 1 wins
        elif p2_strength > p1_strength:
            return -1  # Player 2 wins
        else:
            return 0  # Draw
    
    def cfr(self, states, player, reach_p1, reach_p2, valid_actions_list, depth=0, max_depth=4):
        """Recursive CFR algorithm implementation"""
        # Check for terminal conditions
        p1_state, p2_state = states
        
        # If max depth reached or final street reached, evaluate hands
        if depth >= max_depth or p1_state["street"] >= 4 or p2_state["street"] >= 4:
            return self.determine_winner(p1_state, p2_state)
        
        # Check for fold (terminal)
        if p1_state.get("folded", False):
            return -1  # P1 folded, P2 wins
        if p2_state.get("folded", False):
            return 1   # P2 folded, P1 wins
            
        # Current player's state
        current_state = states[player]
        valid_actions = valid_actions_list[player]
        
        # Get state key
        state_key = self.get_state_key(current_state)
        
        # Get strategy for this information set
        strategy = self.get_strategy(state_key, valid_actions)
        
        # Get valid action indices
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Initialize expected value
        node_value = 0
        action_values = np.zeros(len(strategy))
        
        # Recursively evaluate each action
        for action in valid_action_indices:
            # Copy states for the next recursion
            next_states = [state.copy() for state in states]
            next_valid_actions = [actions.copy() for actions in valid_actions_list]
            
            # Apply action effects
            self.apply_action(next_states, player, action, next_valid_actions)
            
            # Calculate reach probabilities for recursive call
            next_reach_p1 = reach_p1 * strategy[action] if player == 0 else reach_p1
            next_reach_p2 = reach_p2 * strategy[action] if player == 1 else reach_p2
            
            # Recurse (flip player unless the current player gets to act again)
            next_player = 1 - player
            action_value = -self.cfr(next_states, next_player, next_reach_p1, next_reach_p2, next_valid_actions, depth + 1, max_depth)
            
            # Update action value
            action_values[action] = action_value
            
            # Update node value
            node_value += strategy[action] * action_value
        
        # Update regrets for the current player
        reach_prob = reach_p2 if player == 0 else reach_p1
        for action in valid_action_indices:
            regret = action_values[action] - node_value
            self.regrets[state_key][action] += reach_prob * regret
        
        return node_value
    
    def apply_action(self, states, player, action, valid_actions_list):
        """Apply the effects of an action to the game state"""
        current_state = states[player]
        opponent_state = states[1 - player]
        
        # Determine if this action completes a betting round
        completes_betting = False
        
        if action == action_types.FOLD.value:
            # Player folds, opponent wins
            current_state["folded"] = True
            completes_betting = True
            
        elif action == action_types.CHECK.value:
            # Player checks
            if current_state["my_bet"] == opponent_state["opp_bet"]:
                # Check when bets are equal completes the betting round
                completes_betting = True
            
        elif action == action_types.CALL.value:
            # Player calls
            bet_difference = opponent_state["my_bet"] - current_state["my_bet"]
            current_state["my_bet"] += bet_difference
            opponent_state["opp_bet"] += bet_difference
            completes_betting = True
            
        elif action == action_types.RAISE.value:
            # Player raises
            # Simplified: use a reasonable raise amount (2x current bet or 2 if no current bet)
            raise_amount = max(2, 2 * current_state["my_bet"])
            current_state["my_bet"] += raise_amount
            opponent_state["opp_bet"] += raise_amount
            
            # Update valid actions for opponent (they can't check now)
            valid_actions_list[1 - player][action_types.CHECK.value] = False
            
        elif action == action_types.DISCARD.value:
            # Player discards a card
            if len(current_state["my_cards"]) > 1:
                # Discard worst card (simplified)
                card_ranks = [card % 9 for card in current_state["my_cards"]]
                worst_card_idx = np.argmin(card_ranks)
                discarded = current_state["my_cards"].pop(worst_card_idx)
                current_state["discarded_card"] = discarded
                opponent_state["opp_discarded_card"] = discarded
                
                # Draw new card
                available_cards = list(range(27))  # 27 cards in the deck
                for card in current_state["my_cards"] + opponent_state["my_cards"]:
                    if card in available_cards:
                        available_cards.remove(card)
                
                if available_cards:
                    new_card = random.choice(available_cards)
                    current_state["my_cards"].append(new_card)
        
        # If betting round is complete, advance to next street
        if completes_betting:
            if current_state["street"] < 3:  # If not on river yet
                next_street = current_state["street"] + 1
                
                # Update both states to next street
                states[0]["street"] = next_street
                states[1]["street"] = next_street
                
                # Deal community cards for the new street if needed
                if "community_cards" not in states[0] or len(states[0]["community_cards"]) == 0:
                    # Create community cards
                    used_cards = states[0]["my_cards"] + states[1]["my_cards"]
                    if "discarded_card" in states[0] and states[0]["discarded_card"] != -1:
                        used_cards.append(states[0]["discarded_card"])
                    if "discarded_card" in states[1] and states[1]["discarded_card"] != -1:
                        used_cards.append(states[1]["discarded_card"])
                    
                    available_cards = list(range(27))
                    for card in used_cards:
                        if card in available_cards:
                            available_cards.remove(card)
                    
                    # Deal community cards based on street
                    num_cards = len(self.community_cards_by_street[next_street])
                    community_cards = random.sample(available_cards, min(num_cards, len(available_cards)))
                    
                    # Pad to 5 with -1 values
                    while len(community_cards) < 5:
                        community_cards.append(-1)
                        
                    states[0]["community_cards"] = community_cards
                    states[1]["community_cards"] = community_cards
                
                # Reset betting for new street
                states[0]["my_bet"] = 0
                states[0]["opp_bet"] = 0
                states[1]["my_bet"] = 0
                states[1]["opp_bet"] = 0
                
                # Reset valid actions
                for p in range(2):
                    valid_actions_list[p] = [True, True, True, True, True, False]
    
    def generate_random_deal(self):
        """Generate a random initial deal for training"""
        # Create a deck of cards
        deck = list(range(27))  # 27 cards in our modified deck
        random.shuffle(deck)
        
        # Deal hole cards
        p1_cards = [deck.pop() for _ in range(2)]
        p2_cards = [deck.pop() for _ in range(2)]
        
        # Initial states
        p1_state = {
            "street": 0,
            "my_cards": p1_cards,
            "community_cards": [],
            "my_bet": 1,  # Small blind
            "opp_bet": 2,  # Big blind
            "position": 0,  # First to act preflop
            "opp_discarded_card": -1
        }
        
        p2_state = {
            "street": 0,
            "my_cards": p2_cards,
            "community_cards": [],
            "my_bet": 2,  # Big blind
            "opp_bet": 1,  # Small blind
            "position": 1,  # Second to act preflop
            "opp_discarded_card": -1
        }
        
        return p1_state, p2_state
    
    def train(self, num_iterations=10000, save_interval=1000, save_path="poker_strategy.pkl"):
        """Train the CFR algorithm for a number of iterations"""
        print(f"Beginning CFR training for {num_iterations} iterations...")
        
        # Initialize progress tracking
        total_utility = 0
        
        for i in tqdm(range(num_iterations)):
            # Generate random initial game state
            p1_state, p2_state = self.generate_random_deal()
            
            # Valid actions for each player
            # FOLD, RAISE, CHECK, CALL, DISCARD, INVALID
            p1_valid_actions = [True, True, False, True, True, False]  # SB can't check preflop
            p2_valid_actions = [True, True, True, False, True, False]  # BB can't call preflop
            
            # Run CFR iteration
            utility = self.cfr(
                [p1_state, p2_state], 
                0,  # Player 1 starts
                1.0, 1.0,  # Initial reach probabilities
                [p1_valid_actions, p2_valid_actions],
                max_depth=4  # Increased depth to handle more complex games
            )
            
            total_utility += utility
            self.iterations += 1
            self.utility_history.append(total_utility / (i + 1))
            
            # Save strategy periodically
            if (i + 1) % save_interval == 0:
                avg_strategy = self.get_average_strategy()
                with open(save_path, 'wb') as f:
                    pickle.dump(avg_strategy, f)
                
                # Display progress
                print(f"Iteration {i+1}/{num_iterations}: Average utility: {total_utility/(i+1):.4f}")
                print(f"Number of unique states: {len(self.strategy_sum)}")
                
                # Save utility history
                with open(save_path + ".utility", 'wb') as f:
                    pickle.dump(self.utility_history, f)
                
        print(f"Training complete. Final average utility: {total_utility/num_iterations:.4f}")
        
        # Save final strategy
        avg_strategy = self.get_average_strategy()
        with open(save_path, 'wb') as f:
            pickle.dump(avg_strategy, f)
        
        return avg_strategy
    
    def save_strategy(self, filename="poker_strategy.pkl"):
        """Save the current average strategy to a file"""
        avg_strategy = self.get_average_strategy()
        with open(filename, 'wb') as f:
            pickle.dump(avg_strategy, f)
        print(f"Strategy saved to {filename}")

# Function to implement offline CFR training for the poker agent
def train_cfr_agent(iterations=100000, save_interval=1000, save_path="poker_strategy.pkl"):
    """
    Train a poker agent using CFR and save the strategy to a file.
    
    Parameters:
    iterations (int): Number of training iterations
    save_interval (int): How often to save the strategy
    save_path (str): File path to save the strategy
    
    Returns:
    dict: The trained average strategy
    """
    print(f"Starting CFR training for {iterations} iterations...")
    trainer = CFRTrainer()
    
    # Run the training
    strategy = trainer.train(
        num_iterations=iterations,
        save_interval=save_interval,
        save_path=save_path
    )
    
    print(f"Training complete. Final strategy saved to {save_path}")
    print(f"Strategy includes {len(strategy)} unique game states")
    
    return strategy

if __name__ == "__main__":
    # Train the agent
    train_cfr_agent(iterations=10000, save_path="poker_strategy.pkl")

    print("Training Complete")