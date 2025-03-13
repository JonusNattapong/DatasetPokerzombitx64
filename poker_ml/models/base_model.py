"""
Base model class for poker AI models.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod


class BasePokerModel(ABC):
    """
    Abstract base class for poker AI models.
    All poker models should inherit from this class.
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize the base poker model.
        
        Args:
            name: Name of the model
        """
        self.name = name
    
    @abstractmethod
    def predict_action(self, state: Dict[str, Any]) -> str:
        """
        Predict the next action given the current game state.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Predicted action (e.g., 'fold', 'call', 'check', 'bet', 'raise')
        """
        pass
    
    @abstractmethod
    def update(self, state: Dict[str, Any], action: str, reward: float) -> None:
        """
        Update the model based on state, action, and reward.
        
        Args:
            state: Dictionary containing the game state
            action: Action taken
            reward: Reward received
        """
        pass
    
    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get probability distribution over possible actions.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        return {"fold": 0.0, "check": 0.0, "call": 0.0, "bet": 0.0, "raise": 0.0}
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        raise NotImplementedError("Model saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model
        """
        raise NotImplementedError("Model loading not implemented")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {}


class RandomModel(BasePokerModel):
    """
    A simple model that makes random decisions.
    Useful for testing and baseline comparisons.
    """
    
    def __init__(self, name: str = "RandomModel", seed: Optional[int] = None):
        """
        Initialize the random model.
        
        Args:
            name: Name of the model
            seed: Random seed for reproducibility
        """
        super().__init__(name)
        self.rng = np.random.RandomState(seed)
        
    def predict_action(self, state: Dict[str, Any]) -> str:
        """
        Predict a random action from available actions.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Randomly selected action
        """
        available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
        return self.rng.choice(available_actions)
    
    def update(self, state: Dict[str, Any], action: str, reward: float) -> None:
        """
        Update method (no-op for random model).
        
        Args:
            state: Dictionary containing the game state
            action: Action taken
            reward: Reward received
        """
        # Random model doesn't learn, so this is a no-op
        pass
    
    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get uniform probability distribution over available actions.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Dictionary mapping actions to equal probabilities
        """
        available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
        prob = 1.0 / len(available_actions)
        return {action: prob for action in available_actions}


class RuleBasedModel(BasePokerModel):
    """
    A model that makes decisions based on predefined rules.
    """
    
    def __init__(self, name: str = "RuleBasedModel", rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the rule-based model.
        
        Args:
            name: Name of the model
            rules: Dictionary of rules for decision making
        """
        super().__init__(name)
        self.rules = rules or {}
        self.default_rules = {
            # Default preflop strategy
            "preflop": {
                # Hand strength thresholds
                "premium_hands": ["AA", "KK", "QQ", "AKs"],  # Always raise
                "strong_hands": ["JJ", "TT", "99", "AQs", "AKo"],  # Raise or call raise
                "medium_hands": ["88", "77", "66", "AJs", "KQs", "AQo"],  # Call or raise in position
                "suited_connectors": ["T9s", "98s", "87s", "76s", "65s"],  # Play in position
                
                # Position-based rules
                "early_position": {
                    "raise_threshold": 0.8,  # Only raise with top 20% hands
                    "call_threshold": 0.9,   # Only call with top 10% hands
                },
                "middle_position": {
                    "raise_threshold": 0.7,  # Raise with top 30% hands
                    "call_threshold": 0.8,   # Call with top 20% hands
                },
                "late_position": {
                    "raise_threshold": 0.6,  # Raise with top 40% hands
                    "call_threshold": 0.7,   # Call with top 30% hands
                }
            },
            # Default postflop strategy
            "postflop": {
                # Hand strength categories
                "very_strong": 0.9,  # Top set, two pair+, etc.
                "strong": 0.7,       # Top pair good kicker, overpair, etc.
                "medium": 0.5,       # Middle pair, weak top pair, etc.
                "weak": 0.3,         # Bottom pair, draw, etc.
                "bluff": 0.1,        # Air, complete miss
                
                # Betting strategy based on hand strength
                "betting_strategy": {
                    "very_strong": {"fold": 0.0, "check": 0.1, "call": 0.2, "bet": 0.3, "raise": 0.4},
                    "strong": {"fold": 0.0, "check": 0.2, "call": 0.3, "bet": 0.4, "raise": 0.1},
                    "medium": {"fold": 0.1, "check": 0.4, "call": 0.3, "bet": 0.2, "raise": 0.0},
                    "weak": {"fold": 0.4, "check": 0.4, "call": 0.2, "bet": 0.0, "raise": 0.0},
                    "bluff": {"fold": 0.6, "check": 0.3, "call": 0.0, "bet": 0.1, "raise": 0.0}
                }
            }
        }
        # Combine default rules with provided rules, prioritizing provided rules
        for key, value in self.default_rules.items():
            if key not in self.rules:
                self.rules[key] = value
            elif isinstance(value, dict) and isinstance(self.rules[key], dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(self.rules[key], value)
    
    def _update_nested_dict(self, target_dict: Dict, source_dict: Dict) -> None:
        """Helper method to update nested dictionary structures."""
        for key, value in source_dict.items():
            if key not in target_dict:
                target_dict[key] = value
            elif isinstance(value, dict) and isinstance(target_dict[key], dict):
                self._update_nested_dict(target_dict[key], value)
    
    def _evaluate_hand_strength(self, hand: str, board: List[str] = None) -> float:
        """
        Evaluate the strength of a poker hand.
        
        Args:
            hand: String representation of hole cards (e.g., "AsKh")
            board: List of community cards (optional)
            
        Returns:
            Hand strength as a float between 0 and 1
        """
        # This is a placeholder that would be replaced by actual hand strength evaluation
        # In a real implementation, you would use the card_evaluator module
        if not board:
            # Preflop hand strength (simplistic for now)
            if hand in self.rules["preflop"]["premium_hands"]:
                return 0.95
            elif hand in self.rules["preflop"]["strong_hands"]:
                return 0.85
            elif hand in self.rules["preflop"]["medium_hands"]:
                return 0.75
            elif hand in self.rules["preflop"]["suited_connectors"]:
                return 0.65
            else:
                # Return a random value for other hands (should be replaced with actual preflop equity)
                return np.random.uniform(0.3, 0.6)
        else:
            # Postflop hand strength (would use card_evaluator in real implementation)
            # For now, we return a random strength based on the example board
            return np.random.uniform(0.1, 0.9)
    
    def _get_position_type(self, position: str) -> str:
        """
        Determine position type (early, middle, late).
        
        Args:
            position: Position at the table (e.g., 'BTN', 'SB', 'UTG')
            
        Returns:
            Position type as a string
        """
        if position in ['UTG', 'UTG+1', 'UTG+2']:
            return 'early_position'
        elif position in ['MP', 'MP+1', 'HJ']:
            return 'middle_position'
        elif position in ['CO', 'BTN']:
            return 'late_position'
        elif position in ['SB', 'BB']:
            # Special case for blinds, treat differently based on preflop/postflop
            return 'blinds'
        else:
            return 'middle_position'  # Default to middle position if unknown
    
    def _get_optimal_preflop_action(self, state: Dict[str, Any]) -> str:
        """
        Determine optimal preflop action based on hand and position.
        
        Args:
            state: Current game state
            
        Returns:
            Optimal action as a string
        """
        hand = state.get('hand', '')
        position = state.get('position', '')
        position_type = self._get_position_type(position)
        
        # Get hand strength
        hand_strength = self._evaluate_hand_strength(hand)
        
        # Check if it's a premium hand
        if hand in self.rules["preflop"]["premium_hands"]:
            # Always raise with premium hands
            if 'raise' in state.get('available_actions', []):
                return 'raise'
            elif 'call' in state.get('available_actions', []):
                return 'call'
        
        # Position-based decision
        if position_type in self.rules["preflop"]:
            position_rules = self.rules["preflop"][position_type]
            
            # If hand strength exceeds raise threshold, raise
            if hand_strength >= position_rules["raise_threshold"] and 'raise' in state.get('available_actions', []):
                return 'raise'
            # If hand strength exceeds call threshold, call
            elif hand_strength >= position_rules["call_threshold"] and 'call' in state.get('available_actions', []):
                return 'call'
        
        # Default action for weak hands: check if possible, otherwise fold
        if 'check' in state.get('available_actions', []):
            return 'check'
        else:
            return 'fold'
    
    def _get_optimal_postflop_action(self, state: Dict[str, Any]) -> str:
        """
        Determine optimal postflop action based on hand strength and board texture.
        
        Args:
            state: Current game state
            
        Returns:
            Optimal action as a string
        """
        hand = state.get('hand', '')
        board = state.get('board', [])
        pot_size = state.get('pot_size', 0)
        bet_to_call = state.get('bet_to_call', 0)
        
        # Get hand strength
        hand_strength = self._evaluate_hand_strength(hand, board)
        
        # Determine hand category based on strength
        if hand_strength >= self.rules["postflop"]["very_strong"]:
            category = "very_strong"
        elif hand_strength >= self.rules["postflop"]["strong"]:
            category = "strong"
        elif hand_strength >= self.rules["postflop"]["medium"]:
            category = "medium"
        elif hand_strength >= self.rules["postflop"]["weak"]:
            category = "weak"
        else:
            category = "bluff"
        
        # Get action probabilities for this hand category
        action_probs = self.rules["postflop"]["betting_strategy"][category]
        
        # Filter for available actions
        available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
        filtered_probs = {a: p for a, p in action_probs.items() if a in available_actions}
        
        # Normalize probabilities
        total_prob = sum(filtered_probs.values())
        if total_prob > 0:
            normalized_probs = {a: p/total_prob for a, p in filtered_probs.items()}
        else:
            # Default to equal probabilities if total is 0
            normalized_probs = {a: 1.0/len(filtered_probs) for a in filtered_probs}
        
        # Make stochastic decision based on probabilities
        actions, probs = zip(*normalized_probs.items())
        return np.random.choice(actions, p=probs)
    
    def predict_action(self, state: Dict[str, Any]) -> str:
        """
        Predict action based on rules and current game state.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Predicted action
        """
        street = state.get('street', 'preflop')
        
        if street == 'preflop':
            return self._get_optimal_preflop_action(state)
        else:
            return self._get_optimal_postflop_action(state)
    
    def update(self, state: Dict[str, Any], action: str, reward: float) -> None:
        """
        Update method (rule-based models don't learn).
        
        Args:
            state: Dictionary containing the game state
            action: Action taken
            reward: Reward received
        """
        # Rule-based models typically don't learn or update, so this is a no-op
        pass
    
    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get probability distribution over possible actions.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        street = state.get('street', 'preflop')
        hand = state.get('hand', '')
        board = state.get('board', []) if street != 'preflop' else None
        
        # Get hand strength
        hand_strength = self._evaluate_hand_strength(hand, board)
        
        if street == 'preflop':
            # Simplified preflop probabilities based on hand strength
            probs = {
                'fold': max(0, 1.0 - hand_strength - 0.3),
                'check': 0.2,
                'call': hand_strength * 0.4,
                'bet': hand_strength * 0.2,
                'raise': hand_strength * 0.6
            }
        else:
            # Determine hand category based on strength
            if hand_strength >= self.rules["postflop"]["very_strong"]:
                category = "very_strong"
            elif hand_strength >= self.rules["postflop"]["strong"]:
                category = "strong"
            elif hand_strength >= self.rules["postflop"]["medium"]:
                category = "medium"
            elif hand_strength >= self.rules["postflop"]["weak"]:
                category = "weak"
            else:
                category = "bluff"
            
            # Get probabilities from rules
            probs = self.rules["postflop"]["betting_strategy"][category]
            
        # Filter to available actions and normalize
        available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
        filtered_probs = {a: p for a, p in probs.items() if a in available_actions}
        
        # Normalize
        total = sum(filtered_probs.values())
        if total > 0:
            return {a: p/total for a, p in filtered_probs.items()}
        else:
            # Equal distribution if total is 0
            return {a: 1.0/len(available_actions) for a in available_actions}
