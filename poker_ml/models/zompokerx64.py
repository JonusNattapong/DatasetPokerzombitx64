"""
ZomPokerX64 - Advanced Poker AI Model
Created by Zombitx64

This model combines rule-based strategies with neural networks and Monte Carlo simulations
to create a powerful poker AI that can adapt to different playing styles.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import random
import logging
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

# Import safetensors
from safetensors.tensorflow import save_file, load_file

# Import base model
from .base_model import BasePokerModel

# Try to import ML libraries but provide graceful fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential, load_model, save_model
    from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Input
    from tensorflow.python.keras.callbacks import EarlyStopping
    from tensorflow.python.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ZomPokerX64(BasePokerModel):
    """
    ZomPokerX64 Poker AI Model created by Zombitx64.
    
    This advanced poker AI combines multiple strategies:
    1. Pre-calculated hand strength tables
    2. Neural network for decision making
    3. Opponent modeling
    4. Monte Carlo simulations for equity calculation
    5. Adaptive betting patterns
    """
    
    # Class constants
    VERSION = "1.0.0"
    AUTHOR = "Zombitx64"
    CREATION_DATE = "2023-11-02"
    
    # Pre-calculated hand ranks (simplified example - would be much larger in real implementation)
    PREFLOP_HAND_RANKS = {
        'AA': 1, 'KK': 2, 'QQ': 3, 'AKs': 4, 'JJ': 5, 'AQs': 6, 'KQs': 7, 'AJs': 8,
        'KJs': 9, 'TT': 10, 'AKo': 11, 'ATs': 12, 'QJs': 13, 'KTs': 14, 'QTs': 15,
        '99': 16, 'JTs': 17, 'AQo': 18, 'A9s': 19, 'KQo': 20
    }
    
    def __init__(self, name: str = "ZomPokerX64", config_path: Optional[str] = None,
                models_dir: str = "./models", learning_rate: float = 0.001,
                exploration_rate: float = 0.1, use_neural_network: bool = True):
        """
        Initialize the ZomPokerX64 model.
        
        Args:
            name: Name of the model
            config_path: Path to configuration file (JSON)
            models_dir: Directory to save/load models
            learning_rate: Learning rate for model updates
            exploration_rate: Exploration rate for action selection
            use_neural_network: Whether to use neural networks (if available)
        """
        super().__init__(name)
        
        # Basic parameters
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.models_dir = models_dir
        self.use_neural_network = use_neural_network and TENSORFLOW_AVAILABLE
        
        # Create model directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Game state tracking
        self.hand_history = []
        self.opponent_profiles = {}
        self.session_stats = {
            'hands_played': 0,
            'hands_won': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }
        
        # Load configuration if provided, otherwise use defaults
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize models
        self._init_models()
        
        logger.info(f"ZomPokerX64 v{self.VERSION} initialized by {self.AUTHOR}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Create default model configuration."""
        return {
            # Aggression parameters (0-1 scale)
            "aggression": 0.65,
            "bluff_frequency": 0.15,
            "cbet_frequency": 0.75,
            
            # Position weights (multipliers for position-based decisions)
            "position_weights": {
                "BTN": 1.3,  # Button - most aggressive
                "CO": 1.2,   # Cutoff
                "MP": 1.0,   # Middle position
                "UTG": 0.8,  # Under the gun - least aggressive
                "SB": 0.9,   # Small blind
                "BB": 1.0    # Big blind
            },
            
            # Stack depth adjustments
            "deep_stack_threshold": 100,  # BBs
            "short_stack_threshold": 20,  # BBs
            "short_stack_aggression": 1.2,
            "deep_stack_aggression": 0.9,
            
            # Neural network configurations
            "nn_config": {
                "action_model": {
                    "layers": [128, 64, 32],
                    "dropout": 0.2,
                    "activation": "relu"
                },
                "opponent_model": {
                    "layers": [64, 32],
                    "dropout": 0.1,
                    "activation": "relu"
                }
            },
            
            # Ranges
            "open_ranges": {
                "UTG": 0.1,   # Top 10% hands
                "MP": 0.15,   # Top 15% hands
                "CO": 0.25,   # Top 25% hands
                "BTN": 0.4,   # Top 40% hands
                "SB": 0.25,   # Top 25% hands
                "BB": 0.1     # Top 10% for BB open (only vs limpers)
            },
            
            # Meta-strategy parameters (learning)
            "adaptation_rate": 0.05,  # How quickly we adapt to opponent patterns
            "memory_length": 100,     # How many hands we remember for each opponent
            "exploration_decay": 0.999,  # Exploration rate decay per hand
            "checkpoint_interval": 100 # Save checkpoints every 100 hands
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return self._default_config()
    
    def _init_models(self) -> None:
        """Initialize the models used by ZomPokerX64."""
        # Decision model
        self.action_model = None
        self.opponent_model = None
        
        if self.use_neural_network:
            self._init_neural_networks()
        elif SKLEARN_AVAILABLE:
            self._init_sklearn_models()
        else:
            logger.warning("No ML libraries available. Using rule-based strategy only.")
    
    def _init_neural_networks(self) -> None:
        """Initialize neural network models if TensorFlow is available."""
        # Neural network for action prediction
        config = self.config["nn_config"]["action_model"]
        
        action_model = Sequential()
        action_model.add(Input(shape=(20,)))  # Input features: game state, hand features, etc.
        
        # Add hidden layers
        for units in config["layers"]:
            action_model.add(Dense(units, activation=config["activation"]))
            action_model.add(Dropout(config["dropout"]))
        
        # Output layer: 5 actions (fold, check, call, bet, raise)
        action_model.add(Dense(5, activation='softmax'))
        
        action_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.action_model = action_model
        logger.info("Neural network action model initialized")
        
        # Neural network for opponent modeling
        config = self.config["nn_config"]["opponent_model"]
        
        opponent_model = Sequential()
        opponent_model.add(Input(shape=(15,)))  # Input features: opponent actions, positions, etc.
        
        # Add hidden layers
        for units in config["layers"]:
            opponent_model.add(Dense(units, activation=config["activation"]))
            opponent_model.add(Dropout(config["dropout"]))
        
        # Output layer: opponent playing style parameters
        opponent_model.add(Dense(4, activation='sigmoid'))  # [VPIP, PFR, Aggression, Bluff frequency]
        
        opponent_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.opponent_model = opponent_model
        logger.info("Neural network opponent model initialized")
    
    def _init_sklearn_models(self) -> None:
        """Initialize scikit-learn models if available."""
        # Random forest for action prediction
        self.action_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Random forest for opponent modeling
        self.opponent_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        logger.info("scikit-learn models initialized")
    
    def _encode_hand(self, hand: str) -> List[float]:
        """
        Encode a poker hand for model input.
        
        Args:
            hand: String representation of hole cards (e.g., "AsKh")
            
        Returns:
            Encoded hand as a list of features
        """
        if not hand or len(hand) < 4:
            # Default encoding for unknown hand
            return [0.0] * 8
        
        # Extract ranks and suits
        rank1, suit1 = hand[0], hand[1]
        rank2, suit2 = hand[2], hand[3]
        
        # Convert ranks to values
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        rank1_value = ranks.get(rank1, 0)
        rank2_value = ranks.get(rank2, 0)
        
        # Ensure higher rank is first
        if rank1_value < rank2_value:
            rank1_value, rank2_value = rank2_value, rank1_value
            suit1, suit2 = suit2, suit1
        
        # Create features
        features = [
            rank1_value / 14.0,  # Normalize to 0-1
            rank2_value / 14.0,
            1.0 if suit1 == suit2 else 0.0,  # Suited indicator
            1.0 if rank1_value == rank2_value else 0.0,  # Pair indicator
            (rank1_value - rank2_value) / 13.0,  # Rank difference
            ord(suit1) / 115.0,  # Normalize suit (ASCII value)
            ord(suit2) / 115.0,
            (rank1_value + rank2_value) / 28.0  # Average rank
        ]
        return features
    
    def _encode_game_state(self, state: Dict[str, Any]) -> List[float]:
        """
        Encode the game state for model input.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Encoded game state as a list of features
        """
        # Extract relevant information from the game state
        pot_size = state.get('pot_size', 0)
        bet_to_call = state.get('bet_to_call', 0)
        stack_size = state.get('stack_size', 100)
        position = state.get('position', 'UTG')
        street = state.get('street', 'preflop')
        
        # Encode position
        position_encoding = {
            'UTG': [1, 0, 0, 0, 0],
            'MP': [0, 1, 0, 0, 0],
            'CO': [0, 0, 1, 0, 0],
            'BTN': [0, 0, 0, 1, 0],
            'BB': [0, 0, 0, 0, 1]
        }.get(position, [0, 0, 0, 0, 0])
        
        # Encode street
        street_encoding = {
            'preflop': [1, 0, 0, 0],
            'flop': [0, 1, 0, 0],
            'turn': [0, 0, 1, 0],
            'river': [0, 0, 0, 1]
        }.get(street, [1, 0, 0, 0])
        
        # Create features
        features = [
            pot_size / 100.0,  # Normalize pot size
            bet_to_call / 100.0,  # Normalize bet to call
            stack_size / 100.0,  # Normalize stack size
            self.config["aggression"],  # Model aggression
            self.config["bluff_frequency"]  # Model bluff frequency
        ] + position_encoding + street_encoding
        
        return features
    
    def _prepare_model_input(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Prepare the input for the neural network model.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Prepared input as a NumPy array
        """
        # Encode hand and game state
        hand_features = self._encode_hand(state.get('hand', ''))
        game_state_features = self._encode_game_state(state)
        
        # Combine features
        combined_features = hand_features + game_state_features
        
        # Convert to NumPy array
        return np.array([combined_features])
    
    def predict_action(self, state: Dict[str, Any]) -> str:
        """
        Predict the next action given the current game state.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Predicted action (e.g., 'fold', 'call', 'check', 'bet', 'raise')
        """
        # Prepare model input
        model_input = self._prepare_model_input(state)
        
        # Get available actions
        available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
        
        # Predict action using neural network or rule-based strategy
        if self.use_neural_network and self.action_model:
            # Predict action probabilities
            action_probs = self.action_model.predict(model_input)[0]
            
            # Map probabilities to actions
            action_map = {
                0: 'fold',
                1: 'check',
                2: 'call',
                3: 'bet',
                4: 'raise'
            }
            
            # Filter probabilities for available actions
            filtered_probs = {action_map[i]: prob for i, prob in enumerate(action_probs) if action_map[i] in available_actions}
            
            # Normalize probabilities
            total_prob = sum(filtered_probs.values())
            if total_prob > 0:
                normalized_probs = {a: p/total_prob for a, p in filtered_probs.items()}
            else:
                # Default to equal probabilities if total is 0
                normalized_probs = {a: 1.0/len(available_actions) for a in available_actions}
            
            # Exploration vs exploitation
            if random.random() < self.exploration_rate:
                # Explore: choose a random action
                return random.choice(available_actions)
            else:
                # Exploit: choose the action with the highest probability
                return max(normalized_probs, key=normalized_probs.get)
        else:
            # Use rule-based strategy if neural network is not available
            # (This would be a separate RuleBasedModel instance in a real implementation)
            return random.choice(available_actions)
    
    def update(self, state: Dict[str, Any], action: str, reward: float) -> None:
        """
        Update the model based on state, action, and reward.
        
        Args:
            state: Dictionary containing the game state
            action: Action taken
            reward: Reward received
        """
        # Prepare model input
        model_input = self._prepare_model_input(state)
        
        # Update action model (if using neural network)
        if self.use_neural_network and self.action_model:
            # Create target vector (one-hot encode the action)
            action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4}
            action_index = action_map.get(action, 1)  # Default to 'check' if unknown
            target = np.zeros((1, 5))
            target[0, action_index] = 1.0
            
            # Train the model (simplified - would use more sophisticated training in real implementation)
            self.action_model.fit(model_input, target, epochs=1, verbose=0)
        
        # Update opponent model (if available)
        # (This would involve tracking opponent actions and updating the opponent model)
        
        # Update exploration rate
        self.exploration_rate *= self.config["exploration_decay"]
        self.exploration_rate = max(0.01, self.exploration_rate)  # Ensure minimum exploration
        
        # Update session statistics
        self.session_stats['hands_played'] += 1
        if reward > 0:
            self.session_stats['hands_won'] += 1
            self.session_stats['total_profit'] += reward
        else:
            self.session_stats['total_loss'] += abs(reward)
        
        # Store hand history
        self.hand_history.append({
            'state': state,
            'action': action,
            'reward': reward
        })
        
        # Limit hand history length
        if len(self.hand_history) > self.config["memory_length"]:
            self.hand_history.pop(0)

        # Checkpointing
        if self.config["checkpoint_interval"] > 0 and self.session_stats['hands_played'] % self.config["checkpoint_interval"] == 0:
            self._save_checkpoint()
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the model."""
        checkpoint_dir = os.path.join(self.models_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.session_stats['hands_played']}")
        self.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get probability distribution over possible actions.
        
        Args:
            state: Dictionary containing the current game state
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        # Prepare model input
        model_input = self._prepare_model_input(state)
        
        # Predict action probabilities using neural network or rule-based strategy
        if self.use_neural_network and self.action_model:
            # Predict action probabilities
            action_probs = self.action_model.predict(model_input)[0]
            
            # Map probabilities to actions
            action_map = {
                0: 'fold',
                1: 'check',
                2: 'call',
                3: 'bet',
                4: 'raise'
            }
            
            # Get available actions
            available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
            
            # Filter probabilities for available actions
            filtered_probs = {action_map[i]: prob for i, prob in enumerate(action_probs) if action_map[i] in available_actions}
            
            # Normalize probabilities
            total_prob = sum(filtered_probs.values())
            if total_prob > 0:
                normalized_probs = {a: p/total_prob for a, p in filtered_probs.items()}
            else:
                # Default to equal probabilities if total is 0
                normalized_probs = {a: 1.0/len(available_actions) for a in available_actions}
            
            return normalized_probs
        else:
            # Use rule-based strategy if neural network is not available
            # (This would be a separate RuleBasedModel instance in a real implementation)
            available_actions = state.get('available_actions', ['fold', 'check', 'call', 'bet', 'raise'])
            prob = 1.0 / len(available_actions)
            return {action: prob for action in available_actions}
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save session statistics
        stats_path = os.path.join(path, "session_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.session_stats, f, indent=4)
        
        # Save hand history (optional)
        history_path = os.path.join(path, "hand_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(self.hand_history, f)
        
        # Save models
        if self.use_neural_network and self.action_model:
            action_model_path = os.path.join(path, "action_model.safetensors")
            save_file(self.action_model, action_model_path)
            logger.info(f"Action model saved to {action_model_path}")

        if self.use_neural_network and self.opponent_model:
            opponent_model_path = os.path.join(path, "opponent_model.safetensors")
            save_file(self.opponent_model, opponent_model_path)
            logger.info(f"Opponent model saved to {opponent_model_path}")

        logger.info(f"ZomPokerX64 model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to the saved model
        """
        try:
            # Load configuration
            config_path = os.path.join(path, "config.json")
            self.config = self._load_config(config_path)

            # Load session statistics
            stats_path = os.path.join(path, "session_stats.json")
            with open(stats_path, 'r') as f:
                self.session_stats = json.load(f)

            # Load hand history (optional)
            history_path = os.path.join(path, "hand_history.pkl")
            with open(history_path, 'rb') as f:
                self.hand_history = pickle.load(f)

            # Load models
            if self.use_neural_network:
                action_model_path = os.path.join(path, "action_model.safetensors")
                if os.path.exists(action_model_path):
                    self.action_model = load_file(action_model_path)
                    logger.info(f"Loaded neural network action model from {action_model_path}")

                opponent_model_path = os.path.join(path, "opponent_model.safetensors")
                if os.path.exists(opponent_model_path):
                    self.opponent_model = load_file(opponent_model_path)
                    logger.info(f"Loaded neural network opponent model from {opponent_model_path}")

            logger.info(f"ZomPokerX64 model loaded from {path}")

            # Load from checkpoint if available
            checkpoint_dir = os.path.join(self.models_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda d: int(d.split('_')[-1]))
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    logger.info(f"Loading from checkpoint: {checkpoint_path}")
                    
                    # Load config, stats, and history from main model dir, then overwrite with checkpoint
                    config_path = os.path.join(path, "config.json")
                    self.config = self._load_config(config_path)
                    stats_path = os.path.join(path, "session_stats.json")
                    with open(stats_path, 'r') as f:
                        self.session_stats = json.load(f)
                    history_path = os.path.join(path, "hand_history.pkl")
                    if os.path.exists(history_path):
                      with open(history_path, 'rb') as f:
                          self.hand_history = pickle.load(f)

                    # Load checkpoint specific data
                    checkpoint_config_path = os.path.join(checkpoint_path, "config.json")
                    checkpoint_config = self._load_config(checkpoint_config_path)
                    # self.config.update(checkpoint_config) # Don't overwrite main config

                    checkpoint_stats_path = os.path.join(checkpoint_path, "session_stats.json")
                    with open(checkpoint_stats_path, 'r') as f:
                        checkpoint_stats = json.load(f)
                    self.session_stats.update(checkpoint_stats)

                    checkpoint_history_path = os.path.join(checkpoint_path, "hand_history.pkl")
                    if os.path.exists(checkpoint_history_path):
                        with open(checkpoint_history_path, 'rb') as f:
                            checkpoint_history = pickle.load(f)
                        self.hand_history = checkpoint_history # Overwrite main history

                    # Load models
                    if self.use_neural_network:
                        action_model_path = os.path.join(checkpoint_path, "action_model.safetensors")
                        if os.path.exists(action_model_path):
                            self.action_model = load_file(action_model_path)
                            logger.info(f"Loaded neural network action model from {action_model_path}")

                        opponent_model_path = os.path.join(checkpoint_path, "opponent_model.safetensors")
                        if os.path.exists(opponent_model_path):
                            self.opponent_model = load_file(opponent_model_path)
                            logger.info(f"Loaded neural network opponent model from {opponent_model_path}")

        except Exception as e:
            logger.error(f"Failed to load ZomPokerX64 model from {path}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate win rate
        if self.session_stats['hands_played'] > 0:
            win_rate = self.session_stats['total_profit'] / self.session_stats['hands_played']
        else:
            win_rate = 0.0
        
        # Calculate win/loss ratio
        if self.session_stats['total_loss'] > 0:
            win_loss_ratio = self.session_stats['total_profit'] / self.session_stats['total_loss']
        else:
            win_loss_ratio = float('inf') if self.session_stats['total_profit'] > 0 else 0.0
        
        return {
            'hands_played': self.session_stats['hands_played'],
            'hands_won': self.session_stats['hands_won'],
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'exploration_rate': self.exploration_rate
        }
