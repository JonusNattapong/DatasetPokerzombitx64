# poker_ml/features/card_encoder.py
import numpy as np
from typing import List, Dict, Tuple, Union

class CardEncoder:
    """Utility class for encoding poker cards and hands"""
    
    def __init__(self):
        """Initialize the card encoder"""
        self.rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        self.suit_values = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        
    def encode_card(self, card: str) -> Tuple[int, int]:
        """
        Encode a single card as rank and suit values
        
        Args:
            card: Card string in format 'Rs' where R is rank and s is suit
            
        Returns:
            Tuple of (rank_value, suit_value)
        """
        if not card or len(card) < 2:
            return (0, 0)
            
        rank = card[0].upper()
        suit = card[1].lower()
        
        rank_value = self.rank_values.get(rank, 0)
        suit_value = self.suit_values.get(suit, 0)
        
        return (rank_value, suit_value)
    
    def encode_hand(self, hand: str) -> np.ndarray:
        """
        Encode a complete hand (2 or more cards)
        
        Args:
            hand: Space-separated string of cards (e.g., 'As Kh')
            
        Returns:
            Array of encoded card values, shape (n_cards, 2)
        """
        if not hand or not isinstance(hand, str):
            return np.zeros((2, 2))
            
        cards = hand.split()
        encoded = [self.encode_card(card) for card in cards]
        
        # Pad or truncate to ensure consistent shape
        while len(encoded) < 2:
            encoded.append((0, 0))
            
        return np.array(encoded[:5])  # Limit to 5 cards max (for board)
    
    def calculate_hand_strength(self, hole_cards: str, board_cards: str = "") -> float:
        """
        Calculate a simplified hand strength score
        
        Args:
            hole_cards: Player's hole cards (e.g., 'As Kh')
            board_cards: Community cards (e.g., 'Td 7c 3s')
            
        Returns:
            Hand strength score between 0 and 1
        """
        if not hole_cards:
            return 0.0
            
        # Encode hole cards
        hole_encoded = self.encode_hand(hole_cards)
        hole_ranks = hole_encoded[:, 0]
        hole_suits = hole_encoded[:, 1]
        
        # Basic features that correlate with hand strength
        score = 0.0
        
        # High cards are stronger
        rank_score = np.sum(hole_ranks) / 28.0  # Normalize by max possible (14+14)
        score += 0.5 * rank_score
        
        # Pairs are strong
        if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > 0:
            pair_value = hole_ranks[0] / 14.0  # Normalize by max rank
            score += 0.3 * pair_value
        
        # Suited cards are stronger
        if hole_suits[0] == hole_suits[1] and hole_suits[0] > 0:
            score += 0.1
        
        # Connected cards are stronger (straight potential)
        if abs(hole_ranks[0] - hole_ranks[1]) == 1:
            score += 0.1
        elif abs(hole_ranks[0] - hole_ranks[1]) == 2:
            score += 0.05
            
        # If we have board cards, do more detailed evaluation
        if board_cards:
            board_encoded = self.encode_hand(board_cards)
            board_ranks = board_encoded[:, 0]
            board_suits = board_encoded[:, 1]
            
            # Count how many board cards match hole card ranks
            matches = np.sum(np.isin(board_ranks, hole_ranks))
            if matches > 0:
                score += 0.1 * matches
                
            # More detailed analysis could be added here
        
        # Ensure the score is between 0 and 1
        return min(max(score, 0.0), 1.0)