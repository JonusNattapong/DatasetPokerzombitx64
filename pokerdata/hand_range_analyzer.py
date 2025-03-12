"""
Hand range analyzer module for evaluating poker hand ranges.
"""

import pandas as pd
import numpy as np
import re
import itertools
from typing import Dict, List, Tuple, Set, Optional, Union
from .card_evaluator import Card, Hand, analyze_hand, compare_hands


class HandRange:
    """Class for representing and evaluating poker hand ranges."""
    
    # All possible pocket pairs
    PAIRS = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
    
    # Define ranks here as a class attribute so it's available everywhere
    RANKS = 'AKQJT98765432'
    
    # All possible suited and offsuit hands
    SUITED_HANDS = []
    OFFSUIT_HANDS = []
    
    # Initialize all possible hands
    for i, r1 in enumerate(RANKS):
        for r2 in RANKS[i+1:]:
            SUITED_HANDS.append(f"{r1}{r2}s")
            OFFSUIT_HANDS.append(f"{r1}{r2}o")
    
    # All possible hands
    ALL_HANDS = PAIRS + SUITED_HANDS + OFFSUIT_HANDS
    
    def __init__(self, range_str: str = ''):
        """
        Initialize a hand range from a string representation.
        
        Args:
            range_str: String representation of a hand range (e.g., 'AA,KK,AKs')
        """
        self.hands = set()
        if range_str:
            self.add_range(range_str)
    
    def add_range(self, range_str: str):
        """
        Add hands to the range.
        
        Args:
            range_str: String representation of a hand range (e.g., 'AA,KK,AKs')
        """
        # Split the range string by commas
        parts = [part.strip() for part in range_str.split(',')]
        
        for part in parts:
            if not part:
                continue
                
            # Check if it's a range like "ATs+"
            if '+' in part:
                self._add_plus_range(part.rstrip('+'))
            # Check if it's a range like "JTs-87s"
            elif '-' in part:
                start, end = part.split('-')
                self._add_range_between(start, end)
            # Single hand
            else:
                if part in self.ALL_HANDS:
                    self.hands.add(part)
    
    def _add_plus_range(self, base_hand: str):
        """
        Add a 'plus' range to the hand range.
        
        Args:
            base_hand: Base hand for the plus range (e.g., 'ATs')
        """
        # Handle pairs
        if len(base_hand) == 2 and base_hand[0] == base_hand[1]:
            rank_index = self.PAIRS.index(base_hand)
            self.hands.update(self.PAIRS[:rank_index+1])
            return
        
        # Handle suited hands
        if base_hand.endswith('s'):
            base_without_s = base_hand[:-1]
            high_rank = base_without_s[0]
            low_rank = base_without_s[1]
            
            for hand in self.SUITED_HANDS:
                hand_without_s = hand[:-1]
                if (hand_without_s[0] == high_rank and 
                    self.RANKS.index(hand_without_s[1]) <= self.RANKS.index(low_rank)):
                    self.hands.add(hand)
        
        # Handle offsuit hands
        elif base_hand.endswith('o'):
            base_without_o = base_hand[:-1]
            high_rank = base_without_o[0]
            low_rank = base_without_o[1]
            
            for hand in self.OFFSUIT_HANDS:
                hand_without_o = hand[:-1]
                if (hand_without_o[0] == high_rank and 
                    self.RANKS.index(hand_without_o[1]) <= self.RANKS.index(low_rank)):
                    self.hands.add(hand)
    
    def _add_range_between(self, start_hand: str, end_hand: str):
        """
        Add a range of hands between start_hand and end_hand.
        
        Args:
            start_hand: Starting hand of the range
            end_hand: Ending hand of the range
        """
        # Check if both hands are of the same type
        if (start_hand in self.PAIRS and end_hand in self.PAIRS):
            start_idx = self.PAIRS.index(start_hand)
            end_idx = self.PAIRS.index(end_hand)
            self.hands.update(self.PAIRS[end_idx:start_idx+1])
        
        elif (start_hand in self.SUITED_HANDS and end_hand in self.SUITED_HANDS):
            start_idx = self.SUITED_HANDS.index(start_hand)
            end_idx = self.SUITED_HANDS.index(end_hand)
            self.hands.update(self.SUITED_HANDS[start_idx:end_idx+1])
        
        elif (start_hand in self.OFFSUIT_HANDS and end_hand in self.OFFSUIT_HANDS):
            start_idx = self.OFFSUIT_HANDS.index(start_hand)
            end_idx = self.OFFSUIT_HANDS.index(end_hand)
            self.hands.update(self.OFFSUIT_HANDS[start_idx:end_idx+1])
    
    def expand_to_all_combos(self) -> List[str]:
        """
        Expand the hand range to all possible specific card combinations.
        
        Returns:
            List of all specific card combinations (e.g., 'AsAh', 'AsAd', ...)
        """
        suits = 'shdc'  # spades, hearts, diamonds, clubs
        all_combos = []
        
        for hand in self.hands:
            if len(hand) == 2:  # Pair
                rank = hand[0]
                # Generate all 6 possible combos for the pair
                pair_combos = []
                for i, s1 in enumerate(suits):
                    for s2 in suits[i+1:]:
                        pair_combos.append(f"{rank}{s1}{rank}{s2}")
                all_combos.extend(pair_combos)
            
            elif hand.endswith('s'):  # Suited
                rank1 = hand[0]
                rank2 = hand[1]
                # Generate all 4 possible suited combos
                for suit in suits:
                    all_combos.append(f"{rank1}{suit}{rank2}{suit}")
            
            elif hand.endswith('o'):  # Offsuit
                rank1 = hand[0]
                rank2 = hand[1]
                # Generate all 12 possible offsuit combos
                for s1 in suits:
                    for s2 in suits:
                        if s1 != s2:
                            all_combos.append(f"{rank1}{s1}{rank2}{s2}")
        
        return all_combos
    
    def get_equity_vs_range(self, opponent_range, board: str = '') -> float:
        """
        Calculate equity of this range against an opponent's range on a given board.
        
        Args:
            opponent_range: Opponent's hand range
            board: Board cards (e.g., 'As Ks Qs')
            
        Returns:
            Equity as a percentage (0-100)
        """
        if isinstance(opponent_range, str):
            opponent_range = HandRange(opponent_range)
        
        # Expand both ranges to specific card combinations
        our_combos = self.expand_to_all_combos()
        opponent_combos = opponent_range.expand_to_all_combos()
        
        # Remove combos that share cards with the board
        if board:
            board_cards = set(board.split())
            our_combos = [combo for combo in our_combos if not self._shares_cards(combo, board_cards)]
            opponent_combos = [combo for combo in opponent_combos if not self._shares_cards(combo, board_cards)]
        
        total_matchups = 0
        wins = 0
        ties = 0
        
        # Brute force comparison of all possible matchups
        # (Note: this is computationally expensive and could be optimized)
        for our_combo in our_combos:
            our_combo_formatted = f"{our_combo[0:2]} {our_combo[2:4]}"
            for opp_combo in opponent_combos:
                # Skip if the hands share any cards
                if self._shares_cards(our_combo, set(opp_combo)):
                    continue
                
                opp_combo_formatted = f"{opp_combo[0:2]} {opp_combo[2:4]}"
                result = compare_hands(our_combo_formatted, opp_combo_formatted, board)
                
                total_matchups += 1
                if result > 0:
                    wins += 1
                elif result == 0:
                    ties += 0.5
        
        if total_matchups == 0:
            return 0
        
        # Equity = (wins + ties) / total_matchups
        return 100 * (wins + ties) / total_matchups
    
    def _shares_cards(self, combo: str, other_cards) -> bool:
        """
        Check if a combo shares any cards with other cards.
        
        Args:
            combo: Card combination (e.g., 'AsAh')
            other_cards: Set of other cards to check against
            
        Returns:
            True if the combo shares any cards with other_cards
        """
        combo_cards = set([combo[0:2], combo[2:4]])
        for card in other_cards:
            if card in combo_cards:
                return True
        return False
    
    def __len__(self) -> int:
        """Return the number of hands in the range."""
        return len(self.hands)
    
    def __str__(self) -> str:
        """Return a string representation of the hand range."""
        return ','.join(sorted(list(self.hands)))


def analyze_player_range(df: pd.DataFrame, player: str, position: Optional[str] = None,
                        action_type: Optional[str] = None) -> Dict[str, float]:
    """
    Analyze a player's hand range based on their historical actions.
    
    Args:
        df: DataFrame containing poker hand history
        player: Player name to analyze
        position: Optional position to filter by (e.g., 'BTN', 'SB', 'BB')
        action_type: Optional action type to filter by (e.g., 'raises', 'calls', 'folds')
        
    Returns:
        Dictionary with hand range statistics
    """
    # Filter data for the player
    player_df = df[df['name'] == player].copy()
    
    # Further filter by position if specified
    if position:
        player_df = player_df[player_df['position'] == position]
    
    # Filter by action type if specified
    if action_type:
        action_col = 'action_pre'  # Default to preflop actions
        player_df = player_df[player_df[action_col].str.contains(action_type, case=False, na=False)]
    
    # Count hands where we know the cards
    known_hands = player_df[player_df['cards'] != '--'].copy()
    
    # If we don't have enough data, return limited information
    if len(known_hands) < 10:
        return {
            'player': player,
            'position': position if position else 'all',
            'action': action_type if action_type else 'all',
            'total_hands': len(player_df),
            'known_hands': len(known_hands),
            'message': 'Not enough data with known cards'
        }
    
    # Process the known hands to standardize format
    def format_hand(cards_str):
        cards = cards_str.split()
        if len(cards) != 2:
            return None
        
        rank1 = cards[0][0]
        suit1 = cards[0][1]
        rank2 = cards[1][0]
        suit2 = cards[1][1]
        
        # Convert to standard notation (e.g., 'AKs', 'AKo', 'AA')
        if rank1 == rank2:
            return rank1 + rank2  # Pair
        elif suit1 == suit2:
            return rank1 + rank2 + 's'  # Suited
        else:
            return rank1 + rank2 + 'o'  # Offsuit
    
    known_hands['standard_hand'] = known_hands['cards'].apply(format_hand)
    known_hands = known_hands[known_hands['standard_hand'].notna()]
    
    # Calculate frequencies
    hand_counts = known_hands['standard_hand'].value_counts()
    total_known_hands = len(known_hands)
    hand_frequencies = {hand: count/total_known_hands for hand, count in hand_counts.items()}
    
    # Group hands into categories
    categories = {
        'pairs': [h for h in hand_frequencies if len(h) == 2],
        'suited': [h for h in hand_frequencies if h.endswith('s')],
        'offsuit': [h for h in hand_frequencies if h.endswith('o')]
    }
    
    category_frequencies = {
        'pairs': sum(hand_frequencies[h] for h in categories['pairs']),
        'suited': sum(hand_frequencies[h] for h in categories['suited']),
        'offsuit': sum(hand_frequencies[h] for h in categories['offsuit'])
    }
    
    # Calculate VPIP (Voluntarily Put Money In Pot)
    vpip_actions = ['calls', 'raises', 'bets']
    vpip_hands = player_df[player_df['action_pre'].apply(
        lambda x: any(action in x for action in vpip_actions) if isinstance(x, str) else False
    )]
    vpip = len(vpip_hands) / len(player_df) if len(player_df) > 0 else 0
    
    # Calculate PFR (Preflop Raise)
    pfr_hands = player_df[player_df['action_pre'].str.contains('raises', case=False, na=False)]
    pfr = len(pfr_hands) / len(player_df) if len(player_df) > 0 else 0
    
    # Calculate 3-bet frequency
    threeb_hands = player_df[player_df['action_pre'].str.contains('raises.*raises', case=False, na=False)]
    threeb = len(threeb_hands) / len(vpip_hands) if len(vpip_hands) > 0 else 0
    
    return {
        'player': player,
        'position': position if position else 'all',
        'action': action_type if action_type else 'all',
        'total_hands': len(player_df),
        'known_hands': len(known_hands),
        'vpip': vpip * 100,
        'pfr': pfr * 100,
        'threeb': threeb * 100,
        'hand_frequencies': hand_frequencies,
        'category_frequencies': category_frequencies,
        'top_hands': hand_counts[:10].to_dict()
    }
