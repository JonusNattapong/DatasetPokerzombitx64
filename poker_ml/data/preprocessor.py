# poker_ml/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re

class PokerDataPreprocessor:
    """Utility class for preprocessing poker hand data"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.action_mapping = {
            'folds': 'fold',
            'checks': 'check',
            'calls': 'call',
            'bets': 'bet',
            'raises': 'raise',
            'all-in': 'all-in'
        }
        
        self.position_order = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']
        
    def extract_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and normalize poker actions
        
        Args:
            df: DataFrame containing hand data
            
        Returns:
            DataFrame with processed action columns
        """
        # Copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract the primary action from each action string
        for col in ['action_pre', 'action_flop', 'action_turn', 'action_river']:
            if col in result_df.columns:
                result_df[f'{col}_normalized'] = result_df[col].apply(
                    lambda x: self._normalize_action(x) if isinstance(x, str) else np.nan
                )
        
        # Create a target column with the action taken in each decision point
        result_df['action'] = result_df.apply(self._determine_primary_action, axis=1)
        
        return result_df
    
    def _normalize_action(self, action_str: str) -> str:
        """
        Convert action string to normalized form
        
        Args:
            action_str: String describing poker action
            
        Returns:
            Normalized action string
        """
        if pd.isna(action_str) or not isinstance(action_str, str):
            return np.nan
            
        # Find the first matching action
        for key, value in self.action_mapping.items():
            if key in action_str.lower():
                return value
                
        return 'unknown'
    
    def _determine_primary_action(self, row: pd.Series) -> str:
        """
        Determine the primary action for a hand
        
        Args:
            row: DataFrame row
            
        Returns:
            Primary action as string
        """
        # Check action columns in order of betting rounds
        for col in ['action_pre_normalized', 'action_flop_normalized', 
                    'action_turn_normalized', 'action_river_normalized']:
            if col in row and pd.notna(row[col]):
                action = row[col]
                # Group bet/raise together as they're both aggressive actions
                if action in ['bet', 'raise', 'all-in']:
                    return 'raise'
                elif action in ['fold', 'check', 'call']:
                    return action
                    
        return 'unknown'
    
    def encode_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode player positions with one-hot encoding
        
        Args:
            df: DataFrame containing hand data
            
        Returns:
            DataFrame with encoded position columns
        """
        result_df = df.copy()
        
        # Standardize position naming
        if 'position' in result_df.columns:
            result_df['position_std'] = result_df['position'].apply(
                lambda x: self._standardize_position(x) if isinstance(x, str) else np.nan
            )
            
            # One-hot encode the standardized positions
            position_dummies = pd.get_dummies(
                result_df['position_std'], prefix='pos', dummy_na=False
            )
            
            # Join the one-hot encoded columns
            result_df = pd.concat([result_df, position_dummies], axis=1)
            
        return result_df
    
    def _standardize_position(self, position: str) -> str:
        """
        Standardize position names
        
        Args:
            position: Position string
            
        Returns:
            Standardized position string
        """
        position = position.upper()
        
        # Map common variations
        position_map = {
            'BUTTON': 'BTN',
            'DEALER': 'BTN',
            'CUTOFF': 'CO',
            'SMALL BLIND': 'SB',
            'BIG BLIND': 'BB',
            'EARLY POSITION': 'UTG',
            'MIDDLE POSITION': 'MP',
            'LATE POSITION': 'CO'
        }
        
        for key, value in position_map.items():
            if key in position:
                return value
                
        for std_pos in self.position_order:
            if std_pos in position:
                return std_pos
                
        return position
    
    def create_hand_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to hand strength
        
        Args:
            df: DataFrame containing hand data
            
        Returns:
            DataFrame with hand strength features
        """
        result_df = df.copy()
        
        # Add features for known card combinations if cards are available
        if 'cards' in result_df.columns:
            result_df['has_pair'] = result_df['cards'].apply(
                lambda x: self._has_pair(x) if isinstance(x, str) else False
            )
            
            result_df['has_broadways'] = result_df['cards'].apply(
                lambda x: self._has_broadways(x) if isinstance(x, str) else False
            )
            
            result_df['has_suited'] = result_df['cards'].apply(
                lambda x: self._is_suited(x) if isinstance(x, str) else False
            )
            
            result_df['card_ranks'] = result_df['cards'].apply(
                lambda x: self._get_card_ranks(x) if isinstance(x, str) else []
            )
            
            result_df['card_rank_max'] = result_df['card_ranks'].apply(
                lambda x: max(x) if len(x) > 0 else np.nan
            )
            
            result_df['card_rank_min'] = result_df['card_ranks'].apply(
                lambda x: min(x) if len(x) > 0 else np.nan
            )
            
            result_df['card_rank_diff'] = result_df['card_rank_max'] - result_df['card_rank_min']
            
        return result_df
    
    def _has_pair(self, cards_str: str) -> bool:
        """
        Check if the cards form a pair
        
        Args:
            cards_str: String representation of cards
            
        Returns:
            True if cards form a pair, False otherwise
        """
        # Extract card ranks (first character of each card)
        ranks = [c[0] for c in cards_str.split()]
        return len(ranks) >= 2 and ranks[0] == ranks[1]
    
    def _has_broadways(self, cards_str: str) -> bool:
        """
        Check if the cards contain broadways
        
        Args:
            cards_str: String representation of cards
            
        Returns:
            True if cards contain broadway cards, False otherwise
        """
        broadway_ranks = ['T', 'J', 'Q', 'K', 'A']
        ranks = [c[0] for c in cards_str.split()]
        return any(r in broadway_ranks for r in ranks)
    
    def _is_suited(self, cards_str: str) -> bool:
        """
        Check if the cards are suited
        
        Args:
            cards_str: String representation of cards
            
        Returns:
            True if cards are suited, False otherwise
        """
        suits = [c[1] for c in cards_str.split()]
        return len(suits) >= 2 and suits[0] == suits[1]
    
    def _get_card_ranks(self, cards_str: str) -> List[int]:
        """
        Convert card ranks to numeric values
        
        Args:
            cards_str: String representation of cards
            
        Returns:
            List of numeric card ranks
        """
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        ranks = []
        for card in cards_str.split():
            if card[0] in rank_map:
                ranks.append(rank_map[card[0]])
                
        return ranks
    def create_pot_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Create features related to pot odds and betting
            
            Args:
                df: DataFrame containing hand data
                
            Returns:
                DataFrame with pot-related features
            """
            result_df = df.copy()
            
            # Calculate pot odds when applicable
            pot_cols = [c for c in df.columns if 'pot_' in c]
            bet_cols = [c for c in df.columns if 'bet_' in c]
            
            if 'pot_pre' in pot_cols and 'bet_pre' in bet_cols:
                # Calculate pot odds as the ratio of call amount to pot size
                result_df['pot_odds_pre'] = result_df.apply(
                    lambda row: row['bet_pre'] / row['pot_pre'] if row['pot_pre'] > 0 else np.nan,
                    axis=1
                )
                
            if 'pot_flop' in pot_cols and 'bet_flop' in bet_cols:
                result_df['pot_odds_flop'] = result_df.apply(
                    lambda row: row['bet_flop'] / row['pot_flop'] if row['pot_flop'] > 0 else np.nan,
                    axis=1
                )
                
            if 'pot_turn' in pot_cols and 'bet_turn' in bet_cols:
                result_df['pot_odds_turn'] = result_df.apply(
                    lambda row: row['bet_turn'] / row['pot_turn'] if row['pot_turn'] > 0 else np.nan,
                    axis=1
                )
                
            if 'pot_river' in pot_cols and 'bet_river' in bet_cols:
                result_df['pot_odds_river'] = result_df.apply(
                    lambda row: row['bet_river'] / row['pot_river'] if row['pot_river'] > 0 else np.nan,
                    axis=1
                )
                
            # Stack to pot ratio features
            if 'stack' in result_df.columns:
                for pot_col in pot_cols:
                    result_df[f'spr_{pot_col}'] = result_df.apply(
                        lambda row: row['stack'] / row[pot_col] if row[pot_col] > 0 else np.nan,
                        axis=1
                    )
                    
            return result_df