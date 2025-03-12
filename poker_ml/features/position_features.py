# poker_ml/features/position_features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PositionFeatureExtractor:
    """Extract features based on player positions"""
    
    def __init__(self):
        """Initialize the position feature extractor"""
        # Position order from worst to best
        self.position_order = ['SB', 'BB', 'UTG', 'MP', 'HJ', 'CO', 'BTN']
        
    def position_to_numeric(self, position: str) -> float:
        """
        Convert position to numeric value representing position advantage
        
        Args:
            position: Position string
            
        Returns:
            Numeric value where higher is better position (0-1)
        """
        if not position or not isinstance(position, str):
            return np.nan
            
        # Standardize the position
        std_pos = self._standardize_position(position)
        
        # Map to numeric value based on order
        if std_pos in self.position_order:
            idx = self.position_order.index(std_pos)
            # Normalize to 0-1 range
            return idx / (len(self.position_order) - 1)
        
        # Default value for unknown positions
        return 0.5
    
    def _standardize_position(self, position: str) -> str:
        """
        Standardize position string
        
        Args:
            position: Raw position string
            
        Returns:
            Standardized position code
        """
        position = position.upper()
        
        # Map common variations
        position_map = {
            'BUTTON': 'BTN',
            'DEALER': 'BTN',
            'CUTOFF': 'CO',
            'HIJACK': 'HJ',
            'SMALL BLIND': 'SB',
            'BIG BLIND': 'BB',
            'UNDER THE GUN': 'UTG',
            'MIDDLE POSITION': 'MP'
        }
        
        for key, value in position_map.items():
            if key in position:
                return value
                
        for std_pos in self.position_order:
            if std_pos in position:
                return std_pos
                
        return position
    
    def extract_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract position-related features from the dataframe
        
        Args:
            df: DataFrame with poker hand data
            
        Returns:
            DataFrame with additional position features
        """
        result_df = df.copy()
        
        # Add positional advantage score
        if 'position' in result_df.columns:
            result_df['position_advantage'] = result_df['position'].apply(
                lambda p: self.position_to_numeric(p)
            )
            
            # Add one-hot encoding for positions
            position_dummies = pd.get_dummies(
                result_df['position'].apply(self._standardize_position),
                prefix='pos', dummy_na=False
            )
            
            result_df = pd.concat([result_df, position_dummies], axis=1)
            
        # Add relative position features if game has multiple players
        if 'hand_id' in result_df.columns and 'position' in result_df.columns:
            # Group by hand
            hand_groups = result_df.groupby('hand_id')
            
            # For each player in a hand, calculate relative position
            result_df['players_in_hand'] = result_df['hand_id'].apply(
                lambda hand_id: len(hand_groups.get_group(hand_id))
            )
            
            result_df['relative_position'] = result_df.apply(
                lambda row: self._calc_relative_position(row, hand_groups),
                axis=1
            )
            
        return result_df
        
    def _calc_relative_position(self, row: pd.Series, hand_groups) -> float:
        """
        Calculate relative position compared to other players in the hand
        
        Args:
            row: Row from the dataframe
            hand_groups: GroupBy object for hands
            
        Returns:
            Relative position score (0-1)
        """
        try:
            hand_id = row['hand_id']
            hand_df = hand_groups.get_group(hand_id)
            
            # Get positions for all players in the hand
            positions = hand_df['position'].apply(self._standardize_position).tolist()
            
            # Map to numeric values
            position_values = [
                self.position_order.index(p) if p in self.position_order else -1
                for p in positions
            ]
            
            # Get current player's position value
            player_pos = self._standardize_position(row['position'])
            player_value = (
                self.position_order.index(player_pos)
                if player_pos in self.position_order else -1
            )
            
            if player_value == -1 or max(position_values) == min(position_values):
                return 0.5
                
            # Normalize to 0-1 range
            return (player_value - min(position_values)) / (max(position_values) - min(position_values))
            
        except (KeyError, ValueError):
            return 0.5