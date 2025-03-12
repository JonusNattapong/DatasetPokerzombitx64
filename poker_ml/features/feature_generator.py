# poker_ml/features/feature_generator.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from .card_encoder import CardEncoder
from .position_features import PositionFeatureExtractor

class PokerFeatureGenerator:
    """Main class for generating features from poker hand data"""
    
    def __init__(self, target_col: str = 'action'):
        """
        Initialize the feature generator
        
        Args:
            target_col: Name of the target column
        """
        self.target_col = target_col
        self.card_encoder = CardEncoder()
        self.position_extractor = PositionFeatureExtractor()
        self.standard_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.categorical_cols = []
        self.numeric_cols = []
        
    def transform(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform poker data into ML features
        
        Args:
            df: DataFrame with poker hand data
            fit: Whether to fit the transformers
            
        Returns:
            Tuple of (X, y) for ML model inputs
        """
        # Ensure the target column exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
            
        # 1. Extract position features
        df = self.position_extractor.extract_position_features(df)
        
        # 2. Create features for cards
        df = self._add_card_features(df)
        
        # 3. Create pot and betting features
        df = self._add_pot_features(df)
        
        # 4. Create player history features
        df = self._add_player_history(df)
        
        # 5. Identify categorical and numerical columns
        if fit:
            self._identify_column_types(df)
        
        # 6. Prepare feature matrix X
        X = self._prepare_feature_matrix(df, fit)
        
        # 7. Prepare target variable y
        y = df[self.target_col].values if self.target_col in df.columns else None
        
        return X, y
    
    def _add_card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to cards
        
        Args:
            df: DataFrame with poker hand data
            
        Returns:
            DataFrame with additional card features
        """
        result_df = df.copy()
        
        # Add hand strength feature if cards are available
        if 'cards' in result_df.columns:
            # Calculate basic hand strength
            result_df['hand_strength'] = result_df['cards'].apply(
                lambda cards: self.card_encoder.calculate_hand_strength(cards)
            )
            
            # Add more detailed features if board cards are available
            board_cols = [col for col in result_df.columns if 'board_' in col]
            if board_cols:
                # Combine all board cards into a single string
                result_df['board_cards'] = result_df.apply(
                    lambda row: ' '.join(str(row[col]) for col in board_cols if pd.notna(row[col])),
                    axis=1
                )
                
                                # Calculate hand strength with board
                result_df['hand_strength_with_board'] = result_df.apply(
                    lambda row: self.card_encoder.calculate_hand_strength(row['cards'], row['board_cards']),
                    axis=1
                )
                
        return result_df
    
    def _add_pot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to pot odds and betting
        
        Args:
            df: DataFrame with poker hand data
            
        Returns:
            DataFrame with additional pot features
        """
        result_df = df.copy()
        
        # Calculate pot odds when applicable
        pot_cols = [c for c in df.columns if 'pot_' in c]
        bet_cols = [c for c in df.columns if 'bet_' in c]
        
        if 'pot_pre' in pot_cols and 'bet_pre' in bet_cols:
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
    
    def _add_player_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to player history and tendencies
        
        Args:
            df: DataFrame with poker hand data
            
        Returns:
            DataFrame with additional player history features
        """
        result_df = df.copy()
        
        # Example player history features
        if 'vpip' in result_df.columns:
            result_df['vpip'] = result_df['vpip'].fillna(0)
            
        if 'pfr' in result_df.columns:
            result_df['pfr'] = result_df['pfr'].fillna(0)
            
        if 'aggression_factor' in result_df.columns:
            result_df['aggression_factor'] = result_df['aggression_factor'].fillna(1)
            
        # Example of combining history features into a composite score
        result_df['player_aggressiveness'] = (
            0.5 * result_df['vpip'] + 0.5 * result_df['pfr'] * result_df['aggression_factor']
        )
        
        return result_df
    
    def _identify_column_types(self, df: pd.DataFrame):
        """
        Identify categorical and numeric columns
        
        Args:
            df: DataFrame with poker hand data
        """
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude target column from feature columns
        if self.target_col in self.categorical_cols:
            self.categorical_cols.remove(self.target_col)
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
    
    def _prepare_feature_matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """
        Prepare the feature matrix for ML models
        
        Args:
            df: DataFrame with poker hand data
            fit: Whether to fit the transformers
            
        Returns:
            Feature matrix as numpy array
        """
        # Handle missing values
        if fit:
            df[self.numeric_cols] = self.imputer.fit_transform(df[self.numeric_cols])
        else:
            df[self.numeric_cols] = self.imputer.transform(df[self.numeric_cols])
        
        # Standardize numeric features
        if fit:
            df[self.numeric_cols] = self.standard_scaler.fit_transform(df[self.numeric_cols])
        else:
            df[self.numeric_cols] = self.standard_scaler.transform(df[self.numeric_cols])
        
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        
        return df.values