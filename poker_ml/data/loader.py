# poker_ml/data/loader.py
import os
import pandas as pd
from typing import Tuple, List, Dict, Any

class PokerDataLoader:
    """Utility class for loading poker hand history data"""
    
    def __init__(self, data_dir: str = "./processed_data"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing processed hand data
        """
        self.data_dir = data_dir
        
    def load_hand_data(self, filename: str = "hands.csv") -> pd.DataFrame:
        """
        Load processed poker hand data
        
        Args:
            filename: Name of the CSV file containing hand data
            
        Returns:
            DataFrame containing hand data
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hand data file not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.3, val_size: float = 0.15, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets
        
        Args:
            df: DataFrame containing hand data
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split off the test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['action'] if 'action' in df.columns else None
        )
        
        # Then split the remaining data into training and validation sets
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=train_val_df['action'] if 'action' in train_val_df.columns else None
        )
        
        return train_df, val_df, test_df
    
    def load_raw_hand_history(self, filename: str = "output.txt") -> List[str]:
        """
        Load raw hand history text data
        
        Args:
            filename: Name of the text file containing raw hand histories
            
        Returns:
            List of strings, each representing a hand history
        """
        file_path = os.path.join(self.data_dir, "..", "raw_data", filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw hand history file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            # Split the file by hand markers
            text = f.read()
            hands = text.split("PokerStars Hand #")
            hands = ["PokerStars Hand #" + hand for hand in hands[1:]]  # Skip the first empty split
            
        return hands