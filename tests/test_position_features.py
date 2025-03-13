import unittest
import pandas as pd
import numpy as np
from poker_ml.features.position_features import PositionFeatureExtractor

class TestPositionFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PositionFeatureExtractor()
    
    def test_position_to_numeric(self):
        """Test converting positions to numeric values"""
        # Test standard positions
        self.assertEqual(self.extractor.position_to_numeric('BTN'), 1.0)  # Button should be highest
        self.assertEqual(self.extractor.position_to_numeric('SB'), 0.0)   # Small blind should be lowest
        self.assertGreater(self.extractor.position_to_numeric('CO'), 
                          self.extractor.position_to_numeric('MP'))  # CO better than MP
        
        # Test edge cases
        self.assertTrue(np.isnan(self.extractor.position_to_numeric(None)))
        self.assertEqual(self.extractor.position_to_numeric('UNKNOWN'), 0.5)
    
    def test_standardize_position(self):
        """Test position string standardization"""
        # Test various position formats
        self.assertEqual(self.extractor._standardize_position('BUTTON'), 'BTN')
        self.assertEqual(self.extractor._standardize_position('small blind'), 'SB')
        self.assertEqual(self.extractor._standardize_position('UNDER THE GUN'), 'UTG')
        self.assertEqual(self.extractor._standardize_position('co'), 'CO')
        
        # Test unknown position
        self.assertEqual(self.extractor._standardize_position('RANDOM'), 'RANDOM')
    
    def test_extract_position_features(self):
        """Test extracting position features from DataFrame"""
        # Create test DataFrame
        data = {
            'hand_id': [1, 1, 1],
            'position': ['BTN', 'SB', 'BB'],
            'other_col': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        
        # Extract features
        result = self.extractor.extract_position_features(df)
        
        # Check position advantage was added
        self.assertIn('position_advantage', result.columns)
        self.assertEqual(result.loc[0, 'position_advantage'], 1.0)  # BTN should be 1.0
        
        # Check one-hot encoding
        self.assertIn('pos_BTN', result.columns)
        self.assertIn('pos_SB', result.columns)
        self.assertIn('pos_BB', result.columns)
        
        # Check relative position
        self.assertIn('relative_position', result.columns)
        self.assertIn('players_in_hand', result.columns)
        self.assertEqual(result['players_in_hand'].iloc[0], 3)
    
    def test_calc_relative_position(self):
        """Test relative position calculation"""
        # Create test hand group
        data = {
            'hand_id': [1, 1, 1],
            'position': ['BTN', 'SB', 'BB']
        }
        df = pd.DataFrame(data)
        hand_groups = df.groupby('hand_id')
        
        # Test BTN position (should be highest)
        row = pd.Series({'hand_id': 1, 'position': 'BTN'})
        rel_pos = self.extractor._calc_relative_position(row, hand_groups)
        self.assertEqual(rel_pos, 1.0)
        
        # Test SB position (should be lowest)
        row = pd.Series({'hand_id': 1, 'position': 'SB'})
        rel_pos = self.extractor._calc_relative_position(row, hand_groups)
        self.assertEqual(rel_pos, 0.0)
        
        # Test invalid hand
        row = pd.Series({'hand_id': 999, 'position': 'BTN'})
        rel_pos = self.extractor._calc_relative_position(row, hand_groups)
        self.assertEqual(rel_pos, 0.5)

if __name__ == '__main__':
    unittest.main()