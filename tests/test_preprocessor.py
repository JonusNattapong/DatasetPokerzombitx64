import unittest
import pandas as pd
import numpy as np
from poker_ml.data.preprocessor import PokerDataPreprocessor

class TestPokerDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        self.preprocessor = PokerDataPreprocessor()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'hand_id': [1, 2, 3, 4],
            'action_pre': ['raises 10', 'calls 5', 'folds', 'checks'],
            'action_flop': ['bets 20', 'calls 20', 'folds', 'raises 40'],
            'position': ['BUTTON', 'BB', 'SB', 'UTG'],
            'cards': ['As Ah', 'Ks Kh', '2d 3d', 'Jc Th'],  # Changed Tc to Th to make it unsuited
            'pot_pre': [10, 10, 20, 20],
            'pot_flop': [20, 20, 40, 40],
            'bet_pre': [5, 2, 10, 5],
            'bet_flop': [10, 0, 20, 10],
            'stack': [100, 95, 200, 150]
        })

    def test_extract_actions(self):
        """Test action extraction and normalization"""
        processed = self.preprocessor.extract_actions(self.sample_data)
        
        # Check normalized action columns were created
        self.assertIn('action_pre_normalized', processed.columns)
        self.assertIn('action_flop_normalized', processed.columns)
        
        # Check action normalization
        self.assertEqual(processed.loc[0, 'action_pre_normalized'], 'raise')
        self.assertEqual(processed.loc[1, 'action_pre_normalized'], 'call')
        self.assertEqual(processed.loc[2, 'action_pre_normalized'], 'fold')
        self.assertEqual(processed.loc[3, 'action_pre_normalized'], 'check')
        
        # Check primary action determination
        self.assertEqual(processed.loc[0, 'action'], 'raise')
        self.assertEqual(processed.loc[2, 'action'], 'fold')

    def test_encode_positions(self):
        """Test position encoding"""
        processed = self.preprocessor.encode_positions(self.sample_data)
        
        # Check standardized position column
        self.assertIn('position_std', processed.columns)
        self.assertEqual(processed.loc[0, 'position_std'], 'BTN')
        self.assertEqual(processed.loc[1, 'position_std'], 'BB')
        
        # Check one-hot encoded columns
        self.assertIn('pos_BTN', processed.columns)
        self.assertIn('pos_BB', processed.columns)
        self.assertIn('pos_SB', processed.columns)
        self.assertIn('pos_UTG', processed.columns)
        
        # Check one-hot encoding values
        self.assertEqual(processed.loc[0, 'pos_BTN'], 1)
        self.assertEqual(processed.loc[0, 'pos_BB'], 0)

    def test_create_hand_strength_features(self):
        """Test hand strength feature creation"""
        processed = self.preprocessor.create_hand_strength_features(self.sample_data)
        
        # Check hand strength features were created
        self.assertIn('has_pair', processed.columns)
        self.assertIn('has_broadways', processed.columns)
        self.assertIn('has_suited', processed.columns)
        self.assertIn('card_rank_max', processed.columns)
        
        # Test pair detection
        self.assertTrue(processed.loc[0, 'has_pair'])   # As Ah
        self.assertFalse(processed.loc[3, 'has_pair'])  # Jc Tc
        
        # Test broadway detection
        self.assertTrue(processed.loc[0, 'has_broadways'])   # As Ah
        self.assertFalse(processed.loc[2, 'has_broadways'])  # 2d 3d
        
        # Test suited detection
        self.assertTrue(processed.loc[2, 'has_suited'])   # 2d 3d
        self.assertFalse(processed.loc[3, 'has_suited'])  # Jc Tc
        
        # Test rank calculation
        self.assertEqual(processed.loc[0, 'card_rank_max'], 14)  # Ace
        self.assertEqual(processed.loc[2, 'card_rank_min'], 2)   # 2

    def test_create_pot_features(self):
        """Test pot feature creation"""
        processed = self.preprocessor.create_pot_features(self.sample_data)
        
        # Check pot odds features were created
        self.assertIn('pot_odds_pre', processed.columns)
        self.assertIn('pot_odds_flop', processed.columns)
        
        # Check stack to pot ratio features
        self.assertIn('spr_pot_pre', processed.columns)
        self.assertIn('spr_pot_flop', processed.columns)
        
        # Test pot odds calculation
        expected_pot_odds = self.sample_data['bet_pre'] / self.sample_data['pot_pre']
        pd.testing.assert_series_equal(
            processed['pot_odds_pre'],
            expected_pot_odds,
            check_names=False
        )
        
        # Test stack to pot ratio calculation
        expected_spr = self.sample_data['stack'] / self.sample_data['pot_pre']
        pd.testing.assert_series_equal(
            processed['spr_pot_pre'],
            expected_spr,
            check_names=False
        )

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with missing data
        edge_data = pd.DataFrame({
            'action_pre': [None, np.nan, '', 'unknown_action'],
            'position': [None, np.nan, '', 'UNKNOWN'],
            'cards': [None, np.nan, '', 'invalid_cards']
        })
        
        # Should not raise exceptions for any of these operations
        processed = self.preprocessor.extract_actions(edge_data)
        self.assertIn('action_pre_normalized', processed.columns)
        
        processed = self.preprocessor.encode_positions(edge_data)
        self.assertIn('position_std', processed.columns)
        
        processed = self.preprocessor.create_hand_strength_features(edge_data)
        self.assertIn('has_pair', processed.columns)

if __name__ == '__main__':
    unittest.main()