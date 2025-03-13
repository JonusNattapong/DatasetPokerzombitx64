import unittest
import pandas as pd
import numpy as np
from poker_ml.features.feature_generator import PokerFeatureGenerator

class TestPokerFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        self.feature_gen = PokerFeatureGenerator(target_col='action')
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'hand_id': [1, 1, 2, 2],
            'position': ['BTN', 'BB', 'SB', 'CO'],
            'cards': ['As Ah', 'Ks Kh', '2d 3d', 'Jc Tc'],
            'action': ['raise', 'fold', 'call', 'raise'],
            'pot_pre': [10, 10, 20, 20],
            'pot_flop': [20, 20, 40, 40],
            'bet_pre': [5, 2, 10, 5],
            'bet_flop': [10, 0, 20, 10],
            'stack': [100, 95, 200, 150],
            'vpip': [0.3, 0.2, 0.4, 0.35],
            'pfr': [0.25, 0.15, 0.3, 0.28],
            'aggression_factor': [1.2, 0.8, 1.5, 1.3]
        })

    def test_transform_basic(self):
        """Test basic transformation functionality"""
        X, y = self.feature_gen.transform(self.sample_data)
        
        # Check shapes
        self.assertEqual(len(y), len(self.sample_data))
        self.assertEqual(X.shape[0], len(self.sample_data))
        
        # Check target values
        self.assertTrue(np.array_equal(y, self.sample_data['action'].values))

    def test_card_features(self):
        """Test card feature generation"""
        transformed_df = self.feature_gen._add_card_features(self.sample_data)
        
        # Check hand strength was calculated
        self.assertIn('hand_strength', transformed_df.columns)
        
        # Check pair of aces has high hand strength
        aces_strength = transformed_df.loc[0, 'hand_strength']
        low_cards_strength = transformed_df.loc[2, 'hand_strength']
        self.assertGreater(aces_strength, low_cards_strength)

    def test_pot_features(self):
        """Test pot feature generation"""
        transformed_df = self.feature_gen._add_pot_features(self.sample_data)
        
        # Check pot odds features were created
        self.assertIn('pot_odds_pre', transformed_df.columns)
        self.assertIn('pot_odds_flop', transformed_df.columns)
        
        # Check stack to pot ratio features
        self.assertIn('spr_pot_pre', transformed_df.columns)
        self.assertIn('spr_pot_flop', transformed_df.columns)
        
        # Verify pot odds calculation
        expected_pot_odds = self.sample_data['bet_pre'] / self.sample_data['pot_pre']
        pd.testing.assert_series_equal(
            transformed_df['pot_odds_pre'],
            expected_pot_odds,
            check_names=False
        )

    def test_player_history(self):
        """Test player history feature generation"""
        transformed_df = self.feature_gen._add_player_history(self.sample_data)
        
        # Check player aggressiveness was calculated
        self.assertIn('player_aggressiveness', transformed_df.columns)
        
        # Check aggressiveness calculation
        expected_agg = (0.5 * self.sample_data['vpip'] + 
                       0.5 * self.sample_data['pfr'] * self.sample_data['aggression_factor'])
        pd.testing.assert_series_equal(
            transformed_df['player_aggressiveness'],
            expected_agg,
            check_names=False
        )

    def test_column_identification(self):
        """Test column type identification"""
        self.feature_gen._identify_column_types(self.sample_data)
        
        # Check categorical columns
        self.assertIn('position', self.feature_gen.categorical_cols)
        self.assertIn('cards', self.feature_gen.categorical_cols)
        
        # Check numeric columns
        self.assertIn('pot_pre', self.feature_gen.numeric_cols)
        self.assertIn('stack', self.feature_gen.numeric_cols)
        
        # Check target column exclusion
        self.assertNotIn('action', self.feature_gen.categorical_cols)
        self.assertNotIn('action', self.feature_gen.numeric_cols)

    def test_feature_matrix_preparation(self):
        """Test feature matrix preparation"""
        # First identify column types
        self.feature_gen._identify_column_types(self.sample_data)
        
        # Prepare feature matrix
        X = self.feature_gen._prepare_feature_matrix(self.sample_data, fit=True)
        
        # Check result is numpy array
        self.assertIsInstance(X, np.ndarray)
        
        # Check standardization of numeric columns
        numeric_data = self.sample_data[self.feature_gen.numeric_cols].copy()
        
        # First get the mean of original data
        original_mean = numeric_data.values.mean()
        self.assertNotEqual(original_mean, 0)  # Original data should not be centered
        
        # Transform the numeric data
        transformed_data = self.feature_gen.standard_scaler.fit_transform(numeric_data)
        transformed_mean = transformed_data.mean()
        
        # Transformed data should be centered around 0
        self.assertAlmostEqual(transformed_mean, 0, places=10)
        
        # Test transform without fit
        X2 = self.feature_gen._prepare_feature_matrix(self.sample_data, fit=False)
        self.assertEqual(X.shape, X2.shape)

if __name__ == '__main__':
    unittest.main()