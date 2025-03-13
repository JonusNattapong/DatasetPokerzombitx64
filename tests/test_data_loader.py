import unittest
import os
import pandas as pd
import tempfile
from poker_ml.data.loader import PokerDataLoader

class TestPokerDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.loader = PokerDataLoader(data_dir=self.temp_dir)
        
        # Create sample hand data
        self.sample_data = pd.DataFrame({
            'hand_id': range(100),
            'action': ['raise', 'fold', 'call'] * 33 + ['raise'],
            'position': ['BTN', 'SB', 'BB'] * 33 + ['BTN'],
            'amount': range(100)
        })
        
        # Save sample data to CSV
        self.sample_data.to_csv(os.path.join(self.temp_dir, 'hands.csv'), index=False)
        
        # Create raw data directory and sample hand history file
        self.raw_dir = os.path.join(self.temp_dir, "..", "raw_data")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Create sample hand history
        self.sample_history = """
        PokerStars Hand #1: Hold'em No Limit
        Table '1' 6-max
        Seat 1: Player1 (100 in chips)
        
        PokerStars Hand #2: Hold'em No Limit
        Table '2' 6-max
        Seat 1: Player2 (200 in chips)
        """
        with open(os.path.join(self.raw_dir, 'output.txt'), 'w') as f:
            f.write(self.sample_history)

    def tearDown(self):
        """Clean up test environment after each test case"""
        # Remove test files
        if os.path.exists(os.path.join(self.temp_dir, 'hands.csv')):
            os.remove(os.path.join(self.temp_dir, 'hands.csv'))
        if os.path.exists(os.path.join(self.raw_dir, 'output.txt')):
            os.remove(os.path.join(self.raw_dir, 'output.txt'))
        
        # Remove test directories
        if os.path.exists(self.raw_dir):
            os.rmdir(self.raw_dir)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_load_hand_data(self):
        """Test loading hand data from CSV"""
        loaded_data = self.loader.load_hand_data()
        
        # Check data was loaded correctly
        self.assertEqual(len(loaded_data), len(self.sample_data))
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)
        
        # Test error handling for missing file
        with self.assertRaises(FileNotFoundError):
            self.loader.load_hand_data('nonexistent.csv')

    def test_split_data(self):
        """Test data splitting functionality"""
        train_df, val_df, test_df = self.loader.split_data(self.sample_data)
        
        # Check split proportions
        total_size = len(self.sample_data)
        self.assertAlmostEqual(len(test_df) / total_size, 0.3, places=1)
        self.assertAlmostEqual(len(val_df) / total_size, 0.15, places=1)
        self.assertAlmostEqual(len(train_df) / total_size, 0.55, places=1)
        
        # Check no overlap between sets
        train_ids = set(train_df['hand_id'])
        val_ids = set(val_df['hand_id'])
        test_ids = set(test_df['hand_id'])
        
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)
        
        # Check stratification
        def get_action_dist(df):
            return df['action'].value_counts(normalize=True)
        
        train_dist = get_action_dist(train_df)
        val_dist = get_action_dist(val_df)
        test_dist = get_action_dist(test_df)
        
        # Check distributions are similar (within 10%)
        for action in train_dist.index:
            self.assertLess(abs(train_dist[action] - val_dist[action]), 0.1)
            self.assertLess(abs(train_dist[action] - test_dist[action]), 0.1)

    def test_load_raw_hand_history(self):
        """Test loading raw hand history data"""
        hands = self.loader.load_raw_hand_history()
        
        # Check correct number of hands loaded
        self.assertEqual(len(hands), 2)
        
        # Check hand content
        self.assertTrue(all('PokerStars Hand #' in hand for hand in hands))
        
        # Test error handling for missing file
        with self.assertRaises(FileNotFoundError):
            self.loader.load_raw_hand_history('nonexistent.txt')

if __name__ == '__main__':
    unittest.main()