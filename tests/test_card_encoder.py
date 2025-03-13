import unittest
import numpy as np
from poker_ml.features.card_encoder import CardEncoder

class TestCardEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = CardEncoder()

    def test_encode_card(self):
        """Test single card encoding"""
        # Test regular cards
        self.assertEqual(self.encoder.encode_card("As"), (14, 0))  # Ace of spades
        self.assertEqual(self.encoder.encode_card("Kh"), (13, 1))  # King of hearts
        self.assertEqual(self.encoder.encode_card("2d"), (2, 2))   # 2 of diamonds
        self.assertEqual(self.encoder.encode_card("Tc"), (10, 3))  # 10 of clubs
        
        # Test edge cases
        self.assertEqual(self.encoder.encode_card(""), (0, 0))     # Empty string
        self.assertEqual(self.encoder.encode_card("X"), (0, 0))    # Invalid card
        self.assertEqual(self.encoder.encode_card("AX"), (14, 0))  # Invalid suit

    def test_encode_hand(self):
        """Test hand encoding"""
        # Test regular hand
        hand = "As Kh"
        encoded = self.encoder.encode_hand(hand)
        expected = np.array([(14, 0), (13, 1)])
        np.testing.assert_array_equal(encoded, expected)
        
        # Test hand with more than 2 cards (board)
        board = "As Kh Qd Jc Th"
        encoded = self.encoder.encode_hand(board)
        expected = np.array([(14, 0), (13, 1), (12, 2), (11, 3), (10, 1)])
        np.testing.assert_array_equal(encoded, expected)
        
        # Test edge cases
        empty_hand = self.encoder.encode_hand("")
        np.testing.assert_array_equal(empty_hand, np.zeros((2, 2)))
        
        invalid_hand = self.encoder.encode_hand(None)
        np.testing.assert_array_equal(invalid_hand, np.zeros((2, 2)))

    def test_calculate_hand_strength(self):
        """Test hand strength calculation"""
        # Test pair
        strength = self.encoder.calculate_hand_strength("As Ah")
        self.assertGreaterEqual(strength, 0.8)  # High pair should be strong
        
        # Test suited connectors
        strength = self.encoder.calculate_hand_strength("Ah Kh")
        self.assertGreater(strength, 0.6)  # Suited connectors should be decent
        
        # Test weak hand
        strength = self.encoder.calculate_hand_strength("2c 7d")
        self.assertLess(strength, 0.4)  # Weak hand should have low strength
        
        # Test with board
        strength = self.encoder.calculate_hand_strength("As Ks", "Ah Kh Qh")
        self.assertGreater(strength, 0.5)  # Two pair potential
        
        # Test edge cases
        self.assertEqual(self.encoder.calculate_hand_strength(""), 0.0)
        self.assertEqual(self.encoder.calculate_hand_strength(None), 0.0)

if __name__ == '__main__':
    unittest.main()