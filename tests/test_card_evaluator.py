"""
Unit tests for the card evaluator module.
"""

import unittest
from pokerdata.card_evaluator import Card, Hand, analyze_hand, compare_hands


class TestCardEvaluator(unittest.TestCase):
    """Test case for card evaluator classes and functions."""
    
    def test_card_creation(self):
        """Test card creation and properties."""
        card = Card('As')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, 's')
        self.assertEqual(card.rank_value, 14)
        self.assertEqual(card.suit_name, 'spades')
        
        with self.assertRaises(ValueError):
            Card('X')  # Invalid format
        
        with self.assertRaises(ValueError):
            Card('Az')  # Invalid suit
    
    def test_card_comparison(self):
        """Test card comparison operations."""
        ace_spades = Card('As')
        ace_hearts = Card('Ah')
        king_spades = Card('Ks')
        
        self.assertEqual(ace_spades, ace_spades)
        self.assertNotEqual(ace_spades, ace_hearts)
        self.assertGreater(ace_spades, king_spades)
        self.assertLess(king_spades, ace_hearts)
    
    def test_hand_creation(self):
        """Test hand creation and properties."""
        hand = Hand('As Ks')
        self.assertEqual(len(hand.cards), 2)
        self.assertEqual(str(hand), "As Ks")
        
        with self.assertRaises(ValueError):
            Hand('')  # Empty hand
    
    def test_hand_evaluation(self):
        """Test hand evaluation."""
        # Royal flush
        hand = Hand('As Ks')
        hand_name, _, strength = hand.evaluate('Qs Js Ts')
        self.assertEqual(hand_name, 'royal_flush')
        self.assertEqual(strength, 10)
        
        # Straight flush
        hand = Hand('9s 8s')
        hand_name, _, strength = hand.evaluate('7s 6s 5s')
        self.assertEqual(hand_name, 'straight_flush')
        self.assertEqual(strength, 9)
        
        # Four of a kind
        hand = Hand('Ac Ad')
        hand_name, _, strength = hand.evaluate('Ah As Ks')
        self.assertEqual(hand_name, 'four_of_a_kind')
        self.assertEqual(strength, 8)
        
        # Full house
        hand = Hand('Ac Ad')
        hand_name, _, strength = hand.evaluate('Ah Ks Kh')
        self.assertEqual(hand_name, 'full_house')
        self.assertEqual(strength, 7)
        
        # Flush
        hand = Hand('As 3s')
        hand_name, _, strength = hand.evaluate('Ks Qs 7s 2c')
        self.assertEqual(hand_name, 'flush')
        self.assertEqual(strength, 6)
        
        # Straight
        hand = Hand('9c 8d')
        hand_name, _, strength = hand.evaluate('7h 6s 5c')
        self.assertEqual(hand_name, 'straight')
        self.assertEqual(strength, 5)
        
        # Three of a kind
        hand = Hand('Ac Ad')
        hand_name, _, strength = hand.evaluate('Ah Ks Qc')
        self.assertEqual(hand_name, 'three_of_a_kind')
        self.assertEqual(strength, 4)
        
        # Two pair
        hand = Hand('Ac Kd')
        hand_name, _, strength = hand.evaluate('Ah Ks Qc')
        self.assertEqual(hand_name, 'two_pair')
        self.assertEqual(strength, 3)
        
        # Pair
        hand = Hand('Ac Qd')
        hand_name, _, strength = hand.evaluate('Ah Ks 2c')
        self.assertEqual(hand_name, 'pair')
        self.assertEqual(strength, 2)
        
        # High card
        hand = Hand('Ac Qd')
        hand_name, _, strength = hand.evaluate('Kh Js 2c')
        self.assertEqual(hand_name, 'high_card')
        self.assertEqual(strength, 1)
    
    def test_analyze_hand(self):
        """Test hand analysis function."""
        result = analyze_hand('As Ks', 'Qs Js Ts')
        self.assertEqual(result['hand_name'], 'royal_flush')
        self.assertEqual(result['strength'], 10)
        
        # Test error handling
        result = analyze_hand('invalid', 'board')
        self.assertIn('error', result)
    
    def test_compare_hands(self):
        """Test hand comparison function."""
        # Hand1 wins (royal flush vs straight flush)
        self.assertEqual(compare_hands('As Ks', '9s 8s', 'Qs Js Ts 7s 6s'), 1)
        
        # Hand2 wins (straight vs pair)
        self.assertEqual(compare_hands('Ac Kd', '9c 8d', 'Qh Js Ts'), -1)
        
        # Equal hands
        self.assertEqual(compare_hands('Ac Kd', 'Ah Ks', 'Qh Js Ts 2c 3d'), 0)


if __name__ == '__main__':
    unittest.main()
