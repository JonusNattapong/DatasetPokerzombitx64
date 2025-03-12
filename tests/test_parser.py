"""
Unit tests for the parser module.
"""

import unittest
import os
import pandas as pd
import tempfile
from pokerdata.parser import (
    tourn_id, table_number, hand, chairs, get_board, get_combination,
    buyin, get_date_time, get_button, level, get_cards, initial_stack,
    bets, uncalled
)


class TestParserFunctions(unittest.TestCase):
    """Test case for parser helper functions."""
    
    def test_tourn_id(self):
        """Test tournament ID extraction."""
        self.assertEqual(tourn_id("PokerStars Hand #123456789: Tournament #987654321"), 987654321)
        self.assertEqual(tourn_id("No tournament ID"), 0)
    
    def test_table_number(self):
        """Test table number extraction."""
        self.assertEqual(table_number("Table '12345678 1'"), "1")
        self.assertEqual(table_number("Table 'Main 2'"), "2")
    
    def test_hand(self):
        """Test hand ID extraction."""
        self.assertEqual(hand("PokerStars Hand #123456789:"), 123456789)
        self.assertEqual(hand("No hand ID"), 0)
    
    def test_chairs(self):
        """Test chairs extraction."""
        self.assertEqual(chairs("Table '12345 1' 6-max"), 6)
        self.assertEqual(chairs("Table '12345 1' 9-max"), 9)
        self.assertEqual(chairs("No max info"), 9)  # Default value
    
    def test_get_board(self):
        """Test board extraction."""
        self.assertEqual(get_board("*** FLOP *** [As Ks Qs]"), "As Ks Qs")
        self.assertEqual(get_board("*** TURN *** [As Ks Qs] [Js]"), "Js")
        self.assertEqual(get_board("*** SUMMARY ***"), "0")
        self.assertEqual(get_board("No board"), "-")
    
    def test_get_combination(self):
        """Test combination extraction."""
        self.assertEqual(get_combination("Player1: shows [As Ks] (a pair of Aces)"), "a pair of Aces")
        self.assertEqual(get_combination("Seat 1: Player1 folded"), "")
    
    def test_buyin(self):
        """Test buyin extraction."""
        self.assertEqual(buyin("Tournament #123, $1.50+$0.15"), "1.50+0.15")
        self.assertEqual(buyin("No buyin info"), "---")
    
    def test_get_date_time(self):
        """Test date and time extraction."""
        date, time = get_date_time("2023/01/15 14:30:45")
        self.assertEqual(date, "2023/01/15")
        self.assertEqual(time, "14:30:45")
        
        date, time = get_date_time("No date or time")
        self.assertEqual(date, "---")
        self.assertEqual(time, "---")
    
    def test_get_button(self):
        """Test button extraction."""
        self.assertEqual(get_button("Seat #3 is the button"), 3)
        self.assertEqual(get_button("No button info"), 0)
    
    def test_level(self):
        """Test level extraction."""
        self.assertEqual(level("Level I"), 1)
        self.assertEqual(level("Level V"), 5)
        self.assertEqual(level("Level X"), 10)
        self.assertEqual(level("No level info"), 0)
    
    def test_get_cards(self):
        """Test cards extraction."""
        self.assertEqual(get_cards("Dealt to Player1 [As Ks]"), "As Ks")
        self.assertEqual(get_cards("No cards"), "[---]")
    
    def test_initial_stack(self):
        """Test initial stack extraction."""
        self.assertEqual(initial_stack("Seat 1: Player1 (1000 in chips)"), 1000)
        self.assertEqual(initial_stack("Seat 1: Player1"), 0)
    
    def test_bets(self):
        """Test bet extraction."""
        self.assertEqual(bets("Player1: bets 100"), 100)
        self.assertEqual(bets("Player1: raises 200 to 300"), 300)
        self.assertEqual(bets("No bet"), 0)
    
    def test_uncalled(self):
        """Test uncalled bet extraction."""
        self.assertEqual(uncalled("Uncalled bet (100) returned to Player1"), 100)
        self.assertEqual(uncalled("No uncalled bet"), 0)


if __name__ == '__main__':
    unittest.main()
