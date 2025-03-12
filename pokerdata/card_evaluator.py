"""Card evaluation module for poker hand analysis."""

from typing import List, Dict, Tuple, Union, Optional
import re

class Card:
    RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    SUITS = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}
    
    def __init__(self, card_str: str):
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        rank, suit = card_str[0], card_str[1]
        if rank not in self.RANKS:
            raise ValueError(f"Invalid card rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid card suit: {suit}")
        self.rank = rank
        self.suit = suit
        self.rank_value = self.RANKS[rank]
        self.suit_name = self.SUITS[suit]
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
        
    def __lt__(self, other) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank_value < other.rank_value

class Hand:
    HAND_RANKS = {
        "high_card": 1, "pair": 2, "two_pair": 3, "three_of_a_kind": 4,
        "straight": 5, "flush": 6, "full_house": 7, "four_of_a_kind": 8,
        "straight_flush": 9, "royal_flush": 10
    }
    
    def __init__(self, cards_str: str):
        # Handle input strings without delimiters by splitting every 2 characters
        if " " not in cards_str and "," not in cards_str:
            card_strings = [cards_str[i:i+2] for i in range(0, len(cards_str), 2)]
        else:
            card_strings = re.split(r"[\s,]+", cards_str.strip())
        
        self.cards = [Card(card) for card in card_strings if len(card) == 2]
        if len(self.cards) < 2:
            raise ValueError(f"Not enough valid cards in: {cards_str}")
    
    def __str__(self) -> str:
        return " ".join(str(card) for card in self.cards)
    
    def __repr__(self) -> str:
        return f"Hand('{str(self)}')"

    def _find_straight(self, cards: List[Card]) -> Optional[List[Card]]:
        if len(cards) < 5:
            return None

        # Get unique rank values
        values = set(c.rank_value for c in cards)
        
        # Get all values in descending order
        sorted_values = sorted(values, reverse=True)
        
        # Look for 5 consecutive values
        for high_value in range(14, 4, -1):  # Check from Ace down to 5
            needed = list(range(high_value, high_value-5, -1))  # 5 consecutive values
            if all(v in values for v in needed):  # Check if we have all needed values
                straight = []
                # Build straight from highest to lowest using exactly these values
                for v in needed:
                    straight.append(next(c for c in cards if c.rank_value == v))
                return straight  # Return first straight found (will be highest possible)
            
        # After checking for regular straights, check for wheel straight (A-5)
        wheel_values = [5, 4, 3, 2, 14]  # Values in A-5 straight
        if all(v in values for v in wheel_values):
            wheel = []
            # Build wheel straight with correct order (5-high)
            for v in wheel_values[:-1]:  # Add 5,4,3,2 first
                wheel.append(next(c for c in cards if c.rank_value == v))
            wheel.append(next(c for c in cards if c.rank_value == 14))  # Add Ace last
            return wheel  # Return wheel since we've already tried higher straights
            
        return None

    def evaluate(self, board_str: Optional[str] = None) -> Tuple[str, List[Card], int]:
        """Evaluate a poker hand.
        Returns:
        - hand_name: String name of the hand (e.g. 'straight', 'flush')
        - best_cards: List of 5 cards that make up the best hand
        - strength: Integer ranking of the hand (higher is better)
        """
        all_cards = self.cards.copy()
        if board_str:
            board = Hand(board_str)
            all_cards.extend(board.cards)
            
        print(f"\nEvaluating hand {self.cards} with board {board_str}")
        
        all_cards.sort(key=lambda x: x.rank_value, reverse=True)
        suits = {}
        for card in all_cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
            
        flush = None
        for cards in suits.values():
            if len(cards) >= 5:
                flush = sorted(cards, key=lambda x: x.rank_value, reverse=True)[:5]
                straight_flush = self._find_straight(flush)
                if straight_flush:
                    if straight_flush[0].rank_value == 14 and straight_flush[-1].rank_value == 10:
                        return ("royal_flush", straight_flush, self.HAND_RANKS["royal_flush"])
                    return ("straight_flush", straight_flush, self.HAND_RANKS["straight_flush"])
        
        # Check for flush and straight
        straight = self._find_straight(all_cards)
        
        # Return hands in order of rank
        if flush:
            # Flush is higher ranked than straight
            return ("flush", flush, self.HAND_RANKS["flush"])
        if straight:
            print(f"Found straight: {[str(c) for c in straight]}")
            return ("straight", straight, self.HAND_RANKS["straight"])
            
        ranks = {}
        for card in all_cards:
            if card.rank_value not in ranks:
                ranks[card.rank_value] = []
            ranks[card.rank_value].append(card)
        
        # Four of a kind
        for cards in ranks.values():
            if len(cards) == 4:
                kickers = [c for c in all_cards if c not in cards][:1]
                return ("four_of_a_kind", cards + kickers, self.HAND_RANKS["four_of_a_kind"])
        
        # Full house and three of a kind
        threes = sorted([cards for cards in ranks.values() if len(cards) >= 3],
                       key=lambda x: x[0].rank_value, reverse=True)
        pairs = sorted([cards for cards in ranks.values() if len(cards) >= 2],
                      key=lambda x: x[0].rank_value, reverse=True)
        
        if threes:
            three = threes[0][:3]
            pair = next((cards[:2] for cards in pairs if cards[0].rank_value != three[0].rank_value), None)
            if pair:
                return ("full_house", three + pair, self.HAND_RANKS["full_house"])
            kickers = [c for c in all_cards if c.rank_value != three[0].rank_value][:2]
            return ("three_of_a_kind", three + kickers, self.HAND_RANKS["three_of_a_kind"])
        
        # Two pair
        if len(pairs) >= 2:
            two_pairs = pairs[0][:2] + pairs[1][:2]
            kickers = [c for c in all_cards if c not in two_pairs][:1]
            return ("two_pair", two_pairs + kickers, self.HAND_RANKS["two_pair"])
        
        # One pair
        if pairs:
            pair = pairs[0][:2]
            kickers = [c for c in all_cards if c not in pair][:3]
            return ("pair", pair + kickers, self.HAND_RANKS["pair"])
        
        # High card
        return ("high_card", all_cards[:5], self.HAND_RANKS["high_card"])

def analyze_hand(hand_str: str, board_str: Optional[str] = None) -> Dict:
    try:
        hand = Hand(hand_str)
        hand_name, best_cards, strength = hand.evaluate(board_str)
        return {
            "hand": hand_str,
            "board": board_str if board_str else "",
            "hand_name": hand_name,
            "best_five": " ".join(str(card) for card in best_cards),
            "strength": strength
        }
    except Exception as e:
        return {
            "hand": hand_str,
            "board": board_str if board_str else "",
            "error": str(e)
        }

def compare_hands(hand1_str: str, hand2_str: str, board_str: Optional[str] = None) -> int:
    print(f"\nComparing hands:\nHand 1: {hand1_str}\nHand 2: {hand2_str}\nBoard: {board_str}")
    """
    Compare two poker hands and return:
     1 if hand1 wins
    -1 if hand2 wins
     0 if it's a tie
    
    Also handles invalid input more gracefully
    """
    try:
        # Create Hand objects
        hand1 = Hand(hand1_str)
        hand2 = Hand(hand2_str)
        
        # Evaluate both hands
        name1, cards1, strength1 = hand1.evaluate(board_str)
        print(f"Hand 1 evaluated as: {name1} (strength {strength1})")
        print(f"Best cards: {[str(c) for c in cards1]}")
        
        name2, cards2, strength2 = hand2.evaluate(board_str)
        print(f"Hand 2 evaluated as: {name2} (strength {strength2})")
        print(f"Best cards: {[str(c) for c in cards2]}")
        
        # First compare hand strengths
        if strength1 != strength2:
            result = 1 if strength1 > strength2 else -1
            print(f"Different strengths: {strength1} vs {strength2}, returning {result}")
            return result
        
        # For same strength hands, handle special cases
        if name1 == "straight" and name2 == "straight":
            # For straights, we need to compare the highest card in each straight
            # Get the highest card in each straight (normally the first card)
            high_card1 = max(cards1, key=lambda c: c.rank_value)
            high_card2 = max(cards2, key=lambda c: c.rank_value)
            
            # Special case: A-5 straight (wheel) has Ace as the lowest card
            is_wheel1 = set(c.rank_value for c in cards1) == set([14, 5, 4, 3, 2])
            is_wheel2 = set(c.rank_value for c in cards2) == set([14, 5, 4, 3, 2])
            
            # A wheel straight is the lowest straight (5-high)
            if is_wheel1 and not is_wheel2:
                return -1
            elif not is_wheel1 and is_wheel2:
                return 1
            elif is_wheel1 and is_wheel2:
                return 0  # Both are wheels
            
            # For regular straights, compare the highest card
            if high_card1.rank_value != high_card2.rank_value:
                return 1 if high_card1.rank_value > high_card2.rank_value else -1
            return 0
        else:
            # For other hands, compare cards in order
            for card1, card2 in zip(cards1, cards2):
                if card1.rank_value != card2.rank_value:
                    return 1 if card1.rank_value > card2.rank_value else -1
        
        # If we get here, it's a true tie
        return 0
        
    except ValueError as e:
        # Handle invalid card format
        raise ValueError(f"Invalid hand format: {str(e)}")
    except Exception as e:
        # Log unexpected errors but don't expose internals
        print(f"Error comparing hands: {str(e)}")
        raise ValueError("Error comparing hands")
