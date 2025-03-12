"""
Card evaluation module for poker hand analysis.
"""

from typing import List, Dict, Tuple, Union, Optional
import re


class Card:
    """Representation of a playing card."""
    
    # Card ranks and their values (2 is lowest, A is highest)
    RANKS = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    # Card suits
    SUITS = {'s': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs'}
    
    def __init__(self, card_str: str):
        """
        Initialize a card from a string representation.
        
        Args:
            card_str: String representation of a card (e.g., 'As' for Ace of spades)
        """
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}. Must be 2 characters.")
        
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
    
    def __repr__(self) -> str:
        return f"Card('{self.rank}{self.suit}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank_value < other.rank_value


class Hand:
    """Representation of a poker hand."""
    
    # Hand ranks
    HAND_RANKS = {
        'high_card': 1,
        'pair': 2,
        'two_pair': 3,
        'three_of_a_kind': 4,
        'straight': 5,
        'flush': 6,
        'full_house': 7,
        'four_of_a_kind': 8,
        'straight_flush': 9,
        'royal_flush': 10
    }
    
    def __init__(self, cards_str: str):
        """
        Initialize a hand from a string representation.
        
        Args:
            cards_str: Space-separated string of cards (e.g., 'As Ks Qs Js Ts')
        """
        card_strings = re.split(r'[\s,]+', cards_str.strip())
        self.cards = [Card(card) for card in card_strings if len(card) == 2]
        
        if len(self.cards) < 2:
            raise ValueError(f"Not enough valid cards in: {cards_str}")
    
    def __str__(self) -> str:
        return " ".join(str(card) for card in self.cards)
    
    def __repr__(self) -> str:
        return f"Hand('{str(self)}')"
    
    def best_five_cards(self, board_str: Optional[str] = None) -> List[Card]:
        """
        Get the best five cards from the hand and board.
        
        Args:
            board_str: Space-separated string of board cards
            
        Returns:
            List of best five cards
        """
        all_cards = self.cards.copy()
        
        if board_str:
            board = Hand(board_str)
            all_cards.extend(board.cards)
        
        # TODO: Implement logic to find the best 5 cards
        # This is a complex algorithm requiring evaluation of all possible 5-card combinations
        
        # For now, just return the highest 5 cards
        all_cards.sort(reverse=True)
        return all_cards[:5]
    
    def evaluate(self, board_str: Optional[str] = None) -> Tuple[str, List[Card], int]:
        """
        Evaluate the hand strength.
        
        Args:
            board_str: Space-separated string of board cards
            
        Returns:
            Tuple of (hand_name, best_five_cards, hand_strength)
        """
        all_cards = self.cards.copy()
        
        if board_str:
            board = Hand(board_str)
            all_cards.extend(board.cards)
        
        # Sort cards by rank (high to low)
        all_cards.sort(key=lambda x: x.rank_value, reverse=True)
        
        # Group cards by rank and suit
        ranks = {}
        suits = {}
        
        for card in all_cards:
            if card.rank_value not in ranks:
                ranks[card.rank_value] = []
            ranks[card.rank_value].append(card)
            
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        
        # Check for flush
        flush = None
        for suit, cards in suits.items():
            if len(cards) >= 5:
                flush = sorted(cards, key=lambda x: x.rank_value, reverse=True)[:5]
                break
        
        # Check for straight
        straight = self._find_straight(all_cards)
        
        # Check for straight flush
        straight_flush = None
        if flush:
            straight_flush = self._find_straight(flush)
        
        # Royal flush
        if straight_flush and straight_flush[0].rank == 'A':
            return ('royal_flush', straight_flush, self.HAND_RANKS['royal_flush'])
        
        # Straight flush
        if straight_flush:
            return ('straight_flush', straight_flush, self.HAND_RANKS['straight_flush'])
        
        # Four of a kind
        for rank, cards in ranks.items():
            if len(cards) == 4:
                four_cards = cards
                kickers = [card for card in all_cards if card.rank_value != rank]
                return ('four_of_a_kind', four_cards + kickers[:1], self.HAND_RANKS['four_of_a_kind'])
        
        # Full house
        three_of_a_kind = None
        pairs = []
        
        for rank, cards in sorted(ranks.items(), key=lambda x: (len(x[1]), x[0]), reverse=True):
            if len(cards) == 3 and three_of_a_kind is None:
                three_of_a_kind = cards
            elif len(cards) >= 2:
                pairs.append(cards[:2])
        
        if three_of_a_kind and pairs:
            return ('full_house', three_of_a_kind + pairs[0], self.HAND_RANKS['full_house'])
        
        # Flush
        if flush:
            return ('flush', flush, self.HAND_RANKS['flush'])
        
        # Straight
        if straight:
            return ('straight', straight, self.HAND_RANKS['straight'])
        
        # Three of a kind
        if three_of_a_kind:
            kickers = [card for card in all_cards if card.rank_value != three_of_a_kind[0].rank_value][:2]
            return ('three_of_a_kind', three_of_a_kind + kickers, self.HAND_RANKS['three_of_a_kind'])
        
        # Two pair
        if len(pairs) >= 2:
            kickers = [card for card in all_cards 
                      if card.rank_value != pairs[0][0].rank_value 
                      and card.rank_value != pairs[1][0].rank_value][:1]
            return ('two_pair', pairs[0] + pairs[1] + kickers, self.HAND_RANKS['two_pair'])
        
        # Pair
        if len(pairs) == 1:
            kickers = [card for card in all_cards if card.rank_value != pairs[0][0].rank_value][:3]
            return ('pair', pairs[0] + kickers, self.HAND_RANKS['pair'])
        
        # High card
        return ('high_card', all_cards[:5], self.HAND_RANKS['high_card'])
    
    def _find_straight(self, cards: List[Card]) -> Optional[List[Card]]:
        """
        Find a straight in a list of cards.
        
        Args:
            cards: List of cards to check
            
        Returns:
            List of cards forming a straight or None
        """
        if len(cards) < 5:
            return None
        
        # Remove duplicate ranks
        ranks = {}
        for card in cards:
            if card.rank_value not in ranks or card.rank_value > ranks[card.rank_value].rank_value:
                ranks[card.rank_value] = card
        
        # Sort cards by rank
        unique_cards = sorted(ranks.values(), key=lambda x: x.rank_value, reverse=True)
        
        # Handle A-5 straight
        if (any(card.rank == 'A' for card in unique_cards) and
            any(card.rank == '5' for card in unique_cards) and
            any(card.rank == '4' for card in unique_cards) and
            any(card.rank == '3' for card in unique_cards) and
            any(card.rank == '2' for card in unique_cards)):
            
            # Find the cards for the A-5 straight
            ace = next(card for card in unique_cards if card.rank == 'A')
            five = next(card for card in unique_cards if card.rank == '5')
            four = next(card for card in unique_cards if card.rank == '4')
            three = next(card for card in unique_cards if card.rank == '3')
            two = next(card for card in unique_cards if card.rank == '2')
            
            return [five, four, three, two, ace]
        
        # Check for normal straight
        for i in range(len(unique_cards) - 4):
            if unique_cards[i].rank_value - unique_cards[i+4].rank_value == 4:
                return unique_cards[i:i+5]
        
        return None


def analyze_hand(hand_str: str, board_str: Optional[str] = None) -> Dict:
    """
    Analyze a poker hand to determine its strength.
    
    Args:
        hand_str: String representation of a hand (e.g., 'As Ks')
        board_str: String representation of the board cards (e.g., 'Qs Js Ts')
        
    Returns:
        Dictionary with hand analysis results
    """
    try:
        hand = Hand(hand_str)
        hand_name, best_cards, strength = hand.evaluate(board_str)
        
        return {
            'hand': hand_str,
            'board': board_str if board_str else '',
            'hand_name': hand_name,
            'best_five': ' '.join(str(card) for card in best_cards),
            'strength': strength
        }
    except Exception as e:
        return {
            'hand': hand_str,
            'board': board_str if board_str else '',
            'error': str(e)
        }


def compare_hands(hand1_str: str, hand2_str: str, board_str: Optional[str] = None) -> int:
    """
    Compare two poker hands to determine which is stronger.
    
    Args:
        hand1_str: String representation of the first hand
        hand2_str: String representation of the second hand
        board_str: String representation of the board cards
        
    Returns:
        1 if hand1 is stronger, -1 if hand2 is stronger, 0 if equal
    """
    hand1 = Hand(hand1_str)
    hand2 = Hand(hand2_str)
    
    hand1_name, hand1_cards, hand1_strength = hand1.evaluate(board_str)
    hand2_name, hand2_cards, hand2_strength = hand2.evaluate(board_str)
    
    if hand1_strength > hand2_strength:
        return 1
    if hand2_strength > hand1_strength:
        return -1
    
    # If hand types are equal, compare the card values
    for card1, card2 in zip(hand1_cards, hand2_cards):
        if card1.rank_value > card2.rank_value:
            return 1
        if card2.rank_value > card1.rank_value:
            return -1
    
    # Equal hands
    return 0
