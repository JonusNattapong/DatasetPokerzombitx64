
def analyze_hand(hand_str: str, board_str: Optional[str] = None) -> Dict:
    """Analyze a poker hand."""
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
    """Compare two poker hands."""
    try:
        hand1 = Hand(hand1_str)
        hand2 = Hand(hand2_str)
        
        name1, cards1, strength1 = hand1.evaluate(board_str)
        name2, cards2, strength2 = hand2.evaluate(board_str)
        
        if strength1 != strength2:
            return 1 if strength1 > strength2 else -1
            
        # If hand types are equal (e.g., both straights), compare each card
        for card1, card2 in zip(cards1, cards2):
            if card1.rank_value != card2.rank_value:
                return 1 if card1.rank_value > card2.rank_value else -1
        
        return 0
    except:
        return 0
