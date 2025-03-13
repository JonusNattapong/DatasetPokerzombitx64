import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

class PokerVisualizer:
    # Existing methods...

    def plot_game_table(self, players: list, cards: list, dealer_position: int, filename: Optional[str] = None):
        """
        Plot the game table with player positions and cards in 8-bit style.
        
        Args:
            players: List of player names and positions [(name, position), ...]
            cards: List of cards for each player [(card1, card2), ...]
            dealer_position: Position of the dealer button
            filename: Filename to save the plot
        """
        # Create an empty image for the table
        img = Image.new('RGB', (800, 600), color='green')
        draw = ImageDraw.Draw(img)
        
        # Draw the table as a circle
        draw.ellipse((100, 100, 700, 500), outline='white', width=5)
        
        # Define positions for players around the table
        positions = [
            (400, 100), (600, 200), (600, 400), (400, 500), (200, 400), (200, 200)
        ]
        
        # Draw players and their cards
        font = ImageFont.load_default()
        for idx, (name, pos) in enumerate(players):
            x, y = positions[pos]
            draw.text((x, y), name, fill='white', font=font)
            card1, card2 = cards[idx]
            draw.text((x, y + 20), f"{card1} {card2}", fill='white', font=font)
        
        # Draw dealer button
        dealer_x, dealer_y = positions[dealer_position]
        draw.ellipse((dealer_x - 10, dealer_y - 10, dealer_x + 10, dealer_y + 10), fill='red')
        
        # Save or show the image
        if filename:
            img.save(filename)
        else:
            img.show()

# Example usage
# visualizer = PokerVisualizer(df, save_dir='.')
# visualizer.plot_game_table([("Player1", 0), ("Player2", 1), ("Player3", 2)], [("As", "Kd"), ("Qc", "Jh"), ("7s", "8d")], dealer_position=0)