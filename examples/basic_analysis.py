"""
Example script demonstrating basic poker data analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pokerdata import PokerVisualizer, analyze_player_range

# Set path to your data
DATA_PATH = "../result.csv"

def run_basic_analysis():
    """Run a basic analysis on poker data."""
    # Make sure the data exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found")
        print("Please run process_data.py first to generate the dataset")
        return
    
    # Load the dataset
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} hand records")
    
    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Number of tournaments: {df['tourn_id'].nunique()}")
    print(f"Number of players: {df['name'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Calculate player performance
    print("\n=== Player Performance ===")
    player_stats = (df.groupby('name')
                    .agg({
                        'hand_id': 'count',
                        'balance': ['sum', 'mean']
                    })
                    .sort_values(('balance', 'sum'), ascending=False))
    
    player_stats.columns = ['hands_played', 'total_balance', 'avg_balance']
    print(player_stats.head(10))
    
    # Create a visualizer
    visualizer = PokerVisualizer(df)
    
    # Plot balance over time for top player
    top_player = player_stats.index[0]
    print(f"\nGenerating balance chart for top player: {top_player}")
    visualizer.plot_balance_over_time(player=top_player)
    
    # Plot position winrate
    print("\nGenerating position winrate chart")
    visualizer.plot_position_winrate()
    
    # Analyze hand ranges for a player
    player_to_analyze = top_player
    print(f"\n=== Hand Range Analysis for {player_to_analyze} ===")
    analysis = analyze_player_range(df, player_to_analyze)
    
    print(f"VPIP: {analysis['vpip']:.1f}%")
    print(f"PFR: {analysis['pfr']:.1f}%")
    print(f"3-Bet: {analysis['threeb']:.1f}%")
    
    # Print top hands
    print("\nTop hands:")
    for hand, count in list(analysis['top_hands'].items())[:5]:
        print(f"  {hand}: {count}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    run_basic_analysis()
