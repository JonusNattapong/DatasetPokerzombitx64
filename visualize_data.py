"""
Generate visualizations from poker data.
"""

import os
import argparse
import pandas as pd
from pokerdata import PokerVisualizer, generate_all_visualizations


def main():
    """Main function to generate poker data visualizations."""
    parser = argparse.ArgumentParser(description='Generate poker data visualizations.')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to the CSV file containing poker data')
    parser.add_argument('--output', '-o', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--player', '-p', type=str,
                       help='Filter visualizations by player name')
    parser.add_argument('--type', '-t', type=str, choices=['balance', 'actions', 'position', 'hands', 'all'],
                       default='all', help='Type of visualization to generate')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} records from {args.data}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Create visualizer
        visualizer = PokerVisualizer(df, save_dir=args.output)
        
        # Generate selected visualizations
        if args.type == 'all':
            generate_all_visualizations(args.data, args.output, args.player)
        elif args.type == 'balance':
            visualizer.plot_balance_over_time(player=args.player)
        elif args.type == 'actions':
            for stage in ['pre', 'flop', 'turn', 'river']:
                visualizer.plot_action_distribution(stage=stage, player=args.player)
        elif args.type == 'position':
            visualizer.plot_position_winrate(player=args.player)
        elif args.type == 'hands':
            visualizer.plot_hand_strength_distribution(player=args.player)
        
        print(f"Visualizations saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
