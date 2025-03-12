"""
Analyze player hand ranges from poker data.
"""

import os
import argparse
import pandas as pd
import json
from pokerdata import analyze_player_range, HandRange


def main():
    """Main function to analyze player hand ranges."""
    parser = argparse.ArgumentParser(description='Analyze player hand ranges from poker data.')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to the CSV file containing poker data')
    parser.add_argument('--player', '-p', type=str, required=True,
                       help='Player name to analyze')
    parser.add_argument('--position', '-pos', type=str,
                       choices=['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO'],
                       help='Filter by position')
    parser.add_argument('--action', '-a', type=str,
                       choices=['raises', 'calls', 'folds', 'checks'],
                       help='Filter by action type')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save results as JSON')
    parser.add_argument('--equity', '-e', type=str,
                       help='Calculate equity against a hand range (e.g., "AA,KK,QQ")')
    parser.add_argument('--board', '-b', type=str,
                       help='Board cards for equity calculation (e.g., "As Ks Qs")')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} records from {args.data}")
        
        # Analyze player range
        analysis = analyze_player_range(df, args.player, args.position, args.action)
        
        # Calculate equity if requested
        if args.equity and 'hand_frequencies' in analysis:
            player_range = HandRange(','.join(analysis['hand_frequencies'].keys()))
            opponent_range = HandRange(args.equity)
            
            print(f"Calculating equity against range: {args.equity}")
            equity = player_range.get_equity_vs_range(opponent_range, args.board or '')
            analysis['equity_vs_range'] = {
                'opponent_range': args.equity,
                'board': args.board or '',
                'equity': equity
            }
        
        # Print results
        print(f"\nAnalysis for player: {args.player}")
        print(f"Position: {args.position or 'all'}")
        print(f"Action: {args.action or 'all'}")
        print(f"Total hands: {analysis['total_hands']}")
        print(f"Known hands: {analysis['known_hands']}")
        
        if 'vpip' in analysis:
            print(f"VPIP: {analysis['vpip']:.1f}%")
            print(f"PFR: {analysis['pfr']:.1f}%")
            print(f"3-Bet: {analysis['threeb']:.1f}%")
            
            print("\nHand category frequencies:")
            for category, freq in analysis['category_frequencies'].items():
                print(f"  {category}: {freq*100:.1f}%")
            
            print("\nTop hands:")
            for hand, count in analysis['top_hands'].items():
                freq = count / analysis['known_hands'] * 100
                print(f"  {hand}: {count} ({freq:.1f}%)")
                
            if 'equity_vs_range' in analysis:
                equity = analysis['equity_vs_range']['equity']
                opp_range = analysis['equity_vs_range']['opponent_range']
                board = analysis['equity_vs_range']['board']
                print(f"\nEquity vs {opp_range} on board {board or 'none'}: {equity:.1f}%")
        
        # Save to file if output path is provided
        if args.output:
            # Convert numpy values to native Python types for JSON serialization
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # Convert numpy types
                    return obj.item()
                return obj
            
            save_data = convert_to_native_types(analysis)
            
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\nAnalysis saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
