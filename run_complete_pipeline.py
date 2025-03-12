"""
Run the complete poker data analysis pipeline.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run the complete poker data analysis pipeline."""
    parser = argparse.ArgumentParser(description='Run the complete poker data analysis pipeline.')
    
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                       help='Directory containing hand history files')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                       help='Directory to save output files')
    parser.add_argument('--login', '-l', type=str, required=True,
                       help='Your poker login name')
    parser.add_argument('--pattern', '-p', type=str, default='',
                       help='Pattern to match in filenames')
    parser.add_argument('--anonymize', '-a', action='store_true',
                       help='Whether to anonymize player names')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Whether to generate visualizations')
    parser.add_argument('--analyze', action='store_true',
                       help='Whether to perform hand range analysis')
    parser.add_argument('--db', action='store_true',
                       help='Whether to store data in a database')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Process the data
    print("Step 1: Processing hand history data...")
    csv_output_path = os.path.join(args.output_dir, 'poker_data.csv')
    
    process_cmd = [
        sys.executable, 'process_data.py',
        '--directory', args.input_dir,
        '--output', csv_output_path,
        '--login', args.login
    ]
    
    if args.pattern:
        process_cmd.extend(['--pattern', args.pattern])
    
    if args.anonymize:
        process_cmd.append('--anonymize')
    
    try:
        subprocess.run(process_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error processing data. Exiting pipeline.")
        return 1
    
    # Step 2: Store in database if requested
    if args.db:
        print("\nStep 2: Storing data in database...")
        db_path = os.path.join(args.output_dir, 'poker_data.db')
        
        import_cmd = [
            sys.executable, '-c',
            f"from pokerdata import import_csv_to_db; import_csv_to_db('{csv_output_path}', '{db_path}')"
        ]
        
        try:
            subprocess.run(import_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to store data in database. Continuing pipeline.")
    else:
        print("\nStep 2: Skipping database storage.")
    
    # Step 3: Generate visualizations if requested
    if args.visualize:
        print("\nStep 3: Generating visualizations...")
        vis_output_dir = os.path.join(args.output_dir, 'visualizations')
        
        visualize_cmd = [
            sys.executable, 'visualize_data.py',
            '--data', csv_output_path,
            '--output', vis_output_dir
        ]
        
        try:
            subprocess.run(visualize_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to generate visualizations. Continuing pipeline.")
    else:
        print("\nStep 3: Skipping visualization generation.")
    
    # Step 4: Perform player analysis if requested
    if args.analyze:
        print("\nStep 4: Performing player analysis...")
        analysis_output_dir = os.path.join(args.output_dir, 'analysis')
        os.makedirs(analysis_output_dir, exist_ok=True)
        
        # Get the top 5 players with the most hands
        import pandas as pd
        df = pd.read_csv(csv_output_path)
        top_players = df['name'].value_counts().head(5).index.tolist()
        
        if not top_players:
            print("No players found in the dataset. Skipping player analysis.")
        else:
            print(f"Analyzing top {len(top_players)} players...")
            
            for player in top_players:
                print(f"Analyzing player: {player}")
                analysis_output_file = os.path.join(analysis_output_dir, f"{player}_analysis.json")
                
                analyze_cmd = [
                    sys.executable, 'analyze_range.py',
                    '--data', csv_output_path,
                    '--player', player,
                    '--output', analysis_output_file
                ]
                
                try:
                    subprocess.run(analyze_cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"Warning: Failed to analyze player {player}. Continuing with next player.")
    else:
        print("\nStep 4: Skipping player analysis.")
    
    # Final step: Run basic analysis
    print("\nFinal step: Running basic analysis...")
    
    basic_analysis_cmd = [
        sys.executable, 'examples/basic_analysis.py'
    ]
    
    # Temporarily modify the DATA_PATH variable in the basic_analysis.py script
    with open('examples/basic_analysis.py', 'r') as f:
        basic_analysis_script = f.readlines()
    
    for i, line in enumerate(basic_analysis_script):
        if line.strip().startswith('DATA_PATH ='):
            basic_analysis_script[i] = f'DATA_PATH = "{csv_output_path}"\n'
            break
    
    with open('examples/basic_analysis.py', 'w') as f:
        f.writelines(basic_analysis_script)
    
    # Run the basic analysis
    try:
        subprocess.run(basic_analysis_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to run basic analysis.")
    
    # Restore the original DATA_PATH
    with open('examples/basic_analysis.py', 'w') as f:
        for i, line in enumerate(basic_analysis_script):
            if line.strip().startswith('DATA_PATH ='):
                f.write('DATA_PATH = "../result.csv"\n')
            else:
                f.write(line)
    
    print("\nPipeline complete!")
    print(f"- Processed data saved to: {csv_output_path}")
    if args.db:
        print(f"- Database saved to: {db_path}")
    if args.visualize:
        print(f"- Visualizations saved to: {vis_output_dir}")
    if args.analyze:
        print(f"- Player analysis saved to: {analysis_output_dir}")
    
    print("\nThank you for using the PokerData Analysis tools!")


if __name__ == "__main__":
    sys.exit(main())
