"""
Quick start script for poker data analysis.
This script will:
1. Create sample data
2. Process the data
3. Generate visualizations
4. Perform analysis
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def main():
    """Run a quick start demo."""
    print("========== PokerData Analysis Quick Start ==========")
    print("\nThis script will guide you through a complete demo of the poker data analysis package.")
    
    # Create directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    hand_history_dir = data_dir / "handhistory"
    output_dir = data_dir / "output"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    hand_history_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    print("\nStep 1: Generating sample hand history data...")
    
    # Check if download_sample_data.py module exists
    if not (base_dir / "download_sample_data.py").exists():
        print("Error: Cannot find download_sample_data.py")
        print("Please make sure you are running this script from the correct directory.")
        return 1
    
    # Import the create_sample_hand_history function
    sys.path.insert(0, str(base_dir))
    try:
        from download_sample_data import create_sample_hand_history
        create_sample_hand_history(str(hand_history_dir), num_files=30)
    except Exception as e:
        print(f"Error generating sample data: {str(e)}")
        return 1
    
    print("\nStep 2: Processing hand history files...")
    
    csv_output_path = output_dir / "poker_data.csv"
    
    # Run process_data.py
    process_cmd = [
        sys.executable,
        str(base_dir / "process_data.py"),
        "--directory", str(hand_history_dir),
        "--output", str(csv_output_path),
        "--login", "Player1",
        "--pattern", "HH"
    ]
    
    try:
        subprocess.run(process_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing data: {str(e)}")
        return 1
    
    print("\nStep 3: Generating visualizations...")
    
    vis_output_dir = output_dir / "visualizations"
    vis_output_dir.mkdir(exist_ok=True)
    
    # Run visualize_data.py
    visualize_cmd = [
        sys.executable,
        str(base_dir / "visualize_data.py"),
        "--data", str(csv_output_path),
        "--output", str(vis_output_dir)
    ]
    
    try:
        subprocess.run(visualize_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating visualizations: {str(e)}")
        # Continue anyway
    
    print("\nStep 4: Analyzing player ranges...")
    
    analysis_output_dir = output_dir / "analysis"
    analysis_output_dir.mkdir(exist_ok=True)
    
    # Run analyze_range.py
    analyze_cmd = [
        sys.executable,
        str(base_dir / "analyze_range.py"),
        "--data", str(csv_output_path),
        "--player", "Player1",
        "--output", str(analysis_output_dir / "Player1_analysis.json")
    ]
    
    try:
        subprocess.run(analyze_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing player ranges: {str(e)}")
        # Continue anyway
    
    print("\n========== Demo Complete ==========")
    print(f"\nThe following files have been created:")
    print(f"- Sample hand history files: {hand_history_dir}")
    print(f"- Processed data (CSV): {csv_output_path}")
    print(f"- Visualizations: {vis_output_dir}")
    print(f"- Player analysis: {analysis_output_dir}/Player1_analysis.json")
    
    print("\nWhat's next?")
    print("1. Explore the analysis and visualizations in the output directory")
    print("2. Try processing your own hand history files with process_data.py")
    print("3. Generate custom visualizations with visualize_data.py")
    print("4. Analyze specific players with analyze_range.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
