"""
Download and prepare sample poker hand history files for testing.
"""

import os
import zipfile
import requests
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest_path: str) -> None:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Path to save the file to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)


def create_sample_hand_history(output_dir: str, num_files: int = 10) -> None:
    """
    Create sample hand history files for testing.
    
    Args:
        output_dir: Directory to save the files to
        num_files: Number of files to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample hand history template
    template = """PokerStars Hand #{hand_id}: Tournament #{tourn_id}, ${buyin}+${fee} USD Hold'em No Limit - Level {level} ({sb}/{bb}) - {date} {time}
Table '{table_id} {table_num}' {max_seats}-max Seat #{btn_seat} is the button
Seat 1: Player1 ({stack1} in chips) 
Seat 2: Player2 ({stack2} in chips) 
Seat 3: Player3 ({stack3} in chips) 
Player1: posts small blind {sb}
Player2: posts big blind {bb}
*** HOLE CARDS ***
Dealt to Player1 [{card1} {card2}]
Player3: folds 
Player1: calls {sb}
Player2: checks 
*** FLOP *** [{flop1} {flop2} {flop3}]
Player1: checks 
Player2: checks 
*** TURN *** [{flop1} {flop2} {flop3}] [{turn}]
Player1: checks 
Player2: bets {bet}
Player1: folds 
Uncalled bet ({bet}) returned to Player2
Player2 collected {pot} from pot
Player2: doesn't show hand 
*** SUMMARY ***
Total pot {pot} | Rake 0 
Board [{flop1} {flop2} {flop3} {turn}]
Seat 1: Player1 (small blind) folded on the Turn
Seat 2: Player2 (big blind) collected ({pot})
Seat 3: Player3 folded before Flop (didn't bet)
"""

    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    suits = ['s', 'h', 'd', 'c']
    
    # Generate a set of sample hand history files
    for i in range(1, num_files + 1):
        # Generate sample data for the template
        hand_id = 210000000 + i
        tourn_id = 2900000000 + i
        buyin = 0.25
        fee = 0.02
        level = i % 10 + 1
        sb = 10 * (level)
        bb = 20 * (level)
        date = "2023/01/15"
        time = f"{10 + i % 12}:{i % 60:02d}:00"
        table_id = 2900000000 + i
        table_num = i
        max_seats = 6 if i % 2 == 0 else 9
        btn_seat = (i % 3) + 1
        stack1 = 1500 - (i * 20)
        stack2 = 1200 + (i * 15)
        stack3 = 1800 - (i * 10)
        
        # Generate random cards
        import random
        card1 = f"{random.choice(ranks)}{random.choice(suits)}"
        card2 = f"{random.choice(ranks)}{random.choice(suits)}"
        flop1 = f"{random.choice(ranks)}{random.choice(suits)}"
        flop2 = f"{random.choice(ranks)}{random.choice(suits)}"
        flop3 = f"{random.choice(ranks)}{random.choice(suits)}"
        turn = f"{random.choice(ranks)}{random.choice(suits)}"
        
        bet = bb
        pot = sb + bb + bet
        
        # Format the template with the generated data
        hand_history = template.format(
            hand_id=hand_id,
            tourn_id=tourn_id,
            buyin=buyin,
            fee=fee,
            level=level,
            sb=sb,
            bb=bb,
            date=date,
            time=time,
            table_id=table_id,
            table_num=table_num,
            max_seats=max_seats,
            btn_seat=btn_seat,
            stack1=stack1,
            stack2=stack2,
            stack3=stack3,
            card1=card1,
            card2=card2,
            flop1=flop1,
            flop2=flop2,
            flop3=flop3,
            turn=turn,
            bet=bet,
            pot=pot
        )
        
        # Write the hand history to a file
        filename = os.path.join(output_dir, f"HH{tourn_id}-{hand_id}.txt")
        with open(filename, 'w') as f:
            f.write(hand_history)
    
    print(f"Created {num_files} sample hand history files in {output_dir}")


def download_kaggle_dataset(username: str, dataset_name: str, output_dir: str) -> None:
    """
    Download a dataset from Kaggle.
    
    Args:
        username: Kaggle username
        dataset_name: Name of the dataset
        output_dir: Directory to save the dataset to
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset {username}/{dataset_name}...")
        api.dataset_download_files(
            f"{username}/{dataset_name}",
            path=output_dir,
            unzip=True
        )
        
        print(f"Dataset downloaded to {output_dir}")
        
    except ImportError:
        print("Error: Kaggle API not found. Please install it with:")
        print("pip install kaggle")
        print("\nThen configure your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/<your_username>/account")
        print("2. Click on 'Create New API Token'")
        print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
        print("4. Run this script again")


def setup_demo_environment():
    """Set up a complete demo environment with sample data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    handhistory_dir = os.path.join(data_dir, "handhistory")
    output_dir = os.path.join(data_dir, "processed")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(handhistory_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Setting up poker data analysis environment...")
    
    # Create sample hand history files
    num_files = 20
    print(f"\nGenerating {num_files} sample hand history files...")
    create_sample_hand_history(handhistory_dir, num_files)
    
    # Option to download real dataset from Kaggle
    print("\nDo you want to download a real poker dataset from Kaggle? (y/n)")
    download_kaggle = input().strip().lower()
    
    if download_kaggle == 'y':
        # Try to download the dataset
        print("\nAttempting to download poker dataset from Kaggle...")
        download_kaggle_dataset(
            "murilogmamaral",  # Username for example dataset
            "online-poker-games",  # Dataset name
            data_dir
        )
    
    # Process the sample data
    from process_data import main as process_data
    import sys
    
    print("\nProcessing sample data...")
    sys.argv = [
        'process_data.py',
        '--directory', handhistory_dir,
        '--pattern', 'HH',
        '--output', os.path.join(output_dir, 'sample_result.csv'),
        '--login', 'Player1'
    ]
    process_data()
    
    print("\nDemo environment setup complete!")
    print(f"- Hand history files: {handhistory_dir}")
    print(f"- Processed data: {output_dir}")
    print("\nYou can now analyze this data using the provided tools:")
    print("- Run basic_analysis.py to see basic statistics")
    print("- Run visualize_data.py to generate visualizations")
    print("- Run analyze_range.py to analyze player hand ranges")


if __name__ == "__main__":
    print("Poker Data Analysis - Sample Data Setup")
    print("======================================")
    print("\nThis script will help you set up sample data for testing the poker data analysis tools.")
    print("\nOptions:")
    print("1. Set up complete demo environment (recommended)")
    print("2. Create sample hand history files only")
    print("3. Download real poker dataset from Kaggle (requires Kaggle account)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        setup_demo_environment()
    elif choice == '2':
        output_dir = input("Enter directory to save sample files: ").strip()
        num_files = int(input("Enter number of files to create: ").strip())
        create_sample_hand_history(output_dir, num_files)
    elif choice == '3':
        output_dir = input("Enter directory to save dataset: ").strip()
        username = input("Enter Kaggle dataset username (default: murilogmamaral): ").strip() or "murilogmamaral"
        dataset_name = input("Enter Kaggle dataset name (default: online-poker-games): ").strip() or "online-poker-games"
        download_kaggle_dataset(username, dataset_name, output_dir)
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")
