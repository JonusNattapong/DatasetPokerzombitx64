"""
Process poker hand history files and build a structured dataset.
"""

import os
import argparse
import pandas as pd
from pokerdata import load_data, build_dataset, anonymize_dataset
from pokerdata.parser import read_hh
from typing import Dict


def main():
    """Main function to process poker hand history data."""
    parser = argparse.ArgumentParser(description='Process poker hand history files.')

    parser.add_argument('--directory', '-d', type=str, default='./handhistory',
                        help='Directory containing hand history files')
    parser.add_argument('--pattern', '-p', type=str, default='hand',
                        help='Pattern to match in filenames')
    parser.add_argument('--output_csv', '-c', type=str, default='./result.csv',
                        help='Output CSV file path')
    parser.add_argument('--output_txt', '-t', type=str, default='./result.txt',
                        help='Output TXT file path')
    parser.add_argument('--login', '-l', type=str, required=True,
                        help='Your poker login name')
    parser.add_argument('--anonymize', '-a', action='store_true',
                        help='Whether to anonymize player names')
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help='Number of worker processes for parallelization')

    args = parser.parse_args()

    print(f"Loading data from {args.directory} with pattern '{args.pattern}'...")

    # Set working directory
    original_dir = os.getcwd()
    if args.directory:  # Only change directory if it's not empty
        os.chdir(args.directory)

    try:
        # Load unstructured data
        data = load_data(args.pattern)
        # Get raw hand history
        files = [f for f in os.listdir('.') if args.pattern in f]
        raw_data = read_hh(files)


        print(f"Building dataset with {args.workers} worker processes...")
        # Build the dataset
        df = build_dataset(data, args.login, args.workers)

        # Filter out rows with missing player names
        df = df.dropna(subset=['name'])

        # Restore original directory before saving
        os.chdir(original_dir)

        # Anonymize data if requested
        if args.anonymize:
            print("Anonymizing player names...")
            df = anonymize_dataset(df)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)), exist_ok=True)


        # Export the dataset to CSV
        print(f"Saving dataset to {args.output_csv}...")
        df.to_csv(args.output_csv, index=False)

        # Export raw data to TXT, organized by session
        print(f"Saving raw data to {args.output_txt}...")
        with open(args.output_txt, 'w') as txt_file:
            for file_name, file_df in raw_data.items():
                # Assuming each file represents a session
                txt_file.write(f"=== Session: {file_name} ===\n")
                for _, row in file_df.iterrows():
                    txt_file.write(row[0] + '\n')  # Write each line
                txt_file.write("\n") # Separate sessions


        print(f"Done! Processed {len(df)} hand records.")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Make sure we restore the original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
