# Pokerzombitx64 Dataset

## Overview
The Pokerzombitx64 dataset is a comprehensive collection of poker game data designed for analysis, machine learning, and strategy development. This repository contains structured poker hand histories, player statistics, and annotated game situations.

## Dataset Structure
```
DatasetPokerzombitx64/
├── raw_data/       # Unprocessed poker hand histories
├── processed_data/ # Cleaned and formatted data ready for analysis
├── metadata/       # Information about the dataset collection process
├── examples/       # Example usage and analysis scripts
└── docs/           # Additional documentation and explanations
```

## Data Format
Each poker hand is stored in a structured format that includes:
- Hand ID and timestamp
- Game type, stakes, and table information
- Player positions and stack sizes
- Complete action sequence (bets, raises, calls, folds)
- Showdown results when applicable

## Usage
This dataset can be used for:
- Developing poker strategy algorithms
- Training machine learning models for decision making
- Analyzing player tendencies and game patterns
- Simulating poker scenarios and outcomes


![images](images/image.png)


![csv](images/image0.png)



## Getting Started
```python
# Example code to load and analyze the dataset
import pandas as pd

# Load hand history data
hands = pd.read_csv('processed_data/hands.csv')

# Basic statistics
print(f"Total hands: {len(hands)}")
print(f"Average pot size: ${hands['pot_size'].mean():.2f}")
```

## Contributing
Contributions to expand and improve the dataset are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of your changes

## License
This dataset is available under [specify license type] - see LICENSE file for details.

## Contact
For questions or feedback about this dataset, please contact [your contact information].

## Acknowledgements
Special thanks to [mention any contributors or data sources if applicable].
