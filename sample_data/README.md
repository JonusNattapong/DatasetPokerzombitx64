# Sample Poker Hand History Data

This directory contains sample poker hand history files that can be used to test the poker data analysis tool.

## What's Included

1. **Sample Hand History Files** - Simplified hand history text files in PokerStars format
2. **Sample Dataset CSV** - Pre-processed dataset ready for analysis

## How to Use the Sample Data

### Option 1: Using the Sample Data Script

Run the `download_sample_data.py` script from the project root to automatically set up the sample data:

```bash
python download_sample_data.py
```

### Option 2: Using Your Own Data

If you have your own poker hand history files:

1. Create a directory to store your hand history files
2. Place your hand history files in that directory
3. Run the processing script:

```bash
python process_data.py --directory /path/to/hand/history --login YourUsername --output result.csv
```

## Format of Hand History Files

The sample hand history files follow the PokerStars format, which typically includes:

- Game information (hand ID, tournament ID, stakes, etc.)
- Table information (table name, max seats, button position)
- Player information (seat positions, stack sizes)
- Blinds and antes
- Hole cards
- Action for each betting round (preflop, flop, turn, river)
- Showdown results
- Summary information

## Example Hand History

```
PokerStars Hand #210000001: Tournament #2900000001, $0.25+$0.02 USD Hold'em No Limit - Level 1 (10/20) - 2023/01/15 11:01:00
Table '2900000001 1' 9-max Seat #1 is the button
Seat 1: Player1 (1480 in chips) 
Seat 2: Player2 (1215 in chips) 
Seat 3: Player3 (1790 in chips) 
Player1: posts small blind 10
Player2: posts big blind 20
*** HOLE CARDS ***
Dealt to Player1 [As 2h]
Player3: folds 
Player1: calls 10
Player2: checks 
*** FLOP *** [Kd 3s Qc]
Player1: checks 
Player2: checks 
*** TURN *** [Kd 3s Qc] [5h]
Player1: checks 
Player2: bets 20
Player1: folds 
Uncalled bet (20) returned to Player2
Player2 collected 40 from pot
Player2: doesn't show hand 
*** SUMMARY ***
Total pot 40 | Rake 0 
Board [Kd 3s Qc 5h]
Seat 1: Player1 (small blind) folded on the Turn
Seat 2: Player2 (big blind) collected (40)
Seat 3: Player3 folded before Flop (didn't bet)
```

## Data Dictionary for the Processed CSV

The processed CSV file will contain the following columns:

| Column | Description |
|--------|-------------|
| buyin | Amount paid to play the tournament |
| tourn_id | Tournament ID |
| table | Table number |
| hand_id | Hand ID |
| date | Date of the hand |
| time | Time of the hand |
| table_size | Maximum number of players |
| level | Blinds level |
| playing | Number of active players |
| seat | Player's seat number |
| name | Player's name |
| stack | Player's stack size |
| position | Player's position (BTN, SB, BB, etc.) |
| action_pre | Player's preflop actions |
| action_flop | Player's flop actions |
| action_turn | Player's turn actions |
| action_river | Player's river actions |
| all_in | Whether the player went all-in |
| cards | Player's hole cards |
| board_flop | Cards on the flop |
| board_turn | Card on the turn |
| board_river | Card on the river |
| combination | Final hand combination |
| pot_pre | Pot size after preflop |
| pot_flop | Pot size after the flop |
| pot_turn | Pot size after the turn |
| pot_river | Final pot size |
| ante | Ante paid |
| blinds | Blinds paid |
| bet_pre | Bets on preflop |
| bet_flop | Bets on the flop |
| bet_turn | Bets on the turn |
| bet_river | Bets on the river |
| result | Outcome (won, lost, gave up) |
| balance | Profit/loss for the hand |
