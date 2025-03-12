"""
Parser module for processing poker hand history files.
"""

import os
import re
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any


def load_data(pattern: str, directory: str = './') -> List[List[Dict]]:
    """
    Load poker hand history files matching the given pattern.
    
    Args:
        pattern: String pattern to match files
        directory: Directory to search in
        
    Returns:
        Structured hand history data
    """
    files = glob.glob(os.path.join(directory, f"*{pattern}*"))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {directory}")
    
    full_list = read_hh(files)
    loaded = split_hh(full_list)
    return loaded


def read_hh(files: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Read hand history files.
    
    Args:
        files: List of file paths to read
        
    Returns:
        Dictionary with file names as keys and DataFrames as values
    """
    result = {}
    for file in files:
        try:
            # Read CSV with no header and ^ separator
            result[file] = pd.read_csv(file, header=None, sep='^')
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
    return result


def split_hh(hh: Dict[str, pd.DataFrame]) -> List[List[Dict]]:
    """
    Split hand history data into individual hands.
    
    Args:
        hh: Dictionary of hand history DataFrames
        
    Returns:
        Nested list of hand data
    """
    hands = []
    
    for i in range(len(hh) + 1):
        hands.append([])
    
    for i, (file, data) in enumerate(hh.items()):
        if data is None or data.empty:
            continue
            
        poker_indices = data[0].str.contains('PokerStars').fillna(False)
        if not poker_indices.any():
            continue
            
        poker_indices = poker_indices[poker_indices].index.tolist()
        
        for j in range(len(poker_indices)):
            start_idx = poker_indices[j]
            end_idx = poker_indices[j+1] - 1 if j < len(poker_indices) - 1 else len(data)
            
            hand_data = data.loc[start_idx:end_idx].copy()
            hands[i].append({'hand': hand_data})
    
    return hands


def build_dataset(unstructured_data: List[List[Dict]], mylogin: str, n_workers: int = os.cpu_count()) -> pd.DataFrame:
    """
    Build a structured dataset from the unstructured poker hand data.
    
    Args:
        unstructured_data: Nested list of hand data
        mylogin: Player's login name
        n_workers: Number of CPU workers for parallel processing
        
    Returns:
        DataFrame containing processed poker hand data
    """
    tasks = []
    for game_idx, game_data in enumerate(unstructured_data):
        for hand_idx, _ in enumerate(game_data):
            tasks.append((game_idx, hand_idx, unstructured_data, mylogin))
    
    # Use parallel processing to speed up data building
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(store_round, *zip(*tasks)))

    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    return df


def store_round(GAME: int, HAND: int, unstructured_data: List[List[Dict]], mylogin: str) -> pd.DataFrame:
    """
    Process and store a single poker hand round.
    
    Args:
        GAME: Game index
        HAND: Hand index
        unstructured_data: Nested list of hand data
        mylogin: Player's login name
        
    Returns:
        DataFrame with processed hand data
    """
    try:
        base = unstructured_data[GAME][HAND]['hand'][0].tolist()
        
        # Catching lines to filter data and make searches faster
        line_hole_cards = next((i for i, line in enumerate(base) if '*** HOLE CARDS ***' in line), -1)
        line_flop = next((i for i, line in enumerate(base) if '*** FLOP ***' in line), -1)
        line_turn = next((i for i, line in enumerate(base) if '*** TURN ***' in line), -1)
        line_river = next((i for i, line in enumerate(base) if '*** RIVER ***' in line), -1)
        line_final = next((i for i, line in enumerate(base) if '*** SUMMARY ***' in line), -1)
        line_show_down = next((i for i, line in enumerate(base) if '*** SHOW DOWN ***' in line), -1)
        
        if line_flop == -1:
            line_flop = line_final
        if line_turn == -1:
            line_turn = line_final
        if line_river == -1:
            line_river = line_final
            
        if line_hole_cards == -1:
            # Handle corrupt or incomplete data
            return pd.DataFrame()
        
        # Extract table information
        max_seats = int(chairs(base[1]))
        
        players = []
        seats = []
        stacks = []
        
        # Extract player information
        for i in range(1, max_seats + 1):
            seat_pattern = f"Seat {i}: "
            matching_lines = [line for line in base[1:line_hole_cards] if seat_pattern in line]
            
            if matching_lines:
                player_line = matching_lines[0]
                player = re.search(r"Seat \d+: (.+?) \(\d+ in chips", player_line)
                player = player.group(1) if player else "Unknown"
                players.append(player)
                seats.append(i)
                stacks.append(initial_stack(player_line))
        
        dt = get_date_time(base[0])
        date = dt[0]
        time = dt[1]
        
        # Calculate total pot from antes
        line_text_ante = [line for line in base[3:line_hole_cards] if " posts ante" in line]
        pot = [bets(line) for line in line_text_ante]
        total_pot = sum(pot)
        
        playing = len(players)
        n = len(players)
        
        # Initialize the DataFrame for this round
        df_round = pd.DataFrame({
            'buyin': [buyin(base[0])] * n,
            'tourn_id': [tourn_id(base[0])] * n,
            'table': [table_number(base[1])] * n,
            'hand_id': [hand(base[0])] * n,
            'date': [date] * n,
            'time': [time] * n,
            'table_size': [max_seats] * n,
            
            'level': [level(base[0])] * n,
            'playing': [playing] * n,
            
            'seat': seats,
            'name': players,
            'stack': stacks,
            'position': ['x'] * n,
            
            'action_pre': ["-"] * n,
            'action_flop': ["-"] * n,
            'action_turn': ["-"] * n,
            'action_river': ["-"] * n,
            'all_in': [False] * n,
            
            'cards': ["--"] * n,
            
            'board_flop': [get_board(base[line_flop]) if line_flop >= 0 else "0"] * n,
            'board_turn': [get_board(base[line_turn]) if line_turn >= 0 else "0"] * n,
            'board_river': [get_board(base[line_river]) if line_river >= 0 else "0"] * n,
            'combination': [""] * n,
            
            'pot_pre': [total_pot] * n,
            'pot_flop': [0] * n,
            'pot_turn': [0] * n,
            'pot_river': [0] * n,
            
            'ante': [0] * n,
            'blinds': [0] * n,
            
            'bet_pre': [0] * n,
            'bet_flop': [0] * n,
            'bet_turn': [0] * n,
            'bet_river': [0] * n,
            
            'result': [0] * n,
            'balance': [0] * n
        })
        
        # Get my hand
        my_idx = next((i for i, p in enumerate(players) if p == mylogin), None)
        if my_idx is not None:
            my_hand_lines = [line for line in base[line_hole_cards:line_hole_cards+3] if f"Dealt to {mylogin}" in line]
            if my_hand_lines:
                my_hand = get_cards(my_hand_lines[0])
                df_round.at[my_idx, 'cards'] = my_hand
        
        # Get show down cards
        for p_idx, (player, seat_num) in enumerate(zip(players, seats)):
            if line_final > 0:
                show_down_lines = [line for line in base[line_final:] if f"Seat {seat_num}:" in line]
                if show_down_lines and ("mucked" in show_down_lines[0] or "showed" in show_down_lines[0]):
                    df_round.at[p_idx, 'cards'] = get_cards(show_down_lines[0])
                    df_round.at[p_idx, 'combination'] = get_combination(show_down_lines[0])
        
        # Set button position
        button = get_button(base[1])
        
        # Process initial actions
        for j, (player, seat_num) in enumerate(zip(players, seats)):
            # Check for ante
            ante_lines = [line for line in base[3:line_hole_cards] if line.startswith(f"{player}: posts ante")]
            if ante_lines:
                df_round.at[j, 'ante'] = bets(ante_lines[0])
            
            # Set position
            if button == seat_num:
                df_round.at[j, 'position'] = "BTN"
            
            # Check for small blind
            sb_lines = [line for line in base[3:line_hole_cards] if line.startswith(f"{player}: posts small")]
            if sb_lines:
                df_round.at[j, 'position'] = "SB"
                df_round.at[j, 'blinds'] = bets(sb_lines[0])
            
            # Check for big blind
            bb_lines = [line for line in base[3:line_hole_cards] if line.startswith(f"{player}: posts big")]
            if bb_lines:
                df_round.at[j, 'position'] = "BB"
                df_round.at[j, 'blinds'] = bets(bb_lines[0])
            
            all_lines = [line for line in base[3:line_hole_cards] if line.startswith(player)]
            if any("all-in" in line for line in all_lines):
                df_round.at[j, 'all_in'] = True
        
        # Process actions for each betting round
        rounds = [
            ('pre', line_hole_cards, line_flop, 'action_pre', 'bet_pre', 'blinds'),
            ('flop', line_flop, line_turn, 'action_flop', 'bet_flop', None),
            ('turn', line_turn, line_river, 'action_turn', 'bet_turn', None),
            ('river', line_river, line_final, 'action_river', 'bet_river', None)
        ]
        
        for round_name, start_line, end_line, action_col, bet_col, extra_col in rounds:
            for j, player in enumerate(players):
                filtered = base[start_line:end_line]
                lines = [line for line in filtered if line.startswith(f"{player}:")]
                
                if any("all-in" in line for line in lines):
                    df_round.at[j, 'all_in'] = True
                
                if lines:
                    actions = []
                    bets_made = 0
                    
                    for line in lines:
                        action_match = re.search(r": (calls|raises|checks|folds|bets|shows|mucks|doesn't)", line)
                        action = action_match.group(1) if action_match else "ERROR"
                        actions.append(action)
                        
                        if "raises" in line:
                            bets_made = bets(line)
                        else:
                            bets_made += bets(line)
                    
                    action_str = "-".join(actions)
                    df_round.at[j, action_col] = action_str
                    df_round.at[j, bet_col] = bets_made
                    
                    if extra_col and df_round.at[j, extra_col] > 0:
                        df_round.at[j, bet_col] += df_round.at[j, extra_col]
                else:
                    df_round.at[j, action_col] = "x"
        
        # Process results
        for j, player in enumerate(players):
            df_round.at[j, 'result'] = "gave up"
            
            # Calculate initial balance
            total_bet = (df_round.at[j, 'ante'] + 
                         df_round.at[j, 'bet_pre'] + 
                         df_round.at[j, 'bet_flop'] + 
                         df_round.at[j, 'bet_turn'] + 
                         df_round.at[j, 'bet_river'])
            df_round.at[j, 'balance'] = -total_bet
            
            # Check if player collected chips
            if line_show_down <= 0:
                collected_lines = [line for line in base[line_hole_cards:line_final] 
                                if f"{player} collected" in line]
                
                if collected_lines:
                    collected_amount = sum(bets(line) for line in collected_lines)
                    df_round.at[j, 'result'] = "took chips"
                    df_round.at[j, 'balance'] = collected_amount
            else:
                collected_lines = [line for line in base[line_hole_cards:line_final] 
                                if f"{player} collected" in line]
                uncalled_lines = [line for line in base[line_hole_cards:line_final] 
                                if f"returned to {player}" in line]
                
                if collected_lines:
                    collected_amount = sum(bets(line) for line in collected_lines)
                    returned_amount = sum(uncalled(line) for line in uncalled_lines)
                    
                    df_round.at[j, 'result'] = "won"
                    df_round.at[j, 'balance'] = collected_amount - (total_bet - returned_amount)
            
            # Check for showdown loss
            river_action = df_round.at[j, 'action_river']
            if any(term in river_action for term in ["doesn't", "mucks", "shows"]) and df_round.at[j, 'result'] == "gave up":
                df_round.at[j, 'result'] = "lost"
                
                uncalled_lines = [line for line in base[line_hole_cards:line_final] if f"returned to {player}" in line]
                returned_amount = sum(uncalled(line) for line in uncalled_lines)
                
                df_round.at[j, 'balance'] = -total_bet + returned_amount
        
        # Adjust balances for "took chips" players
        took_chips_idx = df_round.index[df_round['result'] == "took chips"].tolist()
        if took_chips_idx:
            total_balance = df_round['balance'].sum()
            for idx in took_chips_idx:
                df_round.at[idx, 'balance'] -= total_balance
        
        # Calculate pot sizes for each round
        df_round['pot_pre'] = df_round['bet_pre'].sum()
        
        if df_round.at[0, 'board_flop'] != "-":
            df_round['pot_flop'] = df_round['bet_flop'].sum() + df_round['bet_pre'].sum()
        
        if df_round.at[0, 'board_turn'] != "-":
            df_round['pot_turn'] = (df_round['bet_turn'].sum() + 
                                  df_round['bet_flop'].sum() + 
                                  df_round['bet_pre'].sum())
        
        if df_round.at[0, 'board_river'] != "-":
            df_round['pot_river'] = (df_round['bet_river'].sum() + 
                                   df_round['bet_turn'].sum() + 
                                   df_round['bet_flop'].sum() + 
                                   df_round['bet_pre'].sum())
        
        return df_round
    
    except Exception as e:
        print(f"Error processing game {GAME}, hand {HAND}: {str(e)}")
        return pd.DataFrame()


# Helper functions for extracting information from hand history lines

def tourn_id(line: str) -> int:
    """Extract tournament ID"""
    match = re.search(r"Hand #\d+: Tournament #(\d+)", line)
    return int(match.group(1)) if match else 0
def tourn_id(line: str) -> int:
    """Extract tournament ID"""
    match = re.search(r"Hand #\d+: Tournament #(\d+)", line)
    return int(match.group(1)) if match else 0
def tourn_id(line: str) -> int:
    """Extract tournament ID"""
    match = re.search(r"Hand #\d+: Tournament #(\d+)", line)
    return int(match.group(1)) if match else 0
def tourn_id(line: str) -> int:
    """Extract tournament ID"""
    match = re.search(r"Hand #\d+: Tournament #(\d+)", line)
    return int(match.group(1)) if match else 0


def table_number(line: str) -> str:
    """Extract table number"""
    match = re.search(r"'(.+)'", line)
    if match:
        parts = match.group(1).split()
        return parts[1] if len(parts) > 1 else parts[0]
    return ""


def hand(line: str) -> int:
    """Extract hand ID"""
    match = re.search(r'#(\d+):', line)
    return int(match.group(1)) if match else 0


def chairs(line: str) -> int:
    """Extract maximum number of chairs"""
    match = re.search(r'(\d+)-max', line)
    return int(match.group(1)) if match else 9


def get_board(line: str) -> str:
    """Extract board cards"""
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")
def get_board(line: str) -> str:
    """Extract board cards"""
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")
def get_board(line: str) -> str:
    """Extract board cards"""
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")
def get_board(line: str) -> str:
    """Extract board cards"""
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")
def get_board(line: str) -> str:
    """Extract board cards"""
    if "*** TURN ***" in line:
        match = re.search(r"\[.*?\] \[(.*?)\]", line)
        return match.group(1) if match else "-"
    else:
        match = re.search(r"\[(.*?)\]", line)
        return match.group(1) if match else ("0" if "*** SUMMARY ***" in line else "-")
    return "0" if "*** SUMMARY ***" in line else "-"


def get_combination(line: str) -> str:
    """Extract hand combination"""
    match = re.search(r"\(([^)]+)\)", line)
    return match.group(1) if match else ""
def get_combination(line: str) -> str:
    """Extract hand combination"""
    match = re.search(r"\(([^)]+)\)", line)
    return match.group(1) if match else ""
def get_combination(line: str) -> str:
    """Extract hand combination"""
    match = re.search(r"\(([^)]+)\)", line)
    return match.group(1) if match else ""
def get_combination(line: str) -> str:
    """Extract hand combination"""
    match = re.search(r"\(([^)]+)\)", line)
    return match.group(1) if match else ""
    match = re.search(r' with (.*)', line)
    return match.group(1) if match else ""


def buyin(line: str) -> str:
    """Extract buy-in amount"""
    match = re.search(r'\$([\d\-.]+)\+\$([\d\-.]+)|[\d\-.]+\+[\d\-.]+', line)
    return match.group(0).replace("$", "") if match else "---"


def get_date_time(line: str) -> Tuple[str, str]:
    """Extract date and time"""
    date_match = re.search(r'(\d{4}/\d{2}/\d{2})', line)
    time_match = re.search(r'(\d+:\d+:\d+)', line)
    
    date = date_match.group(1) if date_match else "---"
    time = time_match.group(1) if time_match else "---"
    
    return date, time


def get_button(line: str) -> int:
    """Extract button position"""
    match = re.search(r'#(\d{1})', line)
    return int(match.group(1)) if match else 0


def level(line: str) -> int:
    """Extract tournament level"""
    match = re.search(r'Level ([IVXLC]+)', line)
    if match:
        roman = match.group(1)
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        result = 0
        for i in range(len(roman)):
            if i > 0 and roman_values[roman[i]] > roman_values[roman[i-1]]:
                result += roman_values[roman[i]] - 2 * roman_values[roman[i-1]]
            else:
                result += roman_values[roman[i]]
        return result
    return 0


def get_cards(line: str) -> str:
    """Extract cards"""
    match = re.search(r'\[(.*?)\]', line)
    return match.group(1) if match else "[---]"


def initial_stack(line: str) -> int:
    """Extract initial stack size"""
    match = re.search(r'(\d+) in chips', line)
    return int(match.group(1)) if match else 0


def bets(line: str) -> float:
    """Extract bet amount"""
    numbers = re.findall(r'\b\d+\b', line)
    if numbers:
        return float(numbers[-1])
    return 0


def uncalled(line: str) -> float:
    """Extract uncalled bet amount"""
    match = re.search(r'\((\d+)\)', line)
    return float(match.group(1)) if match else 0
