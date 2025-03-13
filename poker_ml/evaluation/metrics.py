"""
Metrics module for poker strategy evaluation.
This module provides various metrics for evaluating poker performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_roi(data: pd.DataFrame) -> float:
    """
    Calculate Return on Investment (ROI).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        ROI as a percentage
    """
    required_cols = ['balance', 'ante', 'blinds']
    if not all(col in data.columns for col in required_cols):
        logger.warning("Missing required columns for ROI calculation")
        return 0.0
    
    # Handle null values by filling with 0
    data_clean = data[required_cols].fillna(0)
    
    total_profit = data_clean['balance'].sum()
    total_investment = (data_clean['ante'].sum() + data_clean['blinds'].sum())
    
    if total_investment == 0:
        logger.warning("Total investment is zero, cannot calculate ROI")
        return 0.0
    
    roi = (total_profit / total_investment) * 100
    return roi

def calculate_win_rate(data: pd.DataFrame, unit: str = 'bb/100') -> float:
    """
    Calculate win rate in big blinds per 100 hands or dollars per hour.
    
    Args:
        data: DataFrame containing poker hand data
        unit: Unit of measurement ('bb/100' or '$/hour')
        
    Returns:
        Win rate in specified units
    """
    if 'balance' not in data.columns:
        logger.warning("Missing 'balance' column for win rate calculation")
        return 0.0
    
    # Handle null values
    total_profit = data['balance'].fillna(0).sum()
    total_hands = len(data)
    
    if total_hands == 0:
        logger.warning("No hands found for win rate calculation")
        return 0.0
    
    if unit == 'bb/100':
        if 'blinds' not in data.columns:
            logger.warning("Missing 'blinds' column for bb/100 calculation")
            return 0.0
            
        # Calculate average big blind size, excluding null values
        avg_bb = data['blinds'].dropna().mean()
        if pd.isna(avg_bb) or avg_bb == 0:
            logger.warning("Cannot determine big blind size")
            return 0.0
        
        win_rate = (total_profit / avg_bb) * (100 / total_hands)
        return win_rate
    
    elif unit == '$/hour':
        if 'datetime' not in data.columns:
            logger.warning("Missing 'datetime' column for hourly win rate")
            return 0.0
        
        # Calculate total hours played
        try:
            data['datetime'] = pd.to_datetime(data['datetime'])
            total_seconds = (data['datetime'].max() - data['datetime'].min()).total_seconds()
            total_hours = total_seconds / 3600
        except Exception as e:
            logger.warning(f"Error calculating time difference: {e}")
            total_hours = 0
        
        if total_hours == 0:
            logger.warning("Total hours is zero, using hand count estimate")
            # Estimate 60 hands per hour
            total_hours = total_hands / 60
            
        if total_hours > 0:
            hourly_rate = total_profit / total_hours
        else:
            hourly_rate = 0.0
            
        return hourly_rate
    
    else:
        logger.warning(f"Unknown win rate unit: {unit}")
        return 0.0

def calculate_vpip(data: pd.DataFrame) -> float:
    """
    Calculate VPIP (Voluntarily Put Money In Pot).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        VPIP as a percentage
    """
    if 'action_pre' not in data.columns:
        logger.warning("Missing 'action_pre' column for VPIP calculation")
        return 0.0
    
    # Count hands where player voluntarily put money in pot (call, bet, raise)
    vpip_actions = ['call', 'bet', 'raise']
    vpip_hands = data['action_pre'].fillna('').apply(
        lambda x: any(action in str(x).lower() for action in vpip_actions)
    ).sum()
    
    total_hands = len(data)
    
    if total_hands == 0:
        logger.warning("No hands found for VPIP calculation")
        return 0.0
    
    vpip = (vpip_hands / total_hands) * 100
    return vpip

def calculate_pfr(data: pd.DataFrame) -> float:
    """
    Calculate PFR (Pre-Flop Raise).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        PFR as a percentage
    """
    if 'action_pre' not in data.columns:
        logger.warning("Missing 'action_pre' column for PFR calculation")
        return 0.0
    
    # Count hands where player raised preflop
    pfr_hands = data['action_pre'].fillna('').apply(
        lambda x: 'raise' in str(x).lower()
    ).sum()
    
    total_hands = len(data)
    
    if total_hands == 0:
        logger.warning("No hands found for PFR calculation")
        return 0.0
    
    pfr = (pfr_hands / total_hands) * 100
    return pfr

def calculate_af(data: pd.DataFrame) -> float:
    """
    Calculate AF (Aggression Factor).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Aggression Factor value
    """
    aggression_columns = ['action_pre', 'action_flop', 'action_turn', 'action_river']
    missing_cols = [col for col in aggression_columns if col not in data.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns for AF calculation: {missing_cols}")
        # Use only available columns
        aggression_columns = [col for col in aggression_columns if col in data.columns]
        
    if not aggression_columns:
        logger.warning("No action columns available for AF calculation")
        return 0.0
    
    # Initialize counters
    aggressive_actions = 0  # bets and raises
    passive_actions = 0     # calls
    
    # Count actions across all streets
    for col in aggression_columns:
        # Handle null values
        actions = data[col].fillna('')
        
        # Count raises and bets
        aggressive_actions += actions.apply(
            lambda x: str(x).lower().count('raise') + str(x).lower().count('bet')
        ).sum()
        
        # Count calls
        passive_actions += actions.apply(
            lambda x: str(x).lower().count('call')
        ).sum()
    
    # Calculate AF
    if passive_actions == 0:
        if aggressive_actions == 0:
            return 0.0  # No aggression detected
        else:
            return float('inf')  # All aggressive, no passive actions
    
    af = aggressive_actions / passive_actions
    return af

def calculate_wtsd(data: pd.DataFrame) -> float:
    """
    Calculate WTSD (Went To ShowDown) percentage.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        WTSD as a percentage
    """
    showdown_hands = 0
    
    # Try multiple methods to identify showdown hands
    if 'showdown' in data.columns:
        showdown_hands = data['showdown'].fillna(False).sum()
    elif 'combination' in data.columns:
        showdown_hands = data['combination'].fillna('').apply(
            lambda x: bool(str(x).strip())
        ).sum()
    elif all(col in data.columns for col in ['hole_cards', 'board_cards']):
        showdown_hands = data.apply(
            lambda row: bool(str(row['hole_cards']).strip()) and bool(str(row['board_cards']).strip()),
            axis=1
        ).sum()
    else:
        logger.warning("No reliable showdown indicators found")
        return 0.0
    
    # Count hands that saw the flop
    if 'action_flop' in data.columns:
        flop_hands = data['action_flop'].fillna('').apply(
            lambda x: bool(str(x).strip() and str(x).lower() != 'x')
        ).sum()
    else:
        logger.warning("No 'action_flop' column, using total hands for WTSD")
        flop_hands = len(data)
    
    if flop_hands == 0:
        logger.warning("No flop hands found for WTSD calculation")
        return 0.0
    
    wtsd = (showdown_hands / flop_hands) * 100
    return wtsd

def calculate_wsd(data: pd.DataFrame) -> float:
    """
    Calculate WSD (Won Money at ShowDown) percentage.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        WSD as a percentage
    """
    # Identify showdown hands
    showdown_mask = None
    
    if 'showdown' in data.columns:
        showdown_mask = data['showdown'].fillna(False)
    elif 'combination' in data.columns:
        showdown_mask = data['combination'].fillna('').apply(bool)
    elif all(col in data.columns for col in ['hole_cards', 'board_cards']):
        showdown_mask = data.apply(
            lambda row: bool(str(row['hole_cards']).strip()) and bool(str(row['board_cards']).strip()),
            axis=1
        )
    
    if showdown_mask is None or not showdown_mask.any():
        logger.warning("No showdown hands found for W$SD calculation")
        return 0.0
    
    showdown_data = data[showdown_mask].copy()
    
    # Ensure balance column exists and handle null values
    if 'balance' not in showdown_data.columns:
        logger.warning("Missing 'balance' column for W$SD calculation")
        return 0.0
    
    showdown_data['balance'] = showdown_data['balance'].fillna(0)
    
    # Count showdowns won (positive balance)
    showdowns_won = (showdown_data['balance'] > 0).sum()
    total_showdowns = len(showdown_data)
    
    if total_showdowns == 0:
        return 0.0
    
    w_sd = (showdowns_won / total_showdowns) * 100
    return w_sd

def calculate_3bet(data: pd.DataFrame) -> float:
    """
    Calculate 3-bet percentage.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        3-bet percentage
    """
    if 'action_pre' not in data.columns:
        logger.warning("Missing 'action_pre' column for 3-bet calculation")
        return 0.0
    
    # Handle null values
    actions = data['action_pre'].fillna('')
    
    # Count hands with opportunity to 3-bet (facing a raise)
    opportunity_hands = data[actions.apply(
        lambda x: 'raise' in str(x).lower() and '-' in str(x)  # Indicates action sequence
    )]
    
    if len(opportunity_hands) == 0:
        logger.warning("No 3-bet opportunities found")
        return 0.0
    
    # Count hands where player 3-bet (re-raised)
    threebet_hands = opportunity_hands[opportunity_hands['action_pre'].apply(
        lambda x: str(x).lower().count('raise') >= 2
    )]
    
    threebet_pct = (len(threebet_hands) / len(opportunity_hands)) * 100
    return threebet_pct

def calculate_fold_to_cbet(data: pd.DataFrame) -> float:
    """
    Calculate fold to continuation bet percentage.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Fold to C-bet percentage
    """
    if 'action_pre' not in data.columns or 'action_flop' not in data.columns:
        logger.warning("Missing action columns for fold to C-bet calculation")
        return 0.0
    
    # Handle null values
    data = data.copy()
    data['action_pre'] = data['action_pre'].fillna('')
    data['action_flop'] = data['action_flop'].fillna('')
    
    # Identify hands where player called preflop and faced a c-bet on flop
    preflop_caller = data[data['action_pre'].apply(
        lambda x: 'call' in str(x).lower() and 'raise' not in str(x).lower()
    )]
    
    # Hands where opponent bet on flop
    faced_cbet = preflop_caller[preflop_caller['action_flop'].apply(
        lambda x: 'bet' in str(x).lower() or 'raise' in str(x).lower()
    )]
    
    if len(faced_cbet) == 0:
        logger.warning("No instances of facing C-bet found")
        return 0.0
    
    # Count folds to c-bet
    folded_to_cbet = faced_cbet[faced_cbet['action_flop'].apply(
        lambda x: 'fold' in str(x).lower()
    )]
    
    fold_to_cbet_pct = (len(folded_to_cbet) / len(faced_cbet)) * 100
    return fold_to_cbet_pct

def calculate_cbet(data: pd.DataFrame) -> float:
    """
    Calculate C-bet (Continuation Bet) percentage.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        C-bet percentage
    """
    if 'action_pre' not in data.columns or 'action_flop' not in data.columns:
        logger.warning("Missing action columns for C-bet calculation")
        return 0.0
    
    # Handle null values
    data = data.copy()
    data['action_pre'] = data['action_pre'].fillna('')
    data['action_flop'] = data['action_flop'].fillna('')
    
    # Identify hands where player raised preflop and saw the flop
    preflop_raiser = data[data['action_pre'].apply(
        lambda x: 'raise' in str(x).lower()
    )]
    
    saw_flop = preflop_raiser[preflop_raiser['action_flop'].apply(
        lambda x: bool(str(x).strip() and str(x).lower() != 'x')
    )]
    
    if len(saw_flop) == 0:
        logger.warning("No C-bet opportunities found")
        return 0.0
    
    # Count continuation bets on flop
    cbet_hands = saw_flop[saw_flop['action_flop'].apply(
        lambda x: 'bet' in str(x).lower() or 'raise' in str(x).lower()
    )]
    
    cbet_pct = (len(cbet_hands) / len(saw_flop)) * 100
    return cbet_pct

def calculate_check_raise(data: pd.DataFrame, street: str = 'flop') -> float:
    """
    Calculate check-raise percentage for a specific street.
    
    Args:
        data: DataFrame containing poker hand data
        street: Street to analyze ('flop', 'turn', 'river')
        
    Returns:
        Check-raise percentage
    """
    action_col = f'action_{street}'
    
    if action_col not in data.columns:
        logger.warning(f"Missing '{action_col}' column for check-raise calculation")
        return 0.0
    
    # Handle null values
    actions = data[action_col].fillna('')
    
    # Count hands with opportunity to check-raise (player checked)
    check_opportunities = data[actions.apply(
        lambda x: 'check' in str(x).lower()
    )]
    
    if len(check_opportunities) == 0:
        logger.warning(f"No check-raise opportunities found on {street}")
        return 0.0
    
    # Count hands where player check-raised
    check_raises = check_opportunities[check_opportunities[action_col].apply(
        lambda x: 'check' in str(x).lower() and 'raise' in str(x).lower() and 
                 str(x).lower().index('check') < str(x).lower().index('raise')
    )]
    
    check_raise_pct = (len(check_raises) / len(check_opportunities)) * 100
    return check_raise_pct

def calculate_squeeze_opportunity(data: pd.DataFrame) -> float:
    """
    Calculate squeeze play percentage (3-bet after initial raise and at least one caller).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Squeeze play percentage
    """
    if 'action_pre' not in data.columns:
        logger.warning("Missing 'action_pre' column for squeeze calculation")
        return 0.0
    
    # Handle null values
    actions = data['action_pre'].fillna('')
    
    # Find hands where there was a raise and at least one call before player acted
    squeeze_opportunities = data[actions.apply(
        lambda x: str(x).lower().count('raise') >= 1 and 
                 str(x).lower().count('call') >= 1
    )]
    
    if len(squeeze_opportunities) == 0:
        logger.warning("No squeeze opportunities found")
        return 0.0
    
    # Count hands where player squeezed (raised after raise and call)
    squeeze_plays = squeeze_opportunities[squeeze_opportunities['action_pre'].apply(
        lambda x: 'raise' in str(x).lower() and 
                 str(x).lower().rindex('raise') > str(x).lower().find('call')
    )]
    
    squeeze_pct = (len(squeeze_plays) / len(squeeze_opportunities)) * 100
    return squeeze_pct

def calculate_fold_equity(data: pd.DataFrame) -> float:
    """
    Calculate fold equity (percentage of times a bet or raise causes opponents to fold).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Fold equity as a percentage
    """
    action_columns = ['action_pre', 'action_flop', 'action_turn', 'action_river']
    missing_cols = [col for col in action_columns if col not in data.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns for fold equity calculation: {missing_cols}")
        # Use only available columns
        action_columns = [col for col in action_columns if col in data.columns]
        
    if not action_columns:
        logger.warning("No action columns available for fold equity calculation")
        return 0.0
    
    # Handle null values
    data = data.copy()
    for col in action_columns:
        data[col] = data[col].fillna('')
    
    # Count aggressive actions that led to folds
    aggressive_actions = 0
    folds_after_aggression = 0
    
    for col in action_columns:
        aggressive_hands = data[data[col].apply(
            lambda x: 'bet' in str(x).lower() or 'raise' in str(x).lower()
        )]
        
        aggressive_actions += len(aggressive_hands)
        
        # Count folds after aggression
        folds_after_aggression += aggressive_hands[aggressive_hands[col].apply(
            lambda x: 'fold' in str(x).lower() and 
                     (str(x).lower().find('bet') < str(x).lower().find('fold') or 
                      str(x).lower().find('raise') < str(x).lower().find('fold'))
        )].shape[0]
    
    if aggressive_actions == 0:
        logger.warning("No aggressive actions found for fold equity calculation")
        return 0.0
    
    fold_equity = (folds_after_aggression / aggressive_actions) * 100
    return fold_equity

def calculate_positional_awareness(data: pd.DataFrame) -> float:
    """
    Calculate positional awareness score (higher PFR in late position vs early position).
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Positional awareness score (0 to 1, higher is better)
    """
    if 'position' not in data.columns or 'action_pre' not in data.columns:
        logger.warning("Missing required columns for positional awareness calculation")
        return 0.0
    
    # Handle null values
    data = data.copy()
    data['position'] = data['position'].fillna('')
    data['action_pre'] = data['action_pre'].fillna('')
    
    # Define position categories
    early_positions = ['UTG', 'UTG+1', 'UTG+2', 'EP']
    middle_positions = ['MP', 'MP+1', 'MP+2', 'HJ']
    late_positions = ['CO', 'BTN']
    
    # Calculate PFR by position category
    early_pfr = calculate_pfr(data[data['position'].isin(early_positions)])
    middle_pfr = calculate_pfr(data[data['position'].isin(middle_positions)])
    late_pfr = calculate_pfr(data[data['position'].isin(late_positions)])
    
    # Check for valid PFR values
    if early_pfr == 0 or np.isnan(early_pfr):
        early_pfr = 0.1  # Small non-zero value to avoid division by zero
    
    # Calculate ratios
    late_early_ratio = late_pfr / early_pfr if early_pfr > 0 else 0
    middle_early_ratio = middle_pfr / early_pfr if early_pfr > 0 else 0
    
    # Combine into a single score (normalized to 0-1)
    positional_score = min(1.0, (late_early_ratio * 0.7 + middle_early_ratio * 0.3) / 3.0)
    return positional_score

def calculate_hand_reading_accuracy(predicted_hands: List[str], actual_hands: List[str]) -> float:
    """
    Calculate hand reading accuracy.
    
    Args:
        predicted_hands: List of predicted hand ranges
        actual_hands: List of actual hands played
        
    Returns:
        Hand reading accuracy score (0 to 1)
    """
    if not predicted_hands or not actual_hands:
        logger.warning("Empty hand lists provided")
        return 0.0
        
    if len(predicted_hands) != len(actual_hands):
        logger.warning("Predicted and actual hand lists must have the same length")
        return 0.0
    
    correct = 0
    total = len(predicted_hands)
    
    for pred, actual in zip(predicted_hands, actual_hands):
        try:
            # Handle null values and empty strings
            if not pred or not actual:
                continue
                
            # Convert to string and normalize format
            pred_str = str(pred).upper().strip()
            actual_str = str(actual).upper().strip()
            
            # Compare hand ranges (this is a simplified check)
            if pred_str and actual_str and pred_str[0] == actual_str[0]:
                correct += 1
        except Exception as e:
            logger.warning(f"Error comparing hands: {e}")
            continue
    
    return correct / total if total > 0 else 0.0

def calculate_ev_difference(actual_action: str, optimal_action: str, 
                           action_evs: Dict[str, float]) -> float:
    """
    Calculate the EV (Expected Value) difference between actual and optimal action.
    
    Args:
        actual_action: The action taken
        optimal_action: The GTO optimal action
        action_evs: Dictionary of EV for each possible action
        
    Returns:
        EV difference (negative means suboptimal play)
    """
    if not actual_action or not optimal_action:
        logger.warning("Missing action information")
        return 0.0
        
    if not action_evs:
        logger.warning("Empty EV dictionary")
        return 0.0
        
    if actual_action not in action_evs or optimal_action not in action_evs:
        logger.warning(f"Missing EV data for actions: {actual_action}, {optimal_action}")
        return 0.0
    
    try:
        return action_evs[actual_action] - action_evs[optimal_action]
    except Exception as e:
        logger.warning(f"Error calculating EV difference: {e}")
        return 0.0

def calculate_decision_quality_score(decisions: List[Dict]) -> float:
    """
    Calculate overall decision quality score based on multiple decisions.
    
    Args:
        decisions: List of decision dictionaries, each containing:
                  - actual_action: The action taken
                  - optimal_action: The optimal action
                  - action_evs: Dictionary of EVs for each action
                  - pot_size: Size of the pot for this decision
        
    Returns:
        Decision quality score (0-100)
    """
    if not decisions:
        logger.warning("No decisions provided for quality scoring")
        return 0.0
    
    total_ev_diff = 0.0
    total_potential_ev = 0.0
    valid_decisions = 0
    
    for decision in decisions:
        try:
            actual = decision.get('actual_action')
            optimal = decision.get('optimal_action')
            evs = decision.get('action_evs', {})
            pot_size = float(decision.get('pot_size', 1.0))
            
            if not actual or not optimal or not evs:
                continue
            
            # Scale EV difference by pot size to weight important decisions more
            ev_diff = calculate_ev_difference(actual, optimal, evs) * pot_size
            
            # Calculate maximum potential EV gain
            max_ev = max(evs.values()) if evs else 0
            min_ev = min(evs.values()) if evs else 0
            potential_ev = (max_ev - min_ev) * pot_size if max_ev != min_ev else 1.0
            
            total_ev_diff += ev_diff
            total_potential_ev += potential_ev
            valid_decisions += 1
            
        except Exception as e:
            logger.warning(f"Error processing decision: {e}")
            continue
    
    if valid_decisions == 0 or total_potential_ev == 0:
        return 0.0
        
    # Calculate score as percentage of potential EV captured
    score = 100 * (1 - abs(total_ev_diff) / total_potential_ev)
    return max(0, min(100, score))  # Ensure score is between 0 and 100

def calculate_all_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate all available poker metrics.
    
    Args:
        data: DataFrame containing poker hand data
        
    Returns:
        Dictionary of calculated metrics
    """
    if data is None or data.empty:
        logger.warning("Empty or null DataFrame provided")
        return {}
        
    try:
        metrics = {
            'roi': calculate_roi(data),
            'win_rate_bb100': calculate_win_rate(data, 'bb/100'),
            'win_rate_hourly': calculate_win_rate(data, '$/hour'),
            'vpip': calculate_vpip(data),
            'pfr': calculate_pfr(data),
            'af': calculate_af(data),
            'wtsd': calculate_wtsd(data),
            'wsd': calculate_wsd(data),
            '3bet': calculate_3bet(data),
            'cbet': calculate_cbet(data),
            'fold_to_cbet': calculate_fold_to_cbet(data),
            'check_raise_flop': calculate_check_raise(data, 'flop'),
            'check_raise_turn': calculate_check_raise(data, 'turn'),
            'check_raise_river': calculate_check_raise(data, 'river'),
            'squeeze': calculate_squeeze_opportunity(data),
            'fold_equity': calculate_fold_equity(data),
            'positional_awareness': calculate_positional_awareness(data)
        }
        
        # Remove any metrics that returned NaN values
        metrics = {k: v for k, v in metrics.items() if not pd.isna(v)}
        
        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}
