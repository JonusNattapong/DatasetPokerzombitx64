"""
Visualizer module for creating poker data visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Union, Tuple
import os


class PokerVisualizer:
    """Class for creating poker data visualizations."""
    
    def __init__(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        """
        Initialize the visualizer with poker data.
        
        Args:
            df: DataFrame containing poker hand data
            save_dir: Directory to save plot images (if None, plots will be displayed)
        """
        self.df = df
        self.save_dir = save_dir
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set default plot style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12
        })
    
    def _save_or_show(self, filename: Optional[str] = None):
        """Helper method to either save or show the plot."""
        if self.save_dir and filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Plot saved to {filepath}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_balance_over_time(self, player: Optional[str] = None, 
                               filename: Optional[str] = None):
        """
        Plot cumulative balance over time.
        
        Args:
            player: Player name to filter by (None for all players)
            filename: Filename to save plot (if None and save_dir is set, will use default name)
        """
        # Filter data if player is specified
        if player:
            plot_df = self.df[self.df['name'] == player].copy()
            title = f"Cumulative Balance Over Time - {player}"
        else:
            plot_df = self.df.copy()
            title = "Cumulative Balance Over Time - All Players"
        
        if plot_df.empty:
            print(f"No data available for player: {player}")
            return
        
        # Sort by date and time
        plot_df['datetime'] = pd.to_datetime(plot_df['date'] + ' ' + plot_df['time'])
        plot_df = plot_df.sort_values('datetime')
        
        # Calculate cumulative balance
        plot_df = plot_df.groupby(['name', 'datetime']).agg({'balance': 'sum'}).reset_index()
        plot_df['cumulative_balance'] = plot_df.groupby('name')['balance'].cumsum()
        
        # Create plot
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=plot_df, x='datetime', y='cumulative_balance', hue='name')
        
        plt.title(title)
        plt.xlabel('Date & Time')
        plt.ylabel('Cumulative Balance')
        plt.grid(True)
        plt.legend(title='Player')
        
        # Save or show
        if filename is None and self.save_dir:
            filename = f"balance_over_time{'_' + player if player else ''}.png"
        
        self._save_or_show(filename)
    
    def plot_action_distribution(self, stage: str = 'pre', 
                                player: Optional[str] = None,
                                top_n: int = 10,
                                filename: Optional[str] = None):
        """
        Plot distribution of actions at a given stage.
        
        Args:
            stage: Betting stage ('pre', 'flop', 'turn', 'river')
            player: Player name to filter by (None for all players)
            top_n: Number of top actions to show
            filename: Filename to save plot
        """
        action_col = f'action_{stage}'
        
        if action_col not in self.df.columns:
            print(f"Action column '{action_col}' not found in data")
            return
        
        # Filter data
        if player:
            plot_df = self.df[self.df['name'] == player].copy()
            title = f"{stage.capitalize()} Action Distribution - {player}"
        else:
            plot_df = self.df.copy()
            title = f"{stage.capitalize()} Action Distribution - All Players"
        
        if plot_df.empty:
            print(f"No data available for plotting action distribution")
            return
        
        # Count actions
        action_counts = plot_df[action_col].value_counts().reset_index()
        action_counts.columns = ['action', 'count']
        
        # Get top N actions
        if len(action_counts) > top_n:
            top_actions = action_counts.iloc[:top_n]
            other_count = action_counts.iloc[top_n:]['count'].sum()
            other_row = pd.DataFrame({'action': ['Other'], 'count': [other_count]})
            action_counts = pd.concat([top_actions, other_row])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=action_counts, x='action', y='count')
        
        plt.title(title)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
        
        # Save or show
        if filename is None and self.save_dir:
            filename = f"action_dist_{stage}{'_' + player if player else ''}.png"
        
        self._save_or_show(filename)
    
    def plot_position_winrate(self, player: Optional[str] = None,
                             filename: Optional[str] = None):
        """
        Plot winrate by position.
        
        Args:
            player: Player name to filter by (None for all players)
            filename: Filename to save plot
        """
        # Filter data
        if player:
            plot_df = self.df[self.df['name'] == player].copy()
            title = f"Winrate by Position - {player}"
        else:
            plot_df = self.df.copy()
            title = "Winrate by Position - All Players"
        
        if plot_df.empty:
            print(f"No data available for plotting position winrate")
            return
        
        # Calculate winrate by position
        position_stats = (plot_df.groupby('position')
                         .agg({'balance': ['sum', 'count', 'mean']})
                         .reset_index())
        
        position_stats.columns = ['position', 'total_balance', 'hands_played', 'avg_balance']
        
        # Order positions correctly
        position_order = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN', 'x']
        position_stats['position'] = pd.Categorical(
            position_stats['position'], 
            categories=position_order,
            ordered=True
        )
        position_stats = position_stats.sort_values('position')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=position_stats, x='position', y='avg_balance')
        
        plt.title(title)
        plt.xlabel('Position')
        plt.ylabel('Average Balance')
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
        
        # Save or show
        if filename is None and self.save_dir:
            filename = f"position_winrate{'_' + player if player else ''}.png"
        
        self._save_or_show(filename)
    
    def plot_hand_strength_distribution(self, player: Optional[str] = None,
                                       filename: Optional[str] = None):
        """
        Plot distribution of hand strengths at showdown.
        
        Args:
            player: Player name to filter by (None for all players)
            filename: Filename to save plot
        """
        # Filter data for hands that went to showdown
        showdown_df = self.df[self.df['combination'] != ""].copy()
        
        if player:
            showdown_df = showdown_df[showdown_df['name'] == player]
            title = f"Hand Strength Distribution at Showdown - {player}"
        else:
            title = "Hand Strength Distribution at Showdown - All Players"
        
        if showdown_df.empty:
            print(f"No showdown data available for plotting")
            return
        
        # Extract hand type from combination
        def extract_hand_type(combination):
            if pd.isna(combination) or combination == "":
                return "Unknown"
            
            hand_types = [
                "Royal Flush", "Straight Flush", "Four of a Kind", 
                "Full House", "Flush", "Straight", "Three of a Kind",
                "Two Pair", "Pair", "High Card"
            ]
            
            for hand_type in hand_types:
                if hand_type.lower() in combination.lower():
                    return hand_type
            
            return "Other"
        
        showdown_df['hand_type'] = showdown_df['combination'].apply(extract_hand_type)
        
        # Count hand types
        hand_counts = showdown_df['hand_type'].value_counts().reset_index()
        hand_counts.columns = ['hand_type', 'count']
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=hand_counts, x='hand_type', y='count')
        
        plt.title(title)
        plt.xlabel('Hand Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
        
        # Save or show
        if filename is None and self.save_dir:
            filename = f"hand_strength{'_' + player if player else ''}.png"
        
        self._save_or_show(filename)
    
    def plot_player_stats(self, top_n: int = 10, metric: str = 'hands', 
                         filename: Optional[str] = None):
        """
        Plot player statistics.
        
        Args:
            top_n: Number of top players to show
            metric: Metric to rank players by ('hands', 'winrate', 'total')
            filename: Filename to save plot
        """
        # Group by player name and calculate stats
        player_stats = (self.df.groupby('name')
                       .agg({
                           'hand_id': 'count', 
                           'balance': ['sum', 'mean']
                       })
                       .reset_index())
        
        player_stats.columns = ['name', 'hands_played', 'total_balance', 'avg_balance']
        
        # Filter players with significant number of hands
        min_hands = 10
        player_stats = player_stats[player_stats['hands_played'] >= min_hands]
        
        if player_stats.empty:
            print(f"No players with at least {min_hands} hands played")
            return
        
        # Rank players based on selected metric
        if metric == 'hands':
            player_stats = player_stats.sort_values('hands_played', ascending=False)
            y_col = 'hands_played'
            title = f"Top {top_n} Players by Hands Played"
            ylabel = "Hands Played"
        elif metric == 'winrate':
            player_stats = player_stats.sort_values('avg_balance', ascending=False)
            y_col = 'avg_balance'
            title = f"Top {top_n} Players by Win Rate"
            ylabel = "Average Balance per Hand"
        else:  # total
            player_stats = player_stats.sort_values('total_balance', ascending=False)
            y_col = 'total_balance'
            title = f"Top {top_n} Players by Total Balance"
            ylabel = "Total Balance"
        
        # Take top N players
        player_stats = player_stats.head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=player_stats, x='name', y=y_col)
        
        plt.title(title)
        plt.xlabel('Player')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
        
        # Save or show
        if filename is None and self.save_dir:
            filename = f"player_stats_{metric}.png"
        
        self._save_or_show(filename)


def generate_all_visualizations(data_path: str, output_dir: str, player: Optional[str] = None):
    """
    Generate all visualizations from a poker dataset.
    
    Args:
        data_path: Path to the CSV file containing poker data
        output_dir: Directory to save visualizations
        player: Optional player name to filter visualizations
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Create visualizer
    visualizer = PokerVisualizer(df, save_dir=output_dir)
    
    # Generate all plots
    print("Generating visualizations...")
    
    # Balance over time
    visualizer.plot_balance_over_time(player=player)
    
    # Action distributions
    for stage in ['pre', 'flop', 'turn', 'river']:
        visualizer.plot_action_distribution(stage=stage, player=player)
    
    # Position winrate
    visualizer.plot_position_winrate(player=player)
    
    # Hand strength distribution
    visualizer.plot_hand_strength_distribution(player=player)
    
    # Player stats
    if not player:
        for metric in ['hands', 'winrate', 'total']:
            visualizer.plot_player_stats(metric=metric)
    
    print(f"All visualizations saved to {output_dir}")
