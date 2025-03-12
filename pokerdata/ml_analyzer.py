"""
Machine Learning analyzer module for poker data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import TensorFlow, but handle the case where it's not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, Dropout, Embedding, InputLayer, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available and imported successfully.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow is not available. Machine learning functions will be limited.")

# Try to import scikit-learn for clustering and other ML algorithms
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn is available and imported successfully.")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not available. Some machine learning functions will be limited.")


class PokerMLAnalyzer:
    """Class for performing machine learning analysis on poker data."""
    
    def __init__(self, df: pd.DataFrame = None, model_dir: str = './models'):
        """
        Initialize the analyzer with poker data.
        
        Args:
            df: DataFrame containing poker hand data
            model_dir: Directory to save/load models
        """
        self.df = df
        self.model_dir = model_dir
        self.model = None
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow is not available. Neural network models cannot be used.")
            
    def prepare_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning models.
        
        Args:
            df: DataFrame to prepare features from (uses self.df if None)
            
        Returns:
            Tuple of (prepared DataFrame, list of feature column names)
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No DataFrame provided")
        
        # Create a copy to avoid modifying the original
        prep_df = df.copy()
        
        # Encode categorical features
        categorical_cols = ['position', 'action_pre', 'action_flop', 'action_turn', 'action_river']
        for col in categorical_cols:
            if col in prep_df.columns:
                # Create dummy variables and drop the original column
                dummies = pd.get_dummies(prep_df[col], prefix=col, drop_first=True)
                prep_df = pd.concat([prep_df, dummies], axis=1)
                prep_df.drop(col, axis=1, inplace=True)
        
        # Select numeric features
        numeric_cols = ['stack', 'pot_pre', 'pot_flop', 'pot_turn', 'pot_river',
                       'bet_pre', 'bet_flop', 'bet_turn', 'bet_river',
                       'ante', 'blinds']
        
        # Get all the feature columns (numeric + encoded categorical)
        feature_cols = [col for col in prep_df.columns if col not in 
                        ['name', 'hand_id', 'date', 'time', 'cards', 'board_flop', 
                         'board_turn', 'board_river', 'combination', 'result', 'balance']]
        
        return prep_df, feature_cols
    
    def build_action_prediction_model(self, input_dim: int) -> Sequential:
        """
        Build a neural network model to predict player actions.
        
        Args:
            input_dim: Dimension of the input features
            
        Returns:
            Compiled Keras Sequential model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot build neural network models.")
        
        model = Sequential([
            InputLayer(input_shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')  # 4 classes: fold, check/call, bet/raise, all-in
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_action_prediction_model(self, player: str = None, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a model to predict player actions.
        
        Args:
            player: Player name to build model for (None for all players)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with model, accuracy, and other training results
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot train neural network models.")
        
        # Filter data for the player if specified
        if player:
            train_df = self.df[self.df['name'] == player].copy()
        else:
            train_df = self.df.copy()
        
        # Prepare features
        prep_df, feature_cols = self.prepare_features(train_df)
        
        # Extract features and target
        X = prep_df[feature_cols].values
        
        # Create target from action columns
        def categorize_action(row):
            for col in ['action_pre', 'action_flop', 'action_turn', 'action_river']:
                if col in row and isinstance(row[col], str):
                    if 'fold' in row[col]:
                        return 0  # fold
                    elif 'check' in row[col] or 'call' in row[col]:
                        return 1  # check/call
                    elif 'bet' in row[col] or 'raise' in row[col]:
                        return 2  # bet/raise
                    elif 'all-in' in row[col]:
                        return 3  # all-in
            return 1  # default to check/call
        
        train_df['action_category'] = train_df.apply(categorize_action, axis=1)
        y = to_categorical(train_df['action_category'].values, num_classes=4)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create and train the model
        model = self.build_action_prediction_model(len(feature_cols))
        
        # Set up callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_path = os.path.join(self.model_dir, f"action_model_{player if player else 'all'}.h5")
        checkpoint = ModelCheckpoint(model_path, save_best_only=True)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Save the trained model
        self.model = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'history': history.history,
            'feature_columns': feature_cols
        }
    
    def cluster_players(self, n_clusters: int = 4) -> Dict[str, Any]:
        """
        Cluster players based on their playing style.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with cluster assignments and analysis
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Cannot perform clustering.")
        
        # Aggregate data by player
        player_stats = []
        
        for player_name, player_df in self.df.groupby('name'):
            # Calculate VPIP (Voluntarily Put money In Pot)
            vpip_actions = player_df['action_pre'].apply(
                lambda x: 1 if isinstance(x, str) and ('call' in x or 'bet' in x or 'raise' in x) else 0
            ).mean()
            
            # Calculate PFR (Pre-Flop Raise)
            pfr = player_df['action_pre'].apply(
                lambda x: 1 if isinstance(x, str) and 'raise' in x else 0
            ).mean()
            
            # Calculate AF (Aggression Factor)
            bets_raises = sum(
                player_df[col].apply(lambda x: 1 if isinstance(x, str) and ('bet' in x or 'raise' in x) else 0).sum()
                for col in ['action_pre', 'action_flop', 'action_turn', 'action_river']
            )
            calls = sum(
                player_df[col].apply(lambda x: 1 if isinstance(x, str) and 'call' in x else 0).sum()
                for col in ['action_pre', 'action_flop', 'action_turn', 'action_river']
            )
            af = bets_raises / calls if calls > 0 else bets_raises
            
            # Calculate WTSD (Went To ShowDown)
            hands_played = len(player_df)
            showdowns = player_df['combination'].apply(lambda x: 1 if x and len(x) > 0 else 0).sum()
            wtsd = showdowns / hands_played if hands_played > 0 else 0
            
            player_stats.append({
                'name': player_name,
                'hands_played': hands_played,
                'vpip': vpip_actions,
                'pfr': pfr,
                'af': af,
                'wtsd': wtsd
            })
        
        # Create DataFrame from player stats
        players_df = pd.DataFrame(player_stats)
        
        # Only keep players with enough hands
        players_df = players_df[players_df['hands_played'] >= 10]
        
        if len(players_df) < n_clusters:
            logger.warning(f"Not enough players ({len(players_df)}) for {n_clusters} clusters. Reducing clusters.")
            n_clusters = max(2, len(players_df) // 2)
        
        # Extract features for clustering
        features = ['vpip', 'pfr', 'af', 'wtsd']
        X = players_df[features].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to the DataFrame
        players_df['cluster'] = clusters
        
        # Calculate cluster centers in original scale
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create cluster description based on styles
        cluster_styles = []
        for i, center in enumerate(centers):
            vpip, pfr, af, wtsd = center
            
            # Determine play style based on cluster center
            if vpip < 0.22:  # Tight
                if af > 2.0:  # Aggressive
                    style = "Tight-Aggressive (TAG)"
                else:  # Passive
                    style = "Tight-Passive (Rock)"
            else:  # Loose
                if af > 2.0:  # Aggressive
                    style = "Loose-Aggressive (LAG)"
                else:  # Passive
                    style = "Loose-Passive (Calling Station)"
            
            cluster_styles.append({
                'cluster': i,
                'style': style,
                'center': dict(zip(features, center)),
                'players': players_df[players_df['cluster'] == i]['name'].tolist(),
                'count': sum(clusters == i)
            })
        
        return {
            'clusters': players_df,
            'cluster_styles': cluster_styles,
            'features': features,
            'n_clusters': n_clusters
        }
    
    def visualize_player_clusters(self, cluster_results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualize player clusters.
        
        Args:
            cluster_results: Results from cluster_players method
            save_path: Path to save the visualization (if None, display only)
        """
        players_df = cluster_results['clusters']
        cluster_styles = cluster_results['cluster_styles']
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot of VPIP vs PFR
        scatter = plt.scatter(
            players_df['vpip'], 
            players_df['pfr'],
            c=players_df['cluster'], 
            cmap='viridis',
            s=100 * players_df['hands_played'] / players_df['hands_played'].max(),
            alpha=0.7
        )
        
        # Add cluster centers
        for style_info in cluster_styles:
            center = style_info['center']
            plt.scatter(
                center['vpip'], 
                center['pfr'], 
                marker='X', 
                s=200, 
                c=f"C{style_info['cluster']}", 
                edgecolors='black', 
                linewidths=2
            )
            plt.annotate(
                style_info['style'],
                (center['vpip'], center['pfr']),
                xytext=(10, 5),
                textcoords='offset points',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Add player names
        for i, row in players_df.iterrows():
            plt.annotate(
                row['name'],
                (row['vpip'], row['pfr']),
                fontsize=8,
                alpha=0.7
            )
        
        # Add a diagonal line representing VPIP = PFR
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Set labels and title
        plt.xlabel('VPIP (Voluntarily Put Money In Pot)')
        plt.ylabel('PFR (Pre-Flop Raise)')
        plt.title('Player Clustering by Playing Style')
        
        # Add a colorbar legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster')
        
        # Add a legend for size
        sizes = [10, 30, 50, 100]
        labels = [f"{size}% of max hands" for size in sizes]
        plt.legend(
            handles=[plt.scatter([], [], s=s*2, color='gray', alpha=0.7) for s in sizes],
            labels=labels,
            title="Hands Played",
            loc='upper left'
        )
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def cluster_players(df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, Any]:
    """
    Cluster poker players based on their playing style.
    
    Args:
        df: DataFrame containing poker hand data
        n_clusters: Number of clusters to create
        
    Returns:
        Dictionary with cluster assignments and analysis
    """
    analyzer = PokerMLAnalyzer(df)
    return analyzer.cluster_players(n_clusters)


def visualize_player_clusters(cluster_results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Visualize player clusters.
    
    Args:
        cluster_results: Results from cluster_players function
        save_path: Path to save the visualization (if None, display only)
    """
    analyzer = PokerMLAnalyzer()
    analyzer.visualize_player_clusters(cluster_results, save_path)


def train_decision_model(df: pd.DataFrame, target: str = 'action_pre', player: str = None) -> Dict[str, Any]:
    """
    Train a model to predict player decisions.
    
    Args:
        df: DataFrame containing poker hand data
        target: Target action to predict ('action_pre', 'action_flop', etc.)
        player: Player name to build model for (None for all players)
        
    Returns:
        Dictionary with model and training results
    """
    analyzer = PokerMLAnalyzer(df)
    return analyzer.train_action_prediction_model(player)


def analyze_range_strength(df: pd.DataFrame, player: str, position: Optional[str] = None, 
                          action: Optional[str] = None, num_simulations: int = 1000) -> Dict[str, float]:
    """
    Analyze the strength of a player's hand range in various situations.
    
    Args:
        df: DataFrame containing poker hand data
        player: Player name to analyze
        position: Optional position filter (e.g., 'BTN', 'SB')
        action: Optional action filter (e.g., 'raises', 'calls')
        num_simulations: Number of Monte Carlo simulations to run
        
    Returns:
        Dictionary with range analysis results
    """
    # Filter data
    player_df = df[df['name'] == player].copy()
    
    if position:
        player_df = player_df[player_df['position'] == position]
        
    if action:
        player_df = player_df[player_df['action_pre'].str.contains(action, na=False)]
    
    # Get hands where we know the cards
    known_hands = player_df[player_df['cards'] != '--'].copy()
    
    if len(known_hands) < 5:
        return {
            'error': f"Not enough known hands for player {player}",
            'hands_found': len(known_hands)
        }
    
    # Calculate win rates against random ranges
    # (This would normally use a poker equity calculator or hand vs range simulations)
    # Here we're just providing a simplified placeholder
    
    return {
        'player': player,
        'position': position if position else 'all',
        'action': action if action else 'all',
        'hands_analyzed': len(known_hands),
        'avg_equity': 0.52,  # Placeholder - would be calculated from simulations
        'top_hands': known_hands['cards'].value_counts().head(5).to_dict()
    }


def identify_mistakes(df: pd.DataFrame, player: str, ev_threshold: float = -0.5) -> Dict[str, Any]:
    """
    Identify potentially suboptimal plays in a player's hand history.
    
    Args:
        df: DataFrame containing poker hand data
        player: Player name to analyze
        ev_threshold: Threshold for flagging decisions as mistakes (in BB units)
        
    Returns:
        Dictionary with identified mistakes
    """
    # Placeholder implementation - would require GTO solver integration
    # or a trained model that can evaluate EV of decisions
    
    return {
        'player': player,
        'ev_threshold': ev_threshold,
        'num_hands': len(df[df['name'] == player]),
        'potential_mistakes': [],  # Would contain hand IDs and descriptions of mistakes
        'note': "This is a placeholder implementation. Actual mistake identification requires GTO solver integration."
    }


def analyze_style_evolution(df: pd.DataFrame, player: str, window_size: int = 500) -> Dict[str, Any]:
    """
    Analyze how a player's style evolves over time.
    
    Args:
        df: DataFrame containing poker hand data
        player: Player name to analyze
        window_size: Number of hands for each time window
        
    Returns:
        Dictionary with style evolution analysis
    """
    # Filter data for the player
    player_df = df[df['name'] == player].copy()
    
    if len(player_df) < window_size:
        return {
            'error': f"Not enough hands for player {player}. Need at least {window_size}, found {len(player_df)}."
        }
    
    # Sort by date and time
    player_df['datetime'] = pd.to_datetime(player_df['date'] + ' ' + player_df['time'], errors='coerce')
    player_df = player_df.sort_values('datetime')
    
    # Create windows
    windows = []
    for i in range(0, len(player_df), window_size // 2):  # 50% overlap between windows
        window_df = player_df.iloc[i:min(i + window_size, len(player_df))]
        if len(window_df) < window_size // 2:  # Skip if window is too small
            continue
            
        # Calculate metrics for this window
        vpip = window_df['action_pre'].apply(
            lambda x: 1 if isinstance(x, str) and ('call' in x or 'bet' in x or 'raise' in x) else 0
        ).mean()
        
        pfr = window_df['action_pre'].apply(
            lambda x: 1 if isinstance(x, str) and 'raise' in x else 0
        ).mean()
        
        # Get start and end dates
        start_date = window_df['datetime'].min()
        end_date = window_df['datetime'].max()
        
        windows.append({
            'window_idx': len(windows),
            'start_date': start_date,
            'end_date': end_date,
            'num_hands': len(window_df),
            'vpip': vpip,
            'pfr': pfr,
            'winrate': window_df['balance'].mean()
        })
    
    return {
        'player': player,
        'windows': windows,
        'window_size': window_size,
        'total_hands': len(player_df)
    }


def plot_style_evolution(evolution_data: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot a player's style evolution over time.
    
    Args:
        evolution_data: Results from analyze_style_evolution function
        save_path: Path to save the visualization (if None, display only)
    """
    if 'error' in evolution_data:
        print(f"Error: {evolution_data['error']}")
        return
        
    windows = evolution_data['windows']
    player = evolution_data['player']
    
    # Convert to DataFrame for easier plotting
    evolution_df = pd.DataFrame(windows)
    
    # Set up the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot VPIP
    axes[0].plot(evolution_df['window_idx'], evolution_df['vpip'], 'b-', marker='o', linewidth=2)
    axes[0].set_ylabel('VPIP')
    axes[0].set_title(f"Style Evolution for {player}")
    axes[0].grid(True, alpha=0.3)
    
    # Plot PFR
    axes[1].plot(evolution_df['window_idx'], evolution_df['pfr'], 'r-', marker='o', linewidth=2)
    axes[1].set_ylabel('PFR')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Win Rate
    axes[2].plot(evolution_df['window_idx'], evolution_df['winrate'], 'g-', marker='o', linewidth=2)
    axes[2].set_ylabel('Win Rate (BB/hand)')
    axes[2].set_xlabel('Window Index')
    axes[2].grid(True, alpha=0.3)
    
    # Add date labels on x-axis
    def format_date(idx):
        if idx < len(evolution_df):
            return evolution_df.iloc[idx]['start_date'].strftime('%Y-%m-%d')
        return ''
    
    # Replace x-ticks with dates
    x_ticks = np.arange(0, len(evolution_df), max(1, len(evolution_df) // 5))
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels([format_date(i) for i in x_ticks], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
