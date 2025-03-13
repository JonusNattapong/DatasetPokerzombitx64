"""
Evaluator module for poker machine learning models.
This module provides functionality to evaluate poker strategies and models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .metrics import calculate_roi, calculate_win_rate, calculate_vpip, calculate_pfr, calculate_af
from ..models.base_model import BasePokerModel

# Configure logging
logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """
    Class for evaluating poker strategies and machine learning models.
    """
    
    def __init__(self, test_data: pd.DataFrame, metrics: List[str] = None, output_dir: str = "./evaluation_results"):
        """
        Initialize the evaluator with test data.
        
        Args:
            test_data: DataFrame containing poker hand data for testing
            metrics: List of metrics to use for evaluation (default: all available metrics)
            output_dir: Directory to save evaluation results
        """
        self.test_data = test_data
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Define available metrics
        self.available_metrics = {
            'accuracy': self._calculate_accuracy,
            'precision': self._calculate_precision,
            'recall': self._calculate_recall,
            'f1': self._calculate_f1,
            'roi': calculate_roi,
            'win_rate': calculate_win_rate,
            'vpip': calculate_vpip,
            'pfr': calculate_pfr,
            'af': calculate_af
        }
        
        # Select metrics to use
        self.metrics = metrics if metrics else list(self.available_metrics.keys())
        
        # Validate metrics
        invalid_metrics = [m for m in self.metrics if m not in self.available_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available metrics: {list(self.available_metrics.keys())}")
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score."""
        return precision_score(y_true, y_pred, average='weighted')
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score."""
        return recall_score(y_true, y_pred, average='weighted')
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        return f1_score(y_true, y_pred, average='weighted')
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a machine learning model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate classification metrics
        results = {}
        for metric in self.metrics:
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                results[metric] = self.available_metrics[metric](y_test, y_pred)
        
        return results
    
    def evaluate_poker_strategy(self, strategy: Callable, data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate a poker strategy function.
        
        Args:
            strategy: Function that takes hand data and returns action decisions
            data: DataFrame to evaluate on (defaults to self.test_data)
            
        Returns:
            Dictionary of poker-specific metrics
        """
        if data is None:
            data = self.test_data
        
        # Apply strategy to data
        result_data = strategy(data)
        
        # Calculate poker-specific metrics
        results = {}
        for metric in self.metrics:
            if metric in ['roi', 'win_rate', 'vpip', 'pfr', 'af']:
                results[metric] = self.available_metrics[metric](result_data)
        
        return results
    
    def compare_models(self, models: List[Tuple[str, Any]], X_test: np.ndarray, 
                      y_test: np.ndarray, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple machine learning models.
        
        Args:
            models: List of (model_name, model) tuples
            X_test: Test features
            y_test: True labels
            save_path: Path to save comparison results (optional)
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models:
            model_results = self.evaluate_model(model, X_test, y_test)
            model_results['model'] = model_name
            results.append(model_results)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns to have model name first
        cols = ['model'] + [col for col in comparison_df.columns if col != 'model']
        comparison_df = comparison_df[cols]
        
        # Save if path provided
        if save_path:
            comparison_df.to_csv(os.path.join(self.output_dir, save_path), index=False)
            logger.info(f"Model comparison saved to {save_path}")
        
        return comparison_df
    
    def visualize_results(self, results: pd.DataFrame, metric: str, 
                         title: str = "Model Comparison", save_path: Optional[str] = None):
        """
        Visualize evaluation results.
        
        Args:
            results: DataFrame containing evaluation results
            metric: Metric to visualize
            title: Plot title
            save_path: Path to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        if metric in results.columns:
            # Create bar plot
            ax = sns.barplot(x='model', y=metric, data=results)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom')
            
            plt.title(title)
            plt.ylabel(metric.capitalize())
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(os.path.join(self.output_dir, save_path))
                plt.close()
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
        else:
            logger.warning(f"Metric '{metric}' not found in results")
    
    def plot_confusion_matrix(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                             class_names: List[str] = None, save_path: Optional[str] = None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: True labels
            class_names: Names for the classes
            save_path: Path to save visualization
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            plt.close()
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def evaluate_on_game_simulation(self, models: List[BasePokerModel], 
                                  num_hands: int = 1000, 
                                  initial_stack: float = 100.0,
                                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate models by simulating a poker game.
        
        Args:
            models: List of poker models to evaluate
            num_hands: Number of hands to simulate
            initial_stack: Starting stack size for each model
            save_path: Path to save results
            
        Returns:
            DataFrame with simulation results
        """
        # This would require a poker game simulator component
        # Here we're just providing a placeholder implementation
        logger.warning("Poker game simulation is not fully implemented")
        
        results = []
        for i, model in enumerate(models):
            # Placeholder for simulation results
            model_results = {
                'model_name': model.name if hasattr(model, 'name') else f"Model {i+1}",
                'final_stack': initial_stack * (1 + np.random.normal(0.05, 0.2)),  # Random result
                'hands_played': num_hands,
                'win_rate_bb_per_100': np.random.normal(2.0, 5.0),  # Random result
                'biggest_pot_won': np.random.uniform(5.0, 50.0)  # Random result
            }
            results.append(model_results)
        
        # Create DataFrame
        simulation_df = pd.DataFrame(results)
        
        # Save if path provided
        if save_path:
            simulation_df.to_csv(os.path.join(self.output_dir, save_path), index=False)
            logger.info(f"Simulation results saved to {save_path}")
        
        return simulation_df


def evaluate_decision_accuracy(predicted_actions: List[str], optimal_actions: List[str]) -> float:
    """
    Evaluate the accuracy of poker decisions compared to optimal play.
    
    Args:
        predicted_actions: List of predicted actions
        optimal_actions: List of optimal actions
        
    Returns:
        Accuracy score
    """
    if len(predicted_actions) != len(optimal_actions):
        raise ValueError("Predicted and optimal action lists must have the same length")
    
    correct = sum(1 for p, o in zip(predicted_actions, optimal_actions) if p == o)
    return correct / len(predicted_actions)


def evaluate_equity_preservation(predicted_actions: List[str], 
                               hand_equities: List[float],
                               optimal_actions: List[str]) -> float:
    """
    Evaluate how well decisions preserve equity compared to optimal play.
    
    Args:
        predicted_actions: List of predicted actions
        hand_equities: List of hand equities
        optimal_actions: List of optimal actions
        
    Returns:
        Equity preservation score (1.0 means perfect preservation)
    """
    if not all(len(lst) == len(predicted_actions) for lst in [hand_equities, optimal_actions]):
        raise ValueError("All input lists must have the same length")
    
    equity_preserved = 0.0
    total_equity = sum(hand_equities)
    
    for action, equity, optimal in zip(predicted_actions, hand_equities, optimal_actions):
        if action == optimal:
            equity_preserved += equity
    
    return equity_preserved / total_equity if total_equity > 0 else 0.0


def generate_evaluation_report(model_name: str, metrics: Dict[str, float], 
                             output_dir: str = "./evaluation_results") -> str:
    """
    Generate a text report of evaluation results.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create report filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_evaluation_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Generate report content
    with open(filepath, 'w') as f:
        f.write(f"Evaluation Report for {model_name}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
    
    logger.info(f"Evaluation report saved to {filepath}")
    return filepath
