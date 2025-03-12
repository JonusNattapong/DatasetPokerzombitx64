import numpy as np
from typing import List, Dict

class EnsembleModel:
    """Ensemble model combining neural network and XGBoost models"""
    
    def __init__(self, models: List, weights: List[float]):
        """
        Initialize the ensemble model
        
        Args:
            models: List of models to ensemble
            weights: List of weights for each model
        """
        self.models = models
        self.weights = weights
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        for model in self.models:
            model.train(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble model
        
        Args:
            X: Input features
            
        Returns:
            Weighted average of predicted action probabilities
        """
        predictions = np.zeros((X.shape[0], self.models[0].num_classes))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        
        return predictions / sum(self.weights)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = np.mean(y_pred == y_test)
        return {'accuracy': accuracy}