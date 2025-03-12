import xgboost as xgb
import numpy as np
from typing import Dict

class PokerActionXGB:
    """XGBoost model for poker action prediction"""
    
    def __init__(self, num_classes: int = 4):
        """
        Initialize the XGBoost model
        
        Args:
            num_classes: Number of action classes
        """
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> xgb.XGBClassifier:
        """Build the XGBoost model"""
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.01,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=self.num_classes
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the XGBoost model
        
        Args:
            X: Input features
            
        Returns:
            Predicted action probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        evals_result = self.model.eval_set([(X_test, y_test)], verbose=False)
        accuracy = evals_result['validation_0']['merror']
        return {'accuracy': 1 - accuracy}