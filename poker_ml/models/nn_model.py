import tensorflow as tf
import numpy as np
from typing import Dict
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, InputLayer
from tensorflow.python.keras.callbacks import EarlyStopping
class PokerActionNN:
    """Neural network model for poker action prediction"""
    
    def __init__(self, input_dim: int, num_classes: int = 4):
        """
        Initialize the neural network model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of action classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build the neural network model"""
        model = Sequential()
        model.add(InputLayer(input_shape=(self.input_dim,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.15, epochs: int = 100):
        """
        Train the neural network model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, callbacks=[early_stopping])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the neural network model
        
        Args:
            X: Input features
            
        Returns:
            Predicted action probabilities
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return {'loss': loss, 'accuracy': accuracy}