import unittest
import numpy as np
from poker_ml.models.ensemble import EnsembleModel

class MockModel:
    """Mock model for testing ensemble"""
    def __init__(self, predictions):
        self.predictions = predictions
        self.num_classes = predictions.shape[1]
        self.was_trained = False
        
    def train(self, X_train, y_train):
        self.was_trained = True
        
    def predict(self, X):
        if hasattr(self, 'multi_predictions'):
            return self.multi_predictions
        return np.tile(self.predictions, (len(X), 1))

class TestEnsembleModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case"""
        # Create mock models with known predictions
        self.mock_model1 = MockModel(np.array([[0.7, 0.2, 0.1]]))  # Predicts class 0
        self.mock_model2 = MockModel(np.array([[0.1, 0.8, 0.1]]))  # Predicts class 1
        
        # Create ensemble with equal weights
        self.models = [self.mock_model1, self.mock_model2]
        self.weights = [0.5, 0.5]
        self.ensemble = EnsembleModel(self.models, self.weights)
        
        # Create sample data
        self.X = np.random.rand(10, 5)  # 10 samples, 5 features
        self.y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    def test_initialization(self):
        """Test ensemble model initialization"""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertEqual(len(self.ensemble.weights), 2)
        self.assertAlmostEqual(sum(self.ensemble.weights), 1.0)

    def test_training(self):
        """Test training propagates to all models"""
        self.ensemble.train(self.X, self.y)
        
        # Check all models were trained
        for model in self.models:
            self.assertTrue(model.was_trained)

    def test_prediction(self):
        """Test ensemble prediction averaging"""
        predictions = self.ensemble.predict(self.X)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (len(self.X), 3))
        
        # Check predictions are weighted average
        expected_predictions = np.array([0.4, 0.5, 0.1])  # (0.7+0.1)/2, (0.2+0.8)/2, (0.1+0.1)/2
        np.testing.assert_array_almost_equal(predictions[0], expected_predictions)
        
        # Check predictions sum to 1
        np.testing.assert_array_almost_equal(np.sum(predictions, axis=1), np.ones(len(self.X)))

    def test_evaluation(self):
        """Test ensemble evaluation metrics"""
        # Create a case where ensemble predicts correctly
        y_test = np.array([0, 1])  # True labels
        X_test = np.random.rand(2, 5)  # 2 samples
        
        # Set up predictions for each sample
        model1 = MockModel(np.array([[0.9, 0.05, 0.05]]))
        model1.multi_predictions = np.array([
            [0.9, 0.05, 0.05],  # Strongly predicts class 0 for first sample
            [0.05, 0.9, 0.05]   # Strongly predicts class 1 for second sample
        ])
        
        model2 = MockModel(np.array([[0.8, 0.1, 0.1]]))
        model2.multi_predictions = np.array([
            [0.8, 0.1, 0.1],    # Also predicts class 0 for first sample
            [0.1, 0.8, 0.1]     # Also predicts class 1 for second sample
        ])
        ensemble = EnsembleModel([model1, model2], [0.5, 0.5])
        
        metrics = ensemble.evaluate(X_test, y_test)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['accuracy'], 1.0)  # Both predictions should be correct

    def test_unequal_weights(self):
        """Test ensemble with unequal weights"""
        # Create ensemble with unequal weights
        weights = [0.7, 0.3]
        ensemble = EnsembleModel(self.models, weights)
        
        predictions = ensemble.predict(self.X)
        
        # Check predictions are weighted correctly
        expected_predictions = np.array([
            0.7 * 0.7 + 0.3 * 0.1,  # First class
            0.7 * 0.2 + 0.3 * 0.8,  # Second class
            0.7 * 0.1 + 0.3 * 0.1   # Third class
        ])
        np.testing.assert_array_almost_equal(predictions[0], expected_predictions)

if __name__ == '__main__':
    unittest.main()