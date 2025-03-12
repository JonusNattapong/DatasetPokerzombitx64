from poker_ml.data.loader import PokerDataLoader
from poker_ml.features.feature_generator import PokerFeatureGenerator
from poker_ml.models.nn_model import PokerActionNN
from poker_ml.models.xgb_model import PokerActionXGB
from poker_ml.models.ensemble import EnsembleModel

# Load and preprocess data
data_loader = PokerDataLoader()
df = data_loader.load_hand_data()
_, _, test_df = data_loader.split_data(df)

# Generate features
feature_gen = PokerFeatureGenerator()
X_test, y_test = feature_gen.transform(test_df, fit=False)

# Load models (add loading logic if needed)
nn_model = PokerActionNN(input_dim=X_test.shape[1])
xgb_model = PokerActionXGB()
ensemble = EnsembleModel([nn_model, xgb_model], weights=[0.6, 0.4])

# Evaluate models
metrics = ensemble.evaluate(X_test, y_test)
print(f"Ensemble model accuracy: {metrics['accuracy']:.2f}")