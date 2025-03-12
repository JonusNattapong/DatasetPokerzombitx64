import pandas as pd
from poker_ml.data.loader import PokerDataLoader
from poker_ml.features.feature_generator import PokerFeatureGenerator
from poker_ml.models.nn_model import PokerActionNN
from poker_ml.models.xgb_model import PokerActionXGB
from poker_ml.models.ensemble import EnsembleModel

# Load and preprocess data
data_loader = PokerDataLoader()
df = data_loader.load_hand_data()
train_df, val_df, test_df = data_loader.split_data(df)

# Generate features
feature_gen = PokerFeatureGenerator()
X_train, y_train = feature_gen.transform(train_df)
X_val, y_val = feature_gen.transform(val_df, fit=False)
X_test, y_test = feature_gen.transform(test_df, fit=False)

# Define models
nn_model = PokerActionNN(input_dim=X_train.shape[1])
xgb_model = PokerActionXGB()
ensemble = EnsembleModel([nn_model, xgb_model], weights=[0.6, 0.4])

# Train models
ensemble.train(X_train, y_train)

# Save models if needed (add saving logic)