# modules/predict_logic.py
# This version assumes your saved model file contains ONLY the model object (not a bundle)

import joblib
from modules.feature_prep import prepare_features

model = joblib.load("models/xgboost_model.pkl")
scaler = None
feature_order = None

def predict_benefit(features_dict):
    X_scaled = prepare_features(features_dict, feature_order, scaler)
    prediction = model.predict(X_scaled)[0]
    return round(float(prediction), 2)

