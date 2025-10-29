# This is created for test pipeline 3
import joblib
from modules.feature_prep import prepare_features

bundle = joblib.load("models/simple_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_order = bundle["features"]

def predict_benefit(features_dict):
    X_scaled = prepare_features(features_dict, feature_order, scaler)
    prediction = model.predict(X_scaled)[0]
    return round(float(prediction), 2)
