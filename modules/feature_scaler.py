# This is created for test pipeline 2
import numpy as np

def prepare_features(raw_dict, feature_order, scaler):
    """
    Aligns raw features to training order & applies scaling.
    """
    X = np.array([[raw_dict.get(f, 0) for f in feature_order]])
    X_scaled = scaler.transform(X)
    return X_scaled
