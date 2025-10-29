import numpy as np

def prepare_features(raw_dict, feature_order, scaler):
    """
    Align features to the training order and apply the same scaler.
    """
    X = np.array([[raw_dict.get(f, 0) for f in feature_order]])
    X_scaled = scaler.transform(X)
    return X_scaled
