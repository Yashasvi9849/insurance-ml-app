import numpy as np

def prepare_features(raw_dict, feature_order=None, scaler=None):
    """
    Prepares input features for the model.
    Works even if feature_order or scaler are None.
    """
    # Handle missing feature_order
    if feature_order:
        X = np.array([[raw_dict.get(f, 0) for f in feature_order]])
    else:
        # Use raw_dict values directly (consistent order)
        X = np.array([[v for v in raw_dict.values()]])

    # Handle missing scaler
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    return X_scaled
