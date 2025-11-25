import numpy as np

def extract_features(X_data, y_labels):
    N, T, D = X_data.shape

    # Mean of dimension 0
    feat_mean_x = np.mean(X_data[:, :, 0], axis=1).reshape(-1, 1)

    # Max velocity
    velocity = np.diff(X_data, axis=1)
    feat_max_vel = np.max(np.abs(velocity), axis=(1, 2)).reshape(-1, 1)

    X_features = np.hstack([feat_mean_x, feat_max_vel])
    feature_names = ["Mean_DIM0", "Max_Velocity"]

    return X_features, feature_names
