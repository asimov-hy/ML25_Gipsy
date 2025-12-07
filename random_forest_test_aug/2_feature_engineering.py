import numpy as np
from tslearn.metrics import dtw


def extract_features(X_data, y_labels):
    N, T, D = X_data.shape

    features = []

    # Mean X
    mean_x = np.mean(X_data[:, :, 0], axis=1)

    # Max velocity
    vel = np.linalg.norm(np.diff(X_data, axis=1), axis=2)
    max_vel = np.max(vel, axis=1)

    # Variance ratio (horizontal vs vertical)
    var_y = np.var(X_data[:, :, 1], axis=1)
    var_z = np.var(X_data[:, :, 2], axis=1)
    ratio_yz = var_y / (var_z + 1e-6)

    # Workspace mean Y
    mean_y = np.mean(X_data[:, :, 1], axis=1)

    # Path length
    step = np.linalg.norm(np.diff(X_data, axis=1), axis=2)
    path_length = np.sum(step, axis=1)

    # Linearity ratio
    start = X_data[:, 0, :]
    end = X_data[:, -1, :]
    dist_start_end = np.linalg.norm(end - start, axis=1)
    linearity_ratio = dist_start_end / (path_length + 1e-6)

    # Range (XYZ)
    range_x = np.ptp(X_data[:, :, 0], axis=1)
    range_y = np.ptp(X_data[:, :, 1], axis=1)
    range_z = np.ptp(X_data[:, :, 2], axis=1)

    # Combine features
    X_feat = np.vstack([
        mean_x,
        max_vel,
        ratio_yz,
        mean_y,
        path_length,
        linearity_ratio,
        range_x,
        range_y,
        range_z,
    ]).T

    feature_names = [
        "mean_x",
        "max_velocity",
        "var_ratio_yz",
        "mean_y",
        "path_length",
        "linearity_ratio",
        "range_x",
        "range_y",
        "range_z",
    ]

    return X_feat, feature_names
