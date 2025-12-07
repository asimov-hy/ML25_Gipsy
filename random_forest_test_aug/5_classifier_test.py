import pickle
import numpy as np
import pandas as pd
import sys
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# ============================================================
# 1) Utility Functions (Training과 동일한 전처리)
# ============================================================

def smart_trim(data, threshold=0.5):
    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    active_idx = np.where(vel > threshold)[0]

    if len(active_idx) < 2:
        return data

    start = max(active_idx[0] - 2, 0)
    end = min(active_idx[-1] + 2, len(data))
    return data[start:end]


def rotate_z(data, angle_deg):
    angle = np.radians(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return data @ R


# ============================================================
# 2) Preprocess new single CSV
# ============================================================

def preprocess_new_data(csv_path, target_length=100):
    try:
        df = pd.read_csv(csv_path, header=None, sep='/', skip_blank_lines=True)
        df = df.replace('', np.nan).dropna()

        if df.shape[1] < 3:
            raise ValueError("CSV must contain at least 3 columns (X,Y,Z).")

        data = df.iloc[:, :3].values.astype(float)
        data = data[~np.isnan(data).any(axis=1)]

        if len(data) < 2:
            raise ValueError("Insufficient data length.")

        # --- Smart trimming ---
        data = smart_trim(data)

        # --- Resample ---
        orig_len = data.shape[0]
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_length)
        f_interp = interp1d(x_old, data, axis=0, kind="linear")
        data_resampled = f_interp(x_new)

        # --- Low-pass filtering ---
        FS, CUTOFF = 30.0, 3.0
        nyq = FS * 0.5
        b, a = butter(5, CUTOFF / nyq, btype="low")
        data_denoised = filtfilt(b, a, data_resampled, axis=0)

        return data_denoised

    except Exception as e:
        print(f"❌ Error during preprocessing {csv_path}: {e}")
        return None


# ============================================================
# 3) Feature Extraction (Training과 동일한 9개 feature)
# ============================================================

def extract_features_single(data):
    T = data.shape[0]

    # Max velocity
    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    max_vel = np.max(vel)

    # Variance ratio
    var_y = np.var(data[:, 1])
    var_z = np.var(data[:, 2])
    ratio_yz = var_y / (var_z + 1e-6)

    # Mean X, Mean Y
    mean_x = np.mean(data[:, 0])
    mean_y = np.mean(data[:, 1])

    # Path length
    step = np.linalg.norm(np.diff(data, axis=0), axis=1)
    path_length = np.sum(step)

    # Linearity ratio
    dist_se = np.linalg.norm(data[-1] - data[0])
    linearity_ratio = dist_se / (path_length + 1e-6)

    # Axis ranges
    range_x = np.ptp(data[:, 0])
    range_y = np.ptp(data[:, 1])
    range_z = np.ptp(data[:, 2])

    return np.array([
        mean_x,
        max_vel,
        ratio_yz,
        mean_y,
        path_length,
        linearity_ratio,
        range_x,
        range_y,
        range_z
    ]).reshape(1, -1)


# ============================================================
# 4) Load trained model
# ============================================================

def load_model():
    if not os.path.exists("trained_model.pkl"):
        print("❌ trained_model.pkl not found.")
        print("   Run 4_main.py to train the model first.")
        sys.exit(1)

    with open("trained_model.pkl", "rb") as f:
        return pickle.load(f)


# ============================================================
# 5) Prediction
# ============================================================

def predict_motion(csv_path):
    model_data = load_model()
    model = model_data["model"]
    scaler = model_data["scaler"]
    classes = model_data["classes"]

    print(f"\n--- Processing: {csv_path} ---")

    data = preprocess_new_data(csv_path)
    if data is None:
        return None

    features = extract_features_single(data)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    print("\n==================== RESULT ====================")
    print(f"Predicted Motion:  {pred}")
    print("------------------------------------------------")
    print("Class Probabilities:")
    for cls, p in zip(classes, prob):
        bar = "█" * int(p * 30)
        print(f"{cls:15s} : {p:.4f}  {bar}")
    print("================================================\n")

    return pred, prob


def predict_multiple(csv_paths):
    print("\n========== MULTI-FILE PREDICTION ==========")
    for i, p in enumerate(csv_paths, 1):
        print(f"[{i}/{len(csv_paths)}]")
        predict_motion(p)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    print("\n===== MOTION CLASSIFIER (Random Forest - Augmented Version) =====")

    if len(sys.argv) > 1:
        files = sys.argv[1:]
        if len(files) == 1:
            predict_motion(files[0])
        else:
            predict_multiple(files)
    else:
        print("\nUsage: python 5_classifier_test.py file1.csv [file2.csv ...]")
