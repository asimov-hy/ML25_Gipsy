import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# ======================================================
# 0) SAFE LOADER (slash-separated, comma-separated 대응)
# ======================================================
import re
import numpy as np

def robust_load_csv(path):
    """
    어떤 형태의 CSV든 숫자 3개(x,y,z)만 추출해서 로드하는 초강력 파서.
    data1-filtered, data2-filtered 전부 대응.
    """
    try:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 숫자 3개 추출 (정수 or 실수)
                nums = re.findall(r"-?\d+\.?\d*", line)
                
                if len(nums) >= 3:                     # 최소 3개는 있어야 x,y,z
                    x, y, z = map(float, nums[:3])      # 앞 3개만 사용
                    rows.append([x, y, z])

        if len(rows) == 0:
            print(f"⚠ Parsing failed: {path}")
            return None

        return np.array(rows)

    except Exception as e:
        print(f"⚠ Error loading {path}: {e}")
        return None



# ======================================================
# 1) SMART TRIM
# ======================================================
def smart_trim(data, threshold=0.5):
    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    active_idx = np.where(vel > threshold)[0]

    if len(active_idx) < 2:
        return data

    start = max(active_idx[0] - 2, 0)
    end = min(active_idx[-1] + 2, len(data))
    return data[start:end]


# ======================================================
# 2) TIME WARP
# ======================================================
def time_warp(data, sigma=0.2):
    T = data.shape[0]
    random_curve = np.random.normal(1.0, sigma, size=T)
    cum = np.cumsum(random_curve)
    cum = cum / cum[-1]
    f = interp1d(cum, data, axis=0, fill_value="extrapolate")
    return f(np.linspace(0, 1, T))


# ======================================================
# 3) ROTATION AROUND Z
# ======================================================
def rotate_z(data, angle_deg):
    angle = np.radians(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return data @ R


# ======================================================
# 4) MAGNITUDE WARP
# ======================================================
def magnitude_warp(data, sigma=0.1):
    T = data.shape[0]
    curve = np.random.normal(1.0, sigma, size=(T, 1))
    return data * curve


# ======================================================
# 5) Data Augmentation
# ======================================================
def augment_data(X_data, y_labels, n_augment=5):
    X_aug, y_aug = [], []

    for i in range(len(X_data)):
        original = X_data[i]
        label = y_labels[i]

        # original append
        X_aug.append(original)
        y_aug.append(label)

        for _ in range(n_augment):
            data = original.copy()

            # Rotation
            angle = np.random.uniform(-10, 10)
            data = rotate_z(data, angle)

            # Time warp
            data = time_warp(data)

            # Magnitude warp
            data = magnitude_warp(data)

            # Scaling
            scale = np.random.uniform(0.9, 1.1)
            data = data * scale

            # Add noise
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise

            X_aug.append(data)
            y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)


# ======================================================
# 6) Main Loader
# ======================================================
def load_and_preprocess_data(root_dir, target_length, augment=False):
    X_raw, y_labels = [], []

    # ------------------------------
    # Automatically detect subfolders: circle, vertical, ...
    # ------------------------------
    folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    if len(folders) == 0:
        print("❌ No motion class folders found in DATA_ROOT:", root_dir)
        return None, None

    # ------------------------------
    # Loop through motion classes
    # ------------------------------
    for motion in folders:
        motion_dir = os.path.join(root_dir, motion)
        csvs = glob.glob(os.path.join(motion_dir, "*.csv"))

        if len(csvs) == 0:
            print(f"⚠ No CSV files in folder: {motion_dir}")
            continue

        for path in csvs:
            data = robust_load_csv(path)
            if data is None:
                continue

            # Smart Trim
            data = smart_trim(data)

            # Resample (interpolate to target length)
            T = data.shape[0]
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, data, axis=0, kind="linear")
            data = f(x_new)

            X_raw.append(data)
            y_labels.append(motion)

    if len(X_raw) == 0:
        print("❌ No data loaded.")
        return None, None

    X_raw = np.array(X_raw)
    y_labels = np.array(y_labels)

    # Low-pass filter
    FS, CUTOFF = 30.0, 3.0
    nyq = FS * 0.5
    b, a = butter(5, CUTOFF / nyq, btype="low")

    X_denoised = np.array([filtfilt(b, a, d, axis=0) for d in X_raw])

    # Augmentation
    if augment:
        X_denoised, y_labels = augment_data(X_denoised, y_labels)

    return X_denoised, y_labels
