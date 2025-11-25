import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


def load_and_preprocess_data(root_dir, target_length):
    X_raw, y_labels = [], []

    # Find class folders
    motion_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    if not motion_folders:
        print("Error: No motion folders found in", root_dir)
        return np.array([]), np.array([])

    # Load CSV files
    for motion in motion_folders:
        folder_path = os.path.join(root_dir, motion)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for csv_path in csv_files:
            try:
                # Fix: Add sep='/' to handle X/Y/Z format
                df = pd.read_csv(csv_path, header=None, sep='/', skip_blank_lines=True)

                # Expect X,Y,Z
                if df.shape[1] < 3:
                    continue

                data = df.iloc[:, :3].values
                
                # Remove any rows with NaN values
                data = data[~np.isnan(data).any(axis=1)]
                
                if len(data) < 2:  # Need at least 2 points for interpolation
                    print(f"Warning: Skipping {csv_path} - insufficient data points")
                    continue

                # Resampling
                original_len = data.shape[0]
                x_old = np.linspace(0, 1, original_len)
                x_new = np.linspace(0, 1, target_length)

                f_interp = interp1d(x_old, data, axis=0, kind='linear')
                data_resampled = f_interp(x_new)

                X_raw.append(data_resampled)
                y_labels.append(motion)

            except Exception as e:
                print(f"Error loading file {csv_path}: {e}")

    if len(X_raw) == 0:
        print("Error: No CSV loaded.")
        return np.array([]), np.array([])

    X_raw = np.array(X_raw)
    y_labels = np.array(y_labels)

    # Denoise with low-pass filter
    FS, CUTOFF = 30.0, 3.0
    X_denoised = np.zeros_like(X_raw)

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        try:
            b, a = butter(order, normal_cutoff, btype="low")
            return filtfilt(b, a, data, axis=0)
        except:
            return data

    for i in range(len(X_raw)):
        X_denoised[i] = butter_lowpass_filter(X_raw[i], CUTOFF, FS)

    return X_denoised, y_labels