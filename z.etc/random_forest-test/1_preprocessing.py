import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


def augment_data(X_data, y_labels, n_augment=5):
    X_aug, y_aug = [], []

    for i in range(len(X_data)):
        X_aug.append(X_data[i])
        y_aug.append(y_labels[i])

        for _ in range(n_augment):
            scale = np.random.uniform(0.9, 1.1)
            X_scaled = X_data[i] * scale

            noise = np.random.normal(0, 0.02, X_scaled.shape)
            X_final = X_scaled + noise

            X_aug.append(X_final)
            y_aug.append(y_labels[i])

    return np.array(X_aug), np.array(y_aug)


def load_and_preprocess_data(root_dir, target_length, augment=False):
    X_raw, y_labels = [], []

    motion_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    if not motion_folders:
        return np.array([]), np.array([])

    for motion in motion_folders:
        folder_path = os.path.join(root_dir, motion)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, header=None, sep='/', skip_blank_lines=True)

                if df.shape[1] < 3:
                    continue

                data = df.iloc[:, :3].values
                data = data[~np.isnan(data).any(axis=1)]

                if len(data) < 2:
                    continue

                original_len = data.shape[0]
                x_old = np.linspace(0, 1, original_len)
                x_new = np.linspace(0, 1, target_length)

                f_interp = interp1d(x_old, data, axis=0, kind='linear')
                data_resampled = f_interp(x_new)

                X_raw.append(data_resampled)
                y_labels.append(motion)

            except Exception:
                continue

    if len(X_raw) == 0:
        return np.array([]), np.array([])

    X_raw = np.array(X_raw)
    y_labels = np.array(y_labels)

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

    if augment:
        X_denoised, y_labels = augment_data(X_denoised, y_labels)

    return X_denoised, y_labels