import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw

# --- Configuration ---
DATA_ROOT = './csv_data_7'
TARGET_LENGTH = 100 
N_ESTIMATORS = 100 
RANDOM_STATE = 42

# --- 1. Core Functions for Preprocessing ---

def load_and_preprocess_data(root_dir, target_length):
    """
    Loads all noisy time-series data, standardizes length, and applies denoising.

    This function iterates through all CSV files in the motion folders, 
    resamples the time series to a fixed length, and applies a Butterworth 
    Low-Pass Filter to smooth out sensor noise.

    Args:
        root_dir (str): Path to the root directory containing motion folders (e.g., 'csv_data_7').
        target_length (int): The fixed number of time steps to interpolate all series to.

    Returns:
        tuple: (X_denoised, y_labels)
            X_denoised (numpy.ndarray): The clean, standardized 3D array of time series.
            y_labels (numpy.ndarray): The corresponding string labels for each series.
    """
    X_raw, y_labels = [], []
    # ... [Code for loading, resampling, and appending series to X_raw, y_labels]
    
    # Placeholder for actual loading and resampling logic (as defined previously)
    # The loading and resampling logic remains the same for simplicity.
    
    # --- Simplified Data Loading and Resampling Placeholder ---
    # NOTE: Actual implementation requires file access logic (omitted here for brevity)
    # Assume X_raw and y_labels are populated correctly after this section.
    
    # --- Denoising Step ---
    FS, CUTOFF = 30.0, 3.0 # Example parameters
    X_denoised = np.zeros_like(X_raw)
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        """Applies a Butterworth Low-Pass Filter for robust denoising."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        # Use try/except to handle case where X_raw might be empty or invalid during simulation
        try:
             b, a = butter(order, normal_cutoff, btype='low', analog=False)
             return filtfilt(b, a, data, axis=0)
        except ValueError:
             # Return original data if filtering fails (e.g., invalid input)
             return data
    
    if len(X_raw) > 0:
        for i in range(len(X_raw)):
             X_denoised[i] = butter_lowpass_filter(X_raw[i], CUTOFF, FS)
        return X_denoised, np.array(y_labels)
    else:
        # Placeholder for returning empty data if loading fails
        return np.array([]), np.array([])


def extract_features(X_data, y_labels):
    """
    Transforms clean time-series data into a fixed-length vector of interpretable features.

    Features include statistical metrics, dynamic metrics (velocity), and DTW 
    distances to the average template of each motion class.

    Args:
        X_data (numpy.ndarray): The clean, standardized 3D array of time series.
        y_labels (numpy.ndarray): The corresponding string labels for each series.

    Returns:
        tuple: (X_features, feature_names)
            X_features (numpy.ndarray): 2D matrix of extracted features.
            feature_names (list): List of feature names for explainability output.
    """
    # ... [Code for calculating all features: Statistical, Dynamic, and DTW]
    # The feature calculation logic remains the same (omitted here for brevity)
    
    # --- Simplified Feature Extraction Placeholder ---
    # Create mock features and names for demonstration purposes
    N_SAMPLES, N_TIME_STEPS, N_DIMENSIONS = X_data.shape
    
    # Mock Feature 1: Mean Value of Dimension 0
    X_feat_mean_d0 = np.mean(X_data[:, :, 0], axis=1).reshape(-1, 1)
    
    # Mock Feature 2: DTW distance to a 'circle' template (Requires DTW calculation)
    # This feature is crucial for explainability (quality of motion)
    
    # Mock Feature 3: Max velocity (Requires derivative calculation)
    velocity = np.diff(X_data, axis=1) 
    X_feat_max_vel = np.max(np.abs(velocity), axis=(1, 2)).reshape(-1, 1)

    X_features = np.hstack([X_feat_mean_d0, X_feat_max_vel])
    feature_names = ['Mean_DIM_0', 'Max_Velocity_All']
    
    # Note: In the final version, this needs to include all features (DTW, Stat, Dynamic)
    
    return X_features, feature_names


def train_and_explain_model(X_features, y_labels, feature_names, n_estimators, random_state):
    """
    Trains the Random Forest model and extracts Feature Importance for explanation.

    Args:
        X_features (numpy.ndarray): The 2D matrix of extracted features.
        y_labels (numpy.ndarray): The corresponding string labels.
        feature_names (list): List of feature names.
        n_estimators (int): Number of trees in the Random Forest.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame showing the sorted Feature Importance, 
                          which serves as the model's explanation.
    """
    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # --- Cross-Validation and Training ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    feature_importances_list = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_labels)):
        X_train = X_scaled[train_index]
        y_train = y_labels[train_index]
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
        model.fit(X_train, y_train)
        
        feature_importances_list.append(model.feature_importances_)

    # --- Explainability Output ---
    mean_importances = np.mean(feature_importances_list, axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

# --- 4. Main Program Flow ---

def main_program(data_root, target_length, n_estimators):
    """
    Main function to run the Explainable Motion Classification pipeline.
    """
    
    # 1. LOAD, STANDARDIZE, and DENOISE DATA
    X_denoised, y = load_and_preprocess_data(data_root, target_length)

    if X_denoised.size == 0:
        print("Error: Data loading failed or resulted in empty set.")
        return

    # 2. FEATURE ENGINEERING
    X_features, feature_names = extract_features(X_denoised, y)

    # 3. TRAIN & EXPLAIN
    importance_table = train_and_explain_model(X_features, y, feature_names, n_estimators, RANDOM_STATE)

    print("\n--- Model Summary and Explanation ---")
    print(f"Total samples processed: {X_features.shape[0]}")
    print("\nTop 5 Most Important Features (The Model's Explanation):")
    print(importance_table.head(5).to_markdown(index=False, numalign="left"))


if __name__ == '__main__':
    # NOTE: You must update DATA_ROOT to point to your 'csv_data_7' folder before running.
    main_program(DATA_ROOT, TARGET_LENGTH, N_ESTIMATORS)
    print("Code is structured. Please uncomment and fix placeholders/paths to run.")