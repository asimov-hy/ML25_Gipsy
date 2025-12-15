import pickle
import numpy as np
import pandas as pd
import sys
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


def load_model():
    """Load the trained model and scaler"""
    if not os.path.exists('trained_model.pkl'):
        print("❌ Error: 'trained_model.pkl' not found!")
        print("   Please run '4_main.py' first to train the model.")
        sys.exit(1)
    
    with open('trained_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def preprocess_new_data(csv_path, target_length=100):
    """
    Preprocess a single CSV file the same way as training data
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path, header=None, sep='/', skip_blank_lines=True)
        df = df.replace('', np.nan).dropna()
        
        if df.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns (X, Y, Z)")
        
        data = df.iloc[:, :3].values.astype(float)
        data = data[~np.isnan(data).any(axis=1)]
        
        if len(data) < 2:
            raise ValueError("Insufficient data points")
        
        # Resample to target length
        original_len = data.shape[0]
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, target_length)
        f_interp = interp1d(x_old, data, axis=0, kind='linear')
        data_resampled = f_interp(x_new)
        
        # Apply low-pass filter
        FS, CUTOFF = 30.0, 3.0
        nyquist = 0.5 * FS
        normal_cutoff = CUTOFF / nyquist
        b, a = butter(5, normal_cutoff, btype="low")
        data_denoised = filtfilt(b, a, data_resampled, axis=0)
        
        return data_denoised
        
    except Exception as e:
        print(f"Error preprocessing {csv_path}: {e}")
        return None


def extract_features_single(data):
    """
    Extract features from a single sample (same as training)
    Input: (T, D) array where T=timesteps, D=dimensions
    """
    # Mean of dimension 0
    feat_mean_x = np.mean(data[:, 0])
    
    # Max velocity
    velocity = np.diff(data, axis=0)
    feat_max_vel = np.max(np.abs(velocity))
    
    return np.array([feat_mean_x, feat_max_vel]).reshape(1, -1)


def predict_motion(csv_path):
    """
    Predict the motion class for a new CSV file
    """
    # Load trained model
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    classes = model_data['classes']
    
    # Preprocess new data
    print(f"\nProcessing: {csv_path}")
    data = preprocess_new_data(csv_path)
    
    if data is None:
        return None
    
    # Extract features
    features = extract_features_single(data)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Display results
    print("\n" + "="*50)
    print(f"✅ Predicted Class: {prediction}")
    print("="*50)
    print("Class Probabilities:")
    for cls, prob in zip(classes, probabilities):
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s}: {prob:.4f} {bar}")
    print("="*50)
    
    return prediction, probabilities


def predict_multiple(csv_paths):
    """
    Predict classes for multiple CSV files
    """
    model_data = load_model()
    results = []
    
    print("\n" + "="*60)
    print(f"CLASSIFYING {len(csv_paths)} FILES")
    print("="*60)
    
    for i, csv_path in enumerate(csv_paths, 1):
        print(f"\n[{i}/{len(csv_paths)}]", end=" ")
        result = predict_motion(csv_path)
        if result:
            results.append({
                'file': csv_path,
                'prediction': result[0],
                'probabilities': result[1]
            })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{os.path.basename(r['file']):30s} → {r['prediction']}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("MOTION CLASSIFIER - PREDICTION")
    print("="*60)
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        # Use command line arguments
        csv_files = sys.argv[1:]
        
        if len(csv_files) == 1:
            predict_motion(csv_files[0])
        else:
            predict_multiple(csv_files)
    
    else:
        # Default: test with existing data
        print("\nNo input files provided. Testing with sample data...")
        print("Usage: python 5_classifier_test.py <csv_file1> [csv_file2] ...")
        
        # Try to find a test file
        test_file = "./csv_data_7/circle/1.csv"
        
        if os.path.exists(test_file):
            print(f"\nUsing sample file: {test_file}")
            predict_motion(test_file)
        else:
            print("\n❌ No test file found. Please provide a CSV file:")
            print("   python 5_classifier_test.py path/to/your/file.csv")