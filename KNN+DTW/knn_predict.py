"""KNN + DTW Prediction Script
Predicts gesture classes using a trained KNN model with DTW distance metric.
Loads CSV files containing time series data and outputs predicted classes."""

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "KNN+DTW/models/knn_1.pkl"

# ============================================

import pickle
import numpy as np
import pandas as pd


def load_model(filepath):
    """Load trained model and label map"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_map"]


def normalize_sequence(seq):
    """Per-sample normalization: zero mean, unit std"""
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    std[std == 0] = 1
    return (seq - mean) / std


def prepare_single_sample(seq, max_len):
    """Prepare single sequence for prediction (pad to max_len)"""
    n_features = seq.shape[1]
    X = np.full((1, max_len, n_features), np.nan)
    X[0, :len(seq), :] = seq
    return X


def predict_csv(csv_path, model, label_map, return_proba=False):
    """
    Predict class for a single CSV file.
    
    Args:
        csv_path: Path to CSV with x,y,z columns (no header)
        model: Trained KNN model
        label_map: {class_name: index}
        return_proba: If True, return neighbor voting distribution
    
    Returns:
        predicted_class: Class name string
        confidence: Voting confidence (if return_proba=True)
    """
    # Load and normalize (no header in CSV)
    df = pd.read_csv(csv_path, header=None, names=['x', 'y', 'z'])
    seq = df[['x', 'y', 'z']].values.astype(np.float32)
    seq_norm = normalize_sequence(seq)
    
    # Get max_len from model's training data
    max_len = model._X_fit.shape[1]
    
    # Prepare for prediction
    X = prepare_single_sample(seq_norm, max_len)
    
    # Predict
    pred_idx = model.predict(X)[0]
    
    # Map index to class name
    idx_to_label = {v: k for k, v in label_map.items()}
    predicted_class = idx_to_label[pred_idx]
    
    if return_proba:
        # Get k nearest neighbors and their distances
        distances, indices = model.kneighbors(X)
        neighbor_labels = model._y[indices[0]]
        
        # Calculate confidence as proportion voting for predicted class
        votes = np.bincount(neighbor_labels, minlength=len(label_map))
        confidence = votes[pred_idx] / len(neighbor_labels)
        
        return predicted_class, confidence, votes
    
    return predicted_class


def predict_sequence(seq, model, label_map):
    """
    Predict class for a numpy array directly.
    
    Args:
        seq: numpy array of shape (n_timesteps, 3)
        model: Trained KNN model
        label_map: {class_name: index}
    
    Returns:
        predicted_class: Class name string
    """
    seq_norm = normalize_sequence(seq.astype(np.float32))
    max_len = model._X_fit.shape[1]
    X = prepare_single_sample(seq_norm, max_len)
    
    pred_idx = model.predict(X)[0]
    idx_to_label = {v: k for k, v in label_map.items()}
    
    return idx_to_label[pred_idx]


def predict_batch(csv_paths, model, label_map):
    """
    Predict classes for multiple CSV files.
    
    Returns:
        List of (filepath, predicted_class, confidence)
    """
    results = []
    for path in csv_paths:
        pred_class, confidence, _ = predict_csv(path, model, label_map, return_proba=True)
        results.append((path, pred_class, confidence))
    return results


# ============================================
# MAIN (Demo usage)
# ============================================

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("GESTURE PREDICTION")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model, label_map = load_model(MODEL_PATH)
    print(f"Classes: {list(label_map.keys())}")
    
    # Check for input file
    if len(sys.argv) < 2:
        print("\nUsage: python knn_predict.py <path_to_csv>")
        print("       python knn_predict.py <csv1> <csv2> <csv3> ...")
        print("\nExample:")
        print("  python knn_predict.py data/test_gesture.csv")
        sys.exit(1)
    
    csv_paths = sys.argv[1:]
    
    # Single file
    if len(csv_paths) == 1:
        print(f"\nPredicting: {csv_paths[0]}")
        predicted_class, confidence, votes = predict_csv(
            csv_paths[0], model, label_map, return_proba=True
        )
        
        print(f"\n{'='*30}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'='*30}")
        
        # Show vote distribution
        idx_to_label = {v: k for k, v in label_map.items()}
        print("\nNeighbor votes:")
        for idx, count in enumerate(votes):
            if count > 0:
                print(f"  {idx_to_label[idx]}: {count}")
    
    # Multiple files
    else:
        print(f"\nPredicting {len(csv_paths)} files...")
        results = predict_batch(csv_paths, model, label_map)
        
        print(f"\n{'File':<40} {'Prediction':<15} {'Confidence'}")
        print("-" * 65)
        for path, pred, conf in results:
            filename = path.split('/')[-1][:38]
            print(f"{filename:<40} {pred:<15} {conf:.1%}")