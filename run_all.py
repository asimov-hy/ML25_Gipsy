"""
Run All Pipeline Script
Processes raw text files through filter -> predict pipeline.
Outputs prediction results to text files.
"""

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = "data/RAW_data/data2-RAW"
OUTPUT_DIR = "predictions"
MODEL_PATH = "KNN+DTW/models/knn-data1-filtered.pkl"

TEMP_DIR = "data/temp"

# ============================================

import os
import sys
import shutil
import numpy as np
import pandas as pd
import pickle
from glob import glob


# ============================================
# FILTER FUNCTIONS
# ============================================

def extract_xyz(line):
    """Extract x,y,z from line like: r,39534,...,392/-440/-84,...,#"""
    line = line.strip()
    
    if not line or not line.startswith("r,"):
        return None
    
    fields = line.split(",")
    
    if len(fields) < 7:
        return None
    
    xyz_field = fields[6]
    parts = xyz_field.split("/")
    
    if len(parts) != 3:
        return None
    
    try:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return x, y, z
    except ValueError:
        return None


def filter_file(input_path, output_path):
    """Process single txt file -> csv with x,y,z"""
    data = []
    
    with open(input_path, "r") as f:
        for line in f:
            result = extract_xyz(line)
            if result:
                data.append(result)
    
    if not data:
        return 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for x, y, z in data:
            f.write(f"{x},{y},{z}\n")
    
    return len(data)


# ============================================
# PREDICT FUNCTIONS
# ============================================

def load_model(filepath):
    """Load trained model, label map, and max sequence length"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Handle old models without max_seq_len saved
    max_seq_len = data.get("max_seq_len", data["model"]._X_fit.shape[1])
    
    return data["model"], data["label_map"], max_seq_len


def normalize_sequence(seq):
    """Per-sample normalization: zero mean, unit std"""
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    std[std == 0] = 1
    return (seq - mean) / std


def prepare_single_sample(seq, max_len):
    """Prepare single sequence for prediction (pad or truncate to max_len)"""
    n_features = seq.shape[1]
    X = np.full((1, max_len, n_features), np.nan)
    
    seq_len = min(len(seq), max_len)
    X[0, :seq_len, :] = seq[:seq_len]
    
    return X


def predict_csv(csv_path, model, label_map, max_seq_len):
    """
    Predict class for a single CSV file.
    
    Returns:
        predicted_class, confidence, seq_length, was_truncated
    """
    df = pd.read_csv(csv_path, header=None, names=['x', 'y', 'z'])
    seq = df[['x', 'y', 'z']].values.astype(np.float32)
    seq_norm = normalize_sequence(seq)
    
    seq_length = len(seq_norm)
    was_truncated = seq_length > max_seq_len
    
    X = prepare_single_sample(seq_norm, max_seq_len)
    
    pred_idx = model.predict(X)[0]
    
    idx_to_label = {v: k for k, v in label_map.items()}
    predicted_class = idx_to_label[pred_idx]
    
    distances, indices = model.kneighbors(X)
    neighbor_labels = model._y[indices[0]]
    votes = np.bincount(neighbor_labels, minlength=len(label_map))
    confidence = votes[pred_idx] / len(neighbor_labels)
    
    return predicted_class, confidence, seq_length, was_truncated


# ============================================
# PIPELINE
# ============================================

def process_file(input_path, output_path, temp_csv_path, model, label_map, max_seq_len):
    """Process single raw file through full pipeline."""
    n_samples = filter_file(input_path, temp_csv_path)
    
    if n_samples == 0:
        return None
    
    predicted_class, confidence, seq_length, was_truncated = predict_csv(
        temp_csv_path, model, label_map, max_seq_len
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(predicted_class)
    
    return predicted_class, confidence, seq_length, was_truncated


def run_pipeline(input_dir, output_dir, model_path, temp_dir):
    """Run full pipeline on all files in input directory"""
    
    print("=" * 50)
    print("RUN ALL PIPELINE")
    print("=" * 50)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {model_path}")
    print("-" * 50)
    
    print("\nLoading model...")
    model, label_map, max_seq_len = load_model(model_path)
    print(f"Classes: {list(label_map.keys())}")
    print(f"Max sequence length: {max_seq_len}")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    txt_files = glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        print(f"\nNo .txt files found in {input_dir}")
        return
    
    print(f"\nProcessing {len(txt_files)} files...")
    print("-" * 50)
    
    results = []
    truncated_count = 0
    
    for input_path in txt_files:
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        temp_csv_path = os.path.join(temp_dir, relative_path.replace(".txt", ".csv"))
        
        result = process_file(input_path, output_path, temp_csv_path, model, label_map, max_seq_len)
        
        if result:
            pred_class, confidence, seq_length, was_truncated = result
            results.append((relative_path, pred_class, confidence, seq_length))
            
            if was_truncated:
                truncated_count += 1
                print(f"  {relative_path}: {pred_class} ({confidence:.0%}) [TRUNCATED {seq_length}->{max_seq_len}]")
            else:
                print(f"  {relative_path}: {pred_class} ({confidence:.0%}) [{seq_length} steps]")
        else:
            print(f"  {relative_path}: FAILED (no valid data)")
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Processed: {len(results)} / {len(txt_files)} files")
    
    if truncated_count > 0:
        print(f"WARNING: {truncated_count} files truncated (longer than {max_seq_len})")
        print(f"         Consider retraining with larger MAX_SEQ_LEN")
    
    print(f"Output saved to: {output_dir}/")
    
    if results:
        seq_lengths = [r[3] for r in results]
        print(f"\nSequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.0f}")
        
        print("\nPrediction distribution:")
        class_counts = {}
        for _, pred_class, _, _ in results:
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
    
    return results


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        INPUT_DIR = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    if len(sys.argv) >= 4:
        MODEL_PATH = sys.argv[3]
    
    run_pipeline(INPUT_DIR, OUTPUT_DIR, MODEL_PATH, TEMP_DIR)