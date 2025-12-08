# ============================================
# CONFIGURATION
# ============================================

INPUT_DIRS = [
    "data/data1-filtered",
]

OUTPUT_DIR = "data/processed"

# ============================================

import os
import pickle
import numpy as np
import pandas as pd
from glob import glob


def normalize_sequence(seq):
    """Per-sample normalization: zero mean, unit std"""
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    std[std == 0] = 1
    return (seq - mean) / std


def load_dataset(input_dirs):
    """
    Load from folder structure:
        input_dir/
            circle/
                1.csv
            vertical/
                1.csv
            ...
    
    Returns:
        sequences_raw: List of raw arrays
        sequences_norm: List of normalized arrays
        labels: np.array of class indices
        label_map: {class_name: index}
        file_ids: List of identifiers
    """
    sequences_raw = []
    sequences_norm = []
    labels = []
    file_ids = []
    
    # Handle single path or list
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    # Gather all CSVs
    all_filepaths = []
    for data_dir in input_dirs:
        filepaths = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        all_filepaths.extend(filepaths)
    
    if not all_filepaths:
        raise ValueError(f"No CSV files found in {input_dirs}")
    
    # Class labels from folder names
    classes = sorted(set(os.path.basename(os.path.dirname(f)) for f in all_filepaths))
    label_map = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Found {len(all_filepaths)} files")
    print(f"Classes: {label_map}")
    
    for filepath in all_filepaths:
        df = pd.read_csv(filepath, header=None, names=['x', 'y', 'z'])
        seq_raw = df[['x', 'y', 'z']].values.astype(np.float32)
        seq_norm = normalize_sequence(seq_raw.copy())
        
        class_name = os.path.basename(os.path.dirname(filepath))
        label = label_map[class_name]
        file_id = f"{class_name}/{os.path.splitext(os.path.basename(filepath))[0]}"
        
        sequences_raw.append(seq_raw)
        sequences_norm.append(seq_norm)
        labels.append(label)
        file_ids.append(file_id)
    
    return sequences_raw, sequences_norm, np.array(labels), label_map, file_ids


def get_output_name(input_dirs):
    """Generate output base name from input directories"""
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    # Use folder names, joined with +
    names = [os.path.basename(d.rstrip('/')) for d in input_dirs]
    return "+".join(names)


def save_dataset(output_file, sequences, labels, label_map, file_ids):
    """Save to pickle"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump({
            "sequences": sequences,
            "labels": labels,
            "label_map": label_map,
            "file_ids": file_ids,
        }, f)
    print(f"Saved: {output_file}")


def load_processed(filepath):
    """Load saved dataset"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["sequences"], data["labels"], data["label_map"], data["file_ids"]


def print_stats(sequences, labels, label_map):
    """Print dataset statistics"""
    idx_to_label = {v: k for k, v in label_map.items()}
    seq_lengths = [len(s) for s in sequences]
    
    print(f"\nTotal: {len(sequences)} samples")
    print(f"Lengths: {min(seq_lengths)}-{max(seq_lengths)} (avg {np.mean(seq_lengths):.0f})")
    print("\nPer class:")
    for idx in sorted(idx_to_label.keys()):
        name = idx_to_label[idx]
        count = sum(labels == idx)
        print(f"  {name}: {count}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 40)
    print("DATA LOADER")
    print("=" * 40)
    
    # Generate output paths
    base_name = get_output_name(INPUT_DIRS)
    output_raw = os.path.join(OUTPUT_DIR, f"{base_name}_raw.pkl")
    output_norm = os.path.join(OUTPUT_DIR, f"{base_name}_norm.pkl")
    
    print(f"Input:  {INPUT_DIRS}")
    print(f"Output: {output_raw}")
    print(f"        {output_norm}")
    
    # Load
    seq_raw, seq_norm, labels, label_map, file_ids = load_dataset(INPUT_DIRS)
    
    # Stats
    print_stats(seq_raw, labels, label_map)
    
    # Save both
    print()
    save_dataset(output_raw, seq_raw, labels, label_map, file_ids)
    save_dataset(output_norm, seq_norm, labels, label_map, file_ids)
    
    print("\nUse:")
    print(f"  Geometric classifier: {output_raw}")
    print(f"  KNN / ML:             {output_norm}")