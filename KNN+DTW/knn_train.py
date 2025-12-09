"""
KNN + DTW Training Script
Trains a K-Nearest Neighbors classifier using Dynamic Time Warping (DTW) as the distance metric.
Saves the trained model and label mapping for future predictions."""

# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = "data/processed/data1-filtered_norm.pkl"
OUTPUT_DIR = "KNN+DTW/models"

N_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Set high enough for longest expected sequence (None = use max from data)
MAX_SEQ_LEN = 1000

# DTW constraints (optional, speeds up computation)
DTW_CONSTRAINT = "sakoe_chiba"
SAKOE_CHIBA_RADIUS = 10

# ============================================

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


def load_processed(filepath):
    """Load saved dataset"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["sequences"], data["labels"], data["label_map"], data["file_ids"]


def prepare_for_tslearn(sequences, max_len=None):
    """
    Convert list of variable-length arrays to tslearn format.
    Pads sequences to max_len (or longest sequence if not specified).
    """
    data_max_len = max(len(s) for s in sequences)
    
    # Use specified max_len or data's max
    if max_len is None:
        max_len = data_max_len
    elif max_len < data_max_len:
        print(f"WARNING: max_len ({max_len}) < longest sequence ({data_max_len}), using {data_max_len}")
        max_len = data_max_len
    
    n_features = sequences[0].shape[1]
    
    # Create padded array (NaN padding for tslearn)
    X = np.full((len(sequences), max_len, n_features), np.nan)
    
    for i, seq in enumerate(sequences):
        X[i, :len(seq), :] = seq
    
    return X


def get_output_path(input_file, output_dir):
    """Generate output model path from input filename"""
    base_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(base_name)[0]
    
    for suffix in ['_norm', '_raw']:
        if name_without_ext.endswith(suffix):
            name_without_ext = name_without_ext[:-len(suffix)]
            break
    
    return os.path.join(output_dir, f"knn-{name_without_ext}.pkl")


def save_model(model, label_map, max_seq_len, filepath):
    """Save model and metadata"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": model,
            "label_map": label_map,
            "max_seq_len": max_seq_len,
        }, f)
    print(f"Model saved: {filepath}")


def print_confusion_matrix(y_true, y_pred, label_map):
    """Print formatted confusion matrix"""
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    
    header = "Actual \\ Pred".ljust(15) + "".join(l[:8].ljust(10) for l in labels)
    print(header)
    print("-" * 60)
    
    for i, row in enumerate(cm):
        row_str = labels[i][:12].ljust(15) + "".join(str(v).ljust(10) for v in row)
        print(row_str)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("KNN + DTW TRAINING")
    print("=" * 50)
    
    model_output = get_output_path(INPUT_FILE, OUTPUT_DIR)
    
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {model_output}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    
    # ------------------------------------------
    # 1. Load data
    # ------------------------------------------
    print("\n[1/4] Loading data...")
    sequences, labels, label_map, file_ids = load_processed(INPUT_FILE)
    print(f"Loaded {len(sequences)} samples, {len(label_map)} classes")
    
    seq_lengths = [len(s) for s in sequences]
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.0f}")
    
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # ------------------------------------------
    # 2. Train/test split
    # ------------------------------------------
    print("\n[2/4] Splitting data...")
    
    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    print(f"Train: {len(seq_train)}, Test: {len(seq_test)}")
    
    # Convert to tslearn format with fixed max_len
    X_train = prepare_for_tslearn(seq_train, max_len=MAX_SEQ_LEN)
    X_test = prepare_for_tslearn(seq_test, max_len=MAX_SEQ_LEN)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # ------------------------------------------
    # 3. Train KNN + DTW
    # ------------------------------------------
    print("\n[3/4] Training KNN with DTW...")
    print(f"Parameters: k={N_NEIGHBORS}, constraint={DTW_CONSTRAINT}, radius={SAKOE_CHIBA_RADIUS}")
    
    model = KNeighborsTimeSeriesClassifier(
        n_neighbors=N_NEIGHBORS,
        metric="dtw",
        metric_params={
            "global_constraint": DTW_CONSTRAINT,
            "sakoe_chiba_radius": SAKOE_CHIBA_RADIUS,
        },
        weights="distance",
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # ------------------------------------------
    # 4. Evaluate
    # ------------------------------------------
    print("\n[4/4] Evaluating on test set...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print_confusion_matrix(y_test, y_pred, label_map)
    
    # ------------------------------------------
    # 5. Save model
    # ------------------------------------------
    print("\n" + "=" * 50)
    save_model(model, label_map, MAX_SEQ_LEN, model_output)
    
    print(f"\nModel supports sequences up to {MAX_SEQ_LEN} timesteps")