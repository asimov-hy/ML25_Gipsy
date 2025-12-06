# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = "data/processed/dataset1_norm.pkl"
MODEL_OUTPUT = "KNN+DTW/models/knn_dtw_model1.pkl"

N_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# DTW constraints (optional, speeds up computation)
DTW_CONSTRAINT = "sakoe_chiba"
SAKOE_CHIBA_RADIUS = 10  # Adjust based on your sequence lengths

# ============================================

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def load_processed(filepath):
    """Load saved dataset"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["sequences"], data["labels"], data["label_map"], data["file_ids"]


def prepare_for_tslearn(sequences):
    """
    Convert list of variable-length arrays to tslearn format.
    Pads sequences to same length.
    """
    max_len = max(len(s) for s in sequences)
    n_features = sequences[0].shape[1]
    
    # Create padded array (NaN padding for tslearn)
    X = np.full((len(sequences), max_len, n_features), np.nan)
    
    for i, seq in enumerate(sequences):
        X[i, :len(seq), :] = seq
    
    return X


def save_model(model, label_map, filepath):
    """Save model and metadata"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": model,
            "label_map": label_map,
        }, f)
    print(f"Model saved: {filepath}")


def print_confusion_matrix(y_true, y_pred, label_map):
    """Print formatted confusion matrix"""
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    
    # Header
    header = "Actual \\ Pred".ljust(15) + "".join(l[:8].ljust(10) for l in labels)
    print(header)
    print("-" * 60)
    
    # Rows
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
    
    # ------------------------------------------
    # 1. Load data
    # ------------------------------------------
    print("\n[1/4] Loading data...")
    sequences, labels, label_map, file_ids = load_processed(INPUT_FILE)
    print(f"Loaded {len(sequences)} samples, {len(label_map)} classes")
    
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
    
    # Convert to tslearn format
    X_train = prepare_for_tslearn(seq_train)
    X_test = prepare_for_tslearn(seq_test)
    
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
        weights="distance",  # Weight by inverse distance
        n_jobs=-1,           # Use all CPU cores
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
    
    # Classification report
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    print_confusion_matrix(y_test, y_pred, label_map)
    
    # ------------------------------------------
    # 5. Save model
    # ------------------------------------------
    print("\n" + "=" * 50)
    save_model(model, label_map, MODEL_OUTPUT)
    
    print("\nUsage:")
    print("  from train_knn_dtw import load_model")
    print("  model, label_map = load_model('models/knn_dtw_model.pkl')")