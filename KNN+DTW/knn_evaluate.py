"""
KNN + DTW Evaluation Script
Evaluates a trained KNN model with DTW distance metric on a dataset.
Generates accuracy metrics, confusion matrices, and visualizations.
"""

# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = "data/processed/dataset1_norm.pkl"
MODEL_PATH = "KNN+DTW/models/dataset1.pkl"
OUTPUT_DIR = "KNN+DTW/evaluation"

# Cross-validation settings
CV_FOLDS = 5
K_VALUES_TO_TEST = [1, 3, 5, 7, 9]

# ============================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


def load_processed(filepath):
    """Load saved dataset"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["sequences"], data["labels"], data["label_map"], data["file_ids"]


def load_model(filepath):
    """Load trained model and label map"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_map"]


def get_output_dir(model_path, base_output_dir):
    """Generate output directory from model name"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(base_output_dir, model_name)


def prepare_for_tslearn(sequences):
    """Convert list of variable-length arrays to tslearn format"""
    max_len = max(len(s) for s in sequences)
    n_features = sequences[0].shape[1]
    X = np.full((len(sequences), max_len, n_features), np.nan)
    for i, seq in enumerate(sequences):
        X[i, :len(seq), :] = seq
    return X


def plot_confusion_matrix(y_true, y_pred, label_map, save_path=None):
    """Plot and optionally save confusion matrix"""
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, label_map, save_path=None):
    """Plot confusion matrix normalized by true labels (recall per class)"""
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f')
    plt.title('Normalized Confusion Matrix (Recall)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()
    return cm


def plot_k_comparison(k_values, scores, save_path=None):
    """Plot accuracy vs k value"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    means = [np.mean(s) for s in scores]
    stds = [np.std(s) for s in scores]
    
    ax.errorbar(k_values, means, yerr=stds, marker='o', capsize=5, capthick=2)
    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Cross-Validation Accuracy vs k', fontsize=14)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Mark best k
    best_idx = np.argmax(means)
    ax.annotate(f'Best: k={k_values[best_idx]}\n({means[best_idx]:.1%})',
                xy=(k_values[best_idx], means[best_idx]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_per_class_metrics(y_true, y_pred, label_map, save_path=None):
    """Plot precision, recall, F1 per class"""
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='darkorange')
    ax.bar(x + width, f1, width, label='F1-Score', color='green')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()


def find_misclassified(y_true, y_pred, file_ids, label_map):
    """Find and return misclassified samples"""
    idx_to_label = {v: k for k, v in label_map.items()}
    
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'file_id': file_ids[i],
                'true_label': idx_to_label[true],
                'predicted': idx_to_label[pred],
            })
    
    return misclassified


def run_k_comparison(X, y, k_values, n_folds=5):
    """Run cross-validation for different k values"""
    print(f"\nTesting k values: {k_values}")
    print(f"Using {n_folds}-fold cross-validation")
    print("-" * 40)
    
    all_scores = []
    
    for k in k_values:
        model = KNeighborsTimeSeriesClassifier(
            n_neighbors=k,
            metric="dtw",
            metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 10},
            weights="distance",
            n_jobs=-1,
        )
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        all_scores.append(scores)
        
        print(f"k={k}: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")
    
    return all_scores


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Generate output directory from model name
    output_dir = get_output_dir(MODEL_PATH, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input:  {INPUT_FILE}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Output: {output_dir}")
    
    # ------------------------------------------
    # 1. Load data and model
    # ------------------------------------------
    print("\n[1/5] Loading data and model...")
    sequences, labels, label_map, file_ids = load_processed(INPUT_FILE)
    model, _ = load_model(MODEL_PATH)
    
    X = prepare_for_tslearn(sequences)
    y = labels
    
    idx_to_label = {v: k for k, v in label_map.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    
    print(f"Samples: {len(sequences)}")
    print(f"Classes: {target_names}")
    
    # ------------------------------------------
    # 2. Full dataset evaluation (using model's training)
    # ------------------------------------------
    print("\n[2/5] Evaluating on full dataset...")
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names))
    
    # ------------------------------------------
    # 3. Cross-validation with different k values
    # ------------------------------------------
    print("\n[3/5] Cross-validation comparison...")
    cv_scores = run_k_comparison(X, y, K_VALUES_TO_TEST, n_folds=CV_FOLDS)
    
    # Plot k comparison
    plot_k_comparison(
        K_VALUES_TO_TEST, cv_scores,
        save_path=os.path.join(output_dir, "k_comparison.png")
    )
    
    # ------------------------------------------
    # 4. Visualizations
    # ------------------------------------------
    print("\n[4/5] Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y, y_pred, label_map,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Normalized confusion matrix
    plot_normalized_confusion_matrix(
        y, y_pred, label_map,
        save_path=os.path.join(output_dir, "confusion_matrix_normalized.png")
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        y, y_pred, label_map,
        save_path=os.path.join(output_dir, "per_class_metrics.png")
    )
    
    # ------------------------------------------
    # 5. Error analysis
    # ------------------------------------------
    print("\n[5/5] Error analysis...")
    misclassified = find_misclassified(y, y_pred, file_ids, label_map)
    
    if misclassified:
        print(f"\nMisclassified samples ({len(misclassified)}):")
        print("-" * 60)
        for item in misclassified[:20]:  # Show first 20
            print(f"  {item['file_id']}")
            print(f"    True: {item['true_label']}, Predicted: {item['predicted']}")
        
        if len(misclassified) > 20:
            print(f"  ... and {len(misclassified) - 20} more")
        
        # Save full list
        error_path = os.path.join(output_dir, "misclassified.txt")
        with open(error_path, 'w') as f:
            for item in misclassified:
                f.write(f"{item['file_id']}\t{item['true_label']}\t{item['predicted']}\n")
        print(f"\nFull list saved: {error_path}")
    else:
        print("No misclassified samples!")
    
    # ------------------------------------------
    # Summary
    # ------------------------------------------
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Errors: {len(misclassified)} / {len(y)}")
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - per_class_metrics.png")
    print("  - k_comparison.png")
    if misclassified:
        print("  - misclassified.txt")