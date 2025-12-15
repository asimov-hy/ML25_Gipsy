# ============================================
# CONFIGURATION
# ============================================

DATASET_FILE = "data/processed/dataset1+2_raw.pkl"

# ============================================

import numpy as np
import pickle


def load_processed(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["sequences"], data["labels"], data["label_map"], data["file_ids"]


class ShapeClassifier:
    
    def classify(self, points):
        points = np.array(points)
        
        # Remove consecutive duplicates
        mask = np.any(np.diff(points, axis=0) != 0, axis=1)
        mask = np.concatenate([[True], mask])
        points = points[mask]
        
        if len(points) < 5:
            return "unknown"
        
        # --- Linearity (SVD) ---
        centered = points - points.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        linearity = s[0] / (s.sum() + 1e-8)
        
        # --- Axis ranges ---
        ranges = points.max(axis=0) - points.min(axis=0)
        x_range, y_range, z_range = ranges
        total = x_range + y_range + z_range + 1e-8
        
        y_ratio = y_range / total
        z_ratio = z_range / total
        
        # Circle: lowest linearity (spread in 2D)
        if linearity < 0.70:
            return "circle"
        
        # Vertical: Z dominant, Y small
        if z_ratio > 0.55 and y_ratio < 0.20:
            return "vertical"
        
        # Horizontal: Y dominant, Z small
        if y_ratio > 0.55 and z_ratio < 0.15:
            return "horizontal"
        
        # Diagonal: Y and Z both significant
        if y_ratio > 0.30 and z_ratio > 0.30:
            corr = np.corrcoef(points[:, 1], points[:, 2])[0, 1]
            if corr < 0:
                return "diagonal_right"
            else:
                return "diagonal_left"
        
        return "unknown"
    
    def classify_batch(self, sequences):
        return [self.classify(seq) for seq in sequences]


def evaluate(sequences, labels, label_map, file_ids):
    idx_to_label = {v: k for k, v in label_map.items()}
    classifier = ShapeClassifier()
    predictions = classifier.classify_batch(sequences)
    
    correct = 0
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"{'File':<30} {'True':<15} {'Pred':<15} {'Match'}")
    print("-" * 60)
    
    for pred, label, file_id in zip(predictions, labels, file_ids):
        true_label = idx_to_label[label]
        match = pred == true_label
        correct += match
        symbol = "✓" if match else "✗"
        print(f"{file_id:<30} {true_label:<15} {pred:<15} {symbol}")
    
    accuracy = correct / len(sequences) * 100
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(sequences)} ({accuracy:.1f}%)")
    print("=" * 60)
    
    print("\nPer-class accuracy:")
    for idx in sorted(idx_to_label.keys()):
        class_name = idx_to_label[idx]
        class_mask = labels == idx
        class_preds = [p for p, m in zip(predictions, class_mask) if m]
        class_correct = sum(1 for p in class_preds if p == class_name)
        class_acc = class_correct / len(class_preds) * 100 if class_preds else 0
        print(f"  {class_name}: {class_correct}/{len(class_preds)} ({class_acc:.0f}%)")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print(f"Loading: {DATASET_FILE}")
    sequences, labels, label_map, file_ids = load_processed(DATASET_FILE)
    
    print(f"Loaded {len(sequences)} samples")
    print(f"Classes: {label_map}")
    
    evaluate(sequences, labels, label_map, file_ids)