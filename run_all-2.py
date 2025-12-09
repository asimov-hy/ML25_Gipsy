"""
Combined Prediction Pipeline
Runs Random Forest and/or KNN+DTW models on input data.
Outputs prediction results to matching folder structure.
"""

# ============================================
# CONFIGURATION
# ============================================

# Enable/Disable Models
ENABLE_RF = True          # Random Forest model
ENABLE_KNN_DTW = True     # KNN + DTW model

# Input/Output Directories
INPUT_DIR = "demo"
OUTPUT_DIR_RF = "demo-output/RF"
OUTPUT_DIR_KNN = "demo-output/KNN"
OUTPUT_DIR_MAIN = "demo-output/MAIN"  # Best prediction based on confidence

# Model Paths
RF_MODEL_PATH = "random_forest_integrated_data/trained_model.pkl"
KNN_MODEL_PATH = "KNN+DTW/models/knn-data1-filtered.pkl"

# ============================================
# IMPORTS
# ============================================

import os
import sys
import re
import glob
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# ============================================
# SHARED FUNCTIONS
# ============================================

def extract_number_from_filename(filename):
    """Extract first number from filename for sorting."""
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else 0


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


def collect_txt_files(root):
    """Collect all .txt files recursively."""
    files = glob.glob(os.path.join(root, "**/*.txt"), recursive=True)
    return sorted(files, key=lambda x: extract_number_from_filename(os.path.basename(x)))


def parse_txt_file(path):
    """Parse txt file and return list of (x, y, z) tuples."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                result = extract_xyz(line)
                if result:
                    rows.append(list(result))
    except Exception:
        return None
    return rows if rows else None


# ============================================
# FEATURE EXTRACTION (17 Smart Features for RF)
# ============================================

def extract_smart_features(data):
    """
    Statistical and kinematic features for complex motion. (17 features)
    """
    data = np.array(data)

    # 1) Position stats: mean and variance (6)
    mean_x, mean_y, mean_z = np.mean(data, axis=0)
    var_x, var_y, var_z = np.var(data, axis=0)

    # 2) Correlation coefficients (3)
    corr_xy = pearsonr(data[:, 0], data[:, 1])[0]
    corr_yz = pearsonr(data[:, 1], data[:, 2])[0]
    corr_zx = pearsonr(data[:, 2], data[:, 0])[0]

    # 3) Linearity
    start_point = data[0]
    line_vec = data[-1] - start_point
    
    if np.linalg.norm(line_vec) > 1e-6:
        line_vec = line_vec / np.linalg.norm(line_vec)
        projected_dist = np.dot(data - start_point, line_vec)
        linearity = np.var(projected_dist) / (np.var(np.linalg.norm(data - start_point, axis=1)) + 1e-6)
    else:
        linearity = 0.0

    # 4) Closure
    closure = np.linalg.norm(data[-1] - data[0])

    # 5) Velocity stats
    velocity = np.linalg.norm(np.diff(data, axis=0), axis=1)
    mean_vel = np.mean(velocity)
    max_vel_smart = np.max(velocity)
    
    # 6) Mean turning angle
    angles = []
    for i in range(len(data)-2):
        a = data[i+1] - data[i]
        b = data[i+2] - data[i+1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        ang = np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))
        angles.append(ang)
        
    mean_turn_angle = np.mean(angles) if angles else 0.0

    # 7) Variance ratios (3)
    var_ratio_x = var_y / (var_x + 1e-6)
    var_ratio_y = var_z / (var_y + 1e-6)
    var_ratio_z = var_x / (var_z + 1e-6)

    return np.array([
        mean_x, mean_y, mean_z, var_x, var_y, var_z,
        corr_xy, corr_yz, corr_zx,
        linearity, closure,
        mean_vel, max_vel_smart, mean_turn_angle,
        var_ratio_x, var_ratio_y, var_ratio_z
    ])


def extract_features(X):
    """
    X : (N, 100, 3) shaped preprocessed sequences
    return: (N, 17) feature array
    """
    feature_list = []
    for seq in X:
        f_smart = extract_smart_features(seq)
        feature_list.append(f_smart)
    return np.array(feature_list)


# ============================================
# RANDOM FOREST MODEL
# ============================================

class RandomForestPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.classes = None
        
    def load(self):
        """Load RF model."""
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.classes = list(self.model.classes_)
            return True
        except FileNotFoundError:
            print(f"[RF] Model file not found: {self.model_path}")
            return False
        except Exception as e:
            print(f"[RF] Model load failed: {e}")
            return False
    
    def smart_trim(self, data, threshold=0.5):
        """Trim inactive portions of the signal."""
        vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
        active = np.where(vel > threshold)[0]
        
        if len(active) < 2:
            return data
        
        start = max(active[0] - 2, 0)
        end = min(active[-1] + 2, len(data))
        return data[start:end]
    
    def preprocess(self, raw_data, target_length=100):
        """Preprocess: smart_trim -> resample -> lowpass filter."""
        raw = np.array(raw_data)
        if len(raw) < 5:
            return None
        
        trimmed = self.smart_trim(raw)
        
        x_old = np.linspace(0, 1, len(trimmed))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, trimmed, axis=0, fill_value="extrapolate")
        resampled = f(x_new)
        
        FS, CUTOFF = 30, 3
        b, a = butter(5, CUTOFF / (FS * 0.5))
        filtered = filtfilt(b, a, resampled, axis=0)
        
        return filtered
    
    def predict(self, path):
        """Predict class for a single file. Returns (prediction, confidence)."""
        rows = parse_txt_file(path)
        if rows is None:
            return None, None
        
        data = self.preprocess(rows)
        if data is None:
            return None, None
        
        data_batch = np.array([data])
        feat = extract_features(data_batch)
        feat_scaled = self.scaler.transform(feat)
        
        pred = self.model.predict(feat_scaled)[0]
        
        # Get probability/confidence
        conf = None
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(feat_scaled)[0]
            pred_idx = list(self.model.classes_).index(pred)
            conf = probs[pred_idx]
        
        return pred, conf


# ============================================
# KNN + DTW MODEL
# ============================================

class KNNDTWPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.label_map = None
        self.max_seq_len = None
        
    def load(self):
        """Load KNN model."""
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.label_map = data["label_map"]
            self.max_seq_len = data.get("max_seq_len", data["model"]._X_fit.shape[1])
            return True
        except FileNotFoundError:
            print(f"[KNN] Model file not found: {self.model_path}")
            return False
        except Exception as e:
            print(f"[KNN] Model load failed: {e}")
            return False
    
    def normalize_sequence(self, seq):
        """Per-sample normalization: zero mean, unit std."""
        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
        std[std == 0] = 1
        return (seq - mean) / std
    
    def prepare_sample(self, seq):
        """Prepare single sequence (pad or truncate to max_len)."""
        n_features = seq.shape[1]
        X = np.full((1, self.max_seq_len, n_features), np.nan)
        
        seq_len = min(len(seq), self.max_seq_len)
        X[0, :seq_len, :] = seq[:seq_len]
        
        return X
    
    def predict(self, path):
        """Predict class for a single file. Returns (prediction, confidence)."""
        rows = parse_txt_file(path)
        if rows is None or len(rows) == 0:
            return None, None
        
        seq = np.array(rows, dtype=np.float32)
        seq_norm = self.normalize_sequence(seq)
        X = self.prepare_sample(seq_norm)
        
        pred_idx = self.model.predict(X)[0]
        
        idx_to_label = {v: k for k, v in self.label_map.items()}
        predicted_class = idx_to_label[pred_idx]
        
        # Calculate confidence
        distances, indices = self.model.kneighbors(X)
        neighbor_labels = self.model._y[indices[0]]
        votes = np.bincount(neighbor_labels, minlength=len(self.label_map))
        confidence = votes[pred_idx] / len(neighbor_labels)
        
        return predicted_class, confidence


# ============================================
# MAIN PIPELINE
# ============================================

def run_pipeline(input_dir, output_dir_rf, output_dir_knn, output_dir_main):
    """Run the combined prediction pipeline."""
    
    print("=" * 60)
    print("COMBINED PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Input:        {input_dir}")
    if ENABLE_RF:
        print(f"Output (RF):  {output_dir_rf}")
    if ENABLE_KNN_DTW:
        print(f"Output (KNN): {output_dir_knn}")
    print(f"Output (MAIN): {output_dir_main}")
    print(f"RF Enabled:   {ENABLE_RF}")
    print(f"KNN Enabled:  {ENABLE_KNN_DTW}")
    print("-" * 60)
    
    if not ENABLE_RF and not ENABLE_KNN_DTW:
        print("ERROR: At least one model must be enabled!")
        sys.exit(1)
    
    # Initialize predictors
    rf_predictor = None
    knn_predictor = None
    
    if ENABLE_RF:
        print("\n[RF] Loading Random Forest model...")
        rf_predictor = RandomForestPredictor(RF_MODEL_PATH)
        if not rf_predictor.load():
            print("[RF] WARNING: RF model disabled due to load failure")
            rf_predictor = None
        else:
            print(f"[RF] Classes: {rf_predictor.classes}")
    
    if ENABLE_KNN_DTW:
        print("\n[KNN] Loading KNN+DTW model...")
        knn_predictor = KNNDTWPredictor(KNN_MODEL_PATH)
        if not knn_predictor.load():
            print("[KNN] WARNING: KNN model disabled due to load failure")
            knn_predictor = None
        else:
            print(f"[KNN] Classes: {list(knn_predictor.label_map.keys())}")
            print(f"[KNN] Max sequence length: {knn_predictor.max_seq_len}")
    
    if rf_predictor is None and knn_predictor is None:
        print("\nERROR: No models loaded successfully!")
        sys.exit(1)
    
    # Collect files
    txt_files = collect_txt_files(input_dir)
    
    if not txt_files:
        print(f"\nNo .txt files found in: {input_dir}")
        sys.exit(0)
    
    print(f"\nProcessing {len(txt_files)} files...")
    print("-" * 60)
    
    # Process files
    results = []
    
    for input_path in txt_files:
        # Get relative path and create output path
        relative_path = os.path.relpath(input_path, input_dir)
        filename = os.path.basename(input_path)
        
        # Predict with confidence
        rf_pred, rf_conf = None, None
        knn_pred, knn_conf = None, None
        
        if rf_predictor:
            rf_pred, rf_conf = rf_predictor.predict(input_path)
        
        if knn_predictor:
            knn_pred, knn_conf = knn_predictor.predict(input_path)
        
        # Save RF result to RF output directory
        if rf_predictor and rf_pred is not None:
            rf_output_path = os.path.join(output_dir_rf, relative_path)
            os.makedirs(os.path.dirname(rf_output_path), exist_ok=True)
            with open(rf_output_path, "w", encoding="utf-8") as f:
                f.write(rf_pred)
        
        # Save KNN result to KNN output directory
        if knn_predictor and knn_pred is not None:
            knn_output_path = os.path.join(output_dir_knn, relative_path)
            os.makedirs(os.path.dirname(knn_output_path), exist_ok=True)
            with open(knn_output_path, "w", encoding="utf-8") as f:
                f.write(knn_pred)
        
        # Determine best prediction based on highest confidence sum
        # Group predictions by class and sum their confidences
        class_confidence = {}
        
        if rf_pred is not None and rf_conf is not None:
            class_confidence[rf_pred] = class_confidence.get(rf_pred, 0) + rf_conf
        
        if knn_pred is not None and knn_conf is not None:
            class_confidence[knn_pred] = class_confidence.get(knn_pred, 0) + knn_conf
        
        # Select class with highest combined confidence
        if class_confidence:
            best_pred = max(class_confidence, key=class_confidence.get)
            best_conf = class_confidence[best_pred]
        else:
            best_pred = "ERROR"
            best_conf = 0
        
        # Save best result to main output directory
        if best_pred != "ERROR":
            main_output_path = os.path.join(output_dir_main, relative_path)
            os.makedirs(os.path.dirname(main_output_path), exist_ok=True)
            with open(main_output_path, "w", encoding="utf-8") as f:
                f.write(best_pred)
        
        # Display
        display_parts = [f"{filename}:"]
        if rf_pred:
            rf_conf_str = f"{rf_conf:.0%}" if rf_conf else "N/A"
            display_parts.append(f"RF={rf_pred}({rf_conf_str})")
        if knn_pred:
            knn_conf_str = f"{knn_conf:.0%}" if knn_conf else "N/A"
            display_parts.append(f"KNN={knn_pred}({knn_conf_str})")
        
        display_parts.append(f"-> BEST={best_pred}({best_conf:.0%})")
        
        if rf_pred and knn_pred and rf_pred != knn_pred:
            display_parts.append("[DISAGREE]")
        
        print(" ".join(display_parts))
        
        results.append({
            'file': filename,
            'rf_pred': rf_pred,
            'rf_conf': rf_conf,
            'knn_pred': knn_pred,
            'knn_conf': knn_conf,
            'best_pred': best_pred,
            'best_conf': best_conf
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    success = sum(1 for r in results if r['best_pred'] != "ERROR")
    print(f"Processed: {success}/{total} files")
    
    if rf_predictor:
        rf_counts = {}
        for r in results:
            if r['rf_pred']:
                rf_counts[r['rf_pred']] = rf_counts.get(r['rf_pred'], 0) + 1
        print(f"RF Distribution:   {rf_counts}")
    
    if knn_predictor:
        knn_counts = {}
        for r in results:
            if r['knn_pred']:
                knn_counts[r['knn_pred']] = knn_counts.get(r['knn_pred'], 0) + 1
        print(f"KNN Distribution:  {knn_counts}")
    
    # Main/Best distribution
    best_counts = {}
    for r in results:
        if r['best_pred'] and r['best_pred'] != "ERROR":
            best_counts[r['best_pred']] = best_counts.get(r['best_pred'], 0) + 1
    print(f"MAIN Distribution: {best_counts}")
    
    if rf_predictor and knn_predictor:
        agreements = sum(1 for r in results 
                        if r['rf_pred'] and r['knn_pred'] and r['rf_pred'] == r['knn_pred'])
        both_valid = sum(1 for r in results if r['rf_pred'] and r['knn_pred'])
        if both_valid > 0:
            print(f"Model Agreement:   {agreements}/{both_valid} ({agreements/both_valid:.1%})")
    
    print(f"\nResults saved to:")
    if rf_predictor:
        print(f"  RF:   {output_dir_rf}/")
    if knn_predictor:
        print(f"  KNN:  {output_dir_knn}/")
    print(f"  MAIN: {output_dir_main}/")
    
    return results


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    input_dir = INPUT_DIR
    output_dir_rf = OUTPUT_DIR_RF
    output_dir_knn = OUTPUT_DIR_KNN
    output_dir_main = OUTPUT_DIR_MAIN
    
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir_rf = sys.argv[2]
    if len(sys.argv) >= 4:
        output_dir_knn = sys.argv[3]
    if len(sys.argv) >= 5:
        output_dir_main = sys.argv[4]
    
    run_pipeline(input_dir, output_dir_rf, output_dir_knn, output_dir_main)