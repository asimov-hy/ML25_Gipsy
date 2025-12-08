import pickle
import numpy as np
import pandas as pd
import sys
import os
import re
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import glob
from tabulate import tabulate

# ----------------------------
# SMART FEATURE IMPORT
# ----------------------------
from feature_engineering import extract_smart_features

# ============================================================
# Utility Functions
# ============================================================

def smart_trim(data, threshold=0.5):
    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    active_idx = np.where(vel > threshold)[0]
    if len(active_idx) < 2:
        return data
    start = max(active_idx[0] - 2, 0)
    end = min(active_idx[-1] + 2, len(data))
    return data[start:end]


def preprocess_new_data(csv_path, target_length=100):
    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                nums = re.findall(r"-?\d+\.?\d*", line)
                if len(nums) >= 3:
                    rows.append([float(nums[0]), float(nums[1]), float(nums[2])])

        if len(rows) < 5:
            raise ValueError("Too few valid rows")

        data = np.array(rows)
        data = smart_trim(data)

        # Resample
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, data, axis=0)
        data_resampled = f(x_new)

        # Low-pass filter
        FS, CUTOFF = 30, 3
        b, a = butter(5, CUTOFF / (FS * 0.5))
        data_filtered = filtfilt(b, a, data_resampled, axis=0)

        return data_filtered

    except Exception as e:
        print(f"❌ Error preprocessing {csv_path}: {e}")
        return None


# ============================================================
# Prediction using SMART FEATURES
# ============================================================

def load_model():
    with open("trained_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_motion(csv_path, model, scaler, classes):
    data = preprocess_new_data(csv_path)
    if data is None:
        return None, None

    feat = extract_smart_features(data).reshape(1, -1)   # ★ 핵심: smart features 사용
    feat_scaled = scaler.transform(feat)

    pred = model.predict(feat_scaled)[0]
    prob = model.predict_proba(feat_scaled)[0]

    return pred, prob


def auto_collect_files(root):
    motions = ["circle", "horizontal", "vertical", "diagonal_left", "diagonal_right"]
    files = []
    for m in motions:
        files.extend(glob.glob(os.path.join(root, m, "*.csv")))
    return files


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n===== MOTION CLASSIFIER (Random Forest - Tabulate Report Version) =====")

    model_data = load_model()
    model = model_data["model"]
    scaler = model_data["scaler"]
    classes = list(model.classes_)

    if len(sys.argv) == 1:
        root = "../data/testdata_generated_ver2(100perclass)"  # ★ 기본 테스트 폴더 설정
        print(f"\n🔍 No input provided → Running AUTO TEST on: {root}")

        csv_files = auto_collect_files(root)
        print(f"📁 Found {len(csv_files)} files.\n")

        results = []

        for idx, f in enumerate(csv_files, 1):
            print(f"[{idx}/{len(csv_files)}] Processing {f}")
            pred, prob = predict_motion(f, model, scaler, classes)

            if prob is None:
                continue

            true_label = os.path.basename(os.path.dirname(f))

            results.append({
                "file": os.path.basename(f),
                "true": true_label,
                "pred": pred,
                "correct": "✓" if pred == true_label else "✗",
                "circle": prob[classes.index("circle")],
                "horizontal": prob[classes.index("horizontal")],
                "vertical": prob[classes.index("vertical")],
                "diagonal_left": prob[classes.index("diagonal_left")],
                "diagonal_right": prob[classes.index("diagonal_right")]
            })

        df = pd.DataFrame(results)

        # -----------------------------
        # Summary Table
        # -----------------------------
        print("\n========== FINAL SUMMARY TABLE ==========\n")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".3f"))

        # -----------------------------
        # Class-wise Accuracy
        # -----------------------------
        print("\n========== CLASS-WISE ACCURACY ==========\n")

        acc_rows = []
        for cls in classes:
            df_cls = df[df["true"] == cls]
            total = len(df_cls)
            correct = (df_cls["true"] == df_cls["pred"]).sum()
            acc = (correct / total * 100) if total > 0 else 0

            acc_rows.append([cls, total, correct, f"{acc:.1f}%"])

        print(tabulate(acc_rows, headers=["Class", "Total", "Correct", "Accuracy"],
                       tablefmt="fancy_grid"))

        # -----------------------------
        # Overall Accuracy
        # -----------------------------
        overall_correct = (df["true"] == df["pred"]).sum()
        overall_total = len(df)
        overall_acc = overall_correct / overall_total * 100

        print("\n========== OVERALL ACCURACY ==========\n")
        print(tabulate(
            [[overall_correct, overall_total, f"{overall_acc:.1f}%"]],
            headers=["Correct", "Total", "Accuracy"],
            tablefmt="fancy_grid"
        ))

        df.to_csv("prediction_report.csv", index=False)
        print("\n📁 Saved → prediction_report.csv")

        sys.exit(0)

