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
# FEATURE IMPORT
# ----------------------------
# feature_engineering.py의 수정된 메인 함수를 임포트 (17개 특징 반환)
from feature_engineering import extract_features 

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
        # print(f"❌ Error preprocessing {csv_path}: {e}")
        return None


# ============================================================
# Prediction using 17 FEATURES
# ============================================================

def load_model():
    with open("trained_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_motion(csv_path, model, scaler, classes):
    data = preprocess_new_data(csv_path)
    if data is None:
        return None, None

    # 17개 특징 추출 함수 호출. 단일 시퀀스를 배치 형태로 래핑
    feat, _ = extract_features(np.array([data])) 
    
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

    try:
        model_data = load_model()
    except FileNotFoundError:
        print("\n❌ Error: 'trained_model.pkl' not found. Please run 4_main.py first to train the model.")
        sys.exit(1)
        
    model = model_data["model"]
    scaler = model_data["scaler"]
    classes = list(model.classes_)

    if len(sys.argv) == 1:
        # 테스트 데이터 폴더 경로 설정 (사용자 환경에 맞게 조정)
        root = "../data/testdata_generated_ver2(100perclass)"  # ★ 기본 테스트 폴더 설정
        
        print(f"\n🔍 No input provided → Running AUTO TEST on: {root}")

        csv_files = auto_collect_files(root)
        print(f"📁 Found {len(csv_files)} files.\n")

        results = []

        for idx, f in enumerate(csv_files, 1):
            
            try:
                pred, prob = predict_motion(f, model, scaler, classes)
            except Exception as e:
                # 에러 발생 시 파일 건너뛰기
                continue 

            if prob is None:
                continue

            true_label = os.path.basename(os.path.dirname(f))
            
            prob_dict = {
                "file": os.path.basename(f),
                "true": true_label,
                "pred": pred,
                "correct": "✓" if pred == true_label else "✗",
            }

            for cls_name in classes:
                if cls_name in model.classes_:
                    prob_dict[cls_name] = prob[np.where(model.classes_ == cls_name)[0][0]]
                else:
                    prob_dict[cls_name] = 0.0
            
            results.append(prob_dict)

        df = pd.DataFrame(results)

        # -----------------------------
        # Summary Table
        # -----------------------------
        print("\n========== FINAL SUMMARY TABLE (TOP 10) ==========\n")
        if not df.empty:
            print(tabulate(df.head(10), headers="keys", tablefmt="fancy_grid", floatfmt=".3f"))
        else:
            print("No prediction results to display.")


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
        
        if not df.empty:
            df.to_csv("prediction_report.csv", index=False)
            print("\n📁 Saved → prediction_report.csv")

        sys.exit(0)