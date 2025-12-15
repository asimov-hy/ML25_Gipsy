# run_all_RF.py  (FINAL VERSION for 17 features)
import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# =====================================================
# 1) IMPORT YOUR TRAINED FEATURE EXTRACTOR (17 features)
# =====================================================
# ---> 반드시 feature_engineering.py가 같은 폴더에 있어야 함
from feature_engineering import extract_features  

def extract_number_from_filename(filename):
    """
    파일 이름 안의 첫 번째 숫자를 정수로 추출.
    숫자가 없으면 0을 반환.
    """
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else 0

# =====================================================
# 2) TXT PARSER 
# =====================================================
def extract_xyz(line):
    """
    라인에서 센서 데이터(x, y, z)를 추출합니다.
    예시: r,39534,...,392/-440/-84,...,#
    """
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


# =====================================================
# 3) PREPROCESSING (smart_trim → resample → lowpass)
# =====================================================
def smart_trim(data, threshold=0.5):
    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    active = np.where(vel > threshold)[0]

    if len(active) < 2:
        return data
    
    start = max(active[0] - 2, 0)
    end = min(active[-1] + 2, len(data))
    return data[start:end]


def preprocess_single_txt(path, target_length=100):
    
    # [수정된 파싱 로직]
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                result = extract_xyz(line) # extract_xyz를 호출하여 파싱
                if result:
                    rows.append(list(result)) 
    except Exception:
        return None 

    raw = np.array(rows)
    if len(raw) < 5:
        return None
    # 1) smart trim
    trimmed = smart_trim(raw)

    # Resample
    x_old = np.linspace(0, 1, len(trimmed))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, trimmed, axis=0, fill_value="extrapolate")
    resampled = f(x_new)

    # Low-pass filter
    FS, CUTOFF = 30, 3
    b, a = butter(5, CUTOFF / (FS * 0.5))
    filtered = filtfilt(b, a, resampled, axis=0)

    return filtered


# =====================================================
# 4) MODEL LOADER
# =====================================================
def load_model():
    with open("trained_model.pkl", "rb") as f:
        return pickle.load(f)


# =====================================================
# 5) PREDICT FOR ONE FILE
# =====================================================
def predict_single(path, model, scaler, classes):
    data = preprocess_single_txt(path)
    if data is None:
        return None

    # extract_features expects shape = (N, 100, 3)
    data_batch = np.array([data])  # shape = (1,100,3)

    feat, _ = extract_features(data_batch)  # shape = (1,17)

    feat_scaled = scaler.transform(feat)

    pred = model.predict(feat_scaled)[0]
    return pred


# =====================================================
# 6) COLLECT TXT FILES
# =====================================================
def collect_txt(root):
    # 하위 폴더까지 모든 .txt 파일 검색
    return glob.glob(os.path.join(root, "**/*.txt"), recursive=True)


# =====================================================
# 7) MAIN
# =====================================================
if __name__ == "__main__":

    print("==== Random Forest Batch Test ====\n")

    # Load model
    try:
        model_data = load_model()
        model = model_data["model"]
        scaler = model_data["scaler"]
        classes = list(model.classes_)
    except Exception as e:
        print("Model load failed:", e)
        sys.exit(1)

    # Input directory
    if len(sys.argv) >= 2:
        root = sys.argv[1]
    else:
        root = "."

    txt_files = sorted(
    collect_txt(root),
    key=lambda x: extract_number_from_filename(os.path.basename(x))
)
    txt_files = [f for f in txt_files if os.path.basename(f) != "prediction_results.txt"]

    if len(txt_files) == 0:
        print(f"No .txt files found in: {root}")
        sys.exit(0)
    
    # Predict each file
with open("prediction_results.txt", "w", encoding="utf-8") as out:
    for path in txt_files:
        base = os.path.basename(path)
        pred = predict_single(path, model, scaler, classes)

        if pred is None:
            line = f"{base}: Data Error"
        else:
            line = f"{base}: {pred}"

        print(line)           # 콘솔 출력
        out.write(line + "\n")  # 파일 저장

