import pandas as pd
import numpy as np
from pathlib import Path
import glob

###############################################
# 0) ROBUST LOADER
###############################################
def load_data_robust(filepath):
    try:
        # 1) slash-separated 케이스
        try:
            df = pd.read_csv(filepath, header=None, sep='/', engine='python')
            data = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(data) > 10:
                return data
        except:
            pass

        # 2) comma-separated
        try:
            df = pd.read_csv(filepath, header=None)
            data = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(data) > 10:
                return data
        except:
            pass

        # 3) whitespace-separated
        try:
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            data = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(data) > 10:
                return data
        except:
            pass

        print(f"⚠ WARNING: {filepath} failed to load.")
        return None
    except Exception as e:
        print(f"❌ ERROR loading {filepath}: {e}")
        return None


###############################################
# 1) STATIC 구간 자동 감지
###############################################
def detect_static_region(data, threshold=1.0, min_len=10):
    diffs = np.linalg.norm(np.diff(data, axis=0), axis=1)
    static_len = 0
    for d in diffs:
        if d < threshold:
            static_len += 1
        else:
            break
    return static_len if static_len >= min_len else 0


###############################################
# 2) STATIC-aware AUGMENTATION
###############################################
def augment_trajectory_static_aware(data, n_variations=5):
    augmented = []

    static_len = detect_static_region(data)
    static_part = data[:static_len].copy()
    dynamic_part = data[static_len:].copy()

    for _ in range(n_variations):
        aug_static = static_part.copy()
        aug_dynamic = dynamic_part.copy()

        # scaling
        if len(aug_dynamic) > 0:
            center = np.mean(aug_dynamic, axis=0)
            scale = np.random.uniform(0.95, 1.05)
            aug_dynamic = (aug_dynamic - center) * scale + center

        # noise
        if len(aug_static) > 0:
            aug_static += np.random.normal(0, 0.8, aug_static.shape)

        if len(aug_dynamic) > 0:
            aug_dynamic += np.random.normal(0, 2.0, aug_dynamic.shape)

        # merge
        aug_full = np.vstack([aug_static, aug_dynamic])

        # shift
        aug_full += np.random.uniform(-10, 10, size=3)

        augmented.append(aug_full)

    return augmented


###############################################
# 3) CLEAN VERSION
###############################################
def generate_clean_version(data, n_variations=5):
    clean_list = []

    static_len = detect_static_region(data)
    static_part = data[:static_len].copy()
    dynamic_part = data[static_len:].copy()

    for _ in range(n_variations):
        c_static = static_part.copy()
        c_dynamic = dynamic_part.copy()

        c_dynamic += np.random.normal(0, 0.3, c_dynamic.shape)

        scale = np.random.uniform(0.98, 1.02)
        center = np.mean(c_dynamic, axis=0)
        c_dynamic = (c_dynamic - center) * scale + center

        clean_list.append(np.vstack([c_static, c_dynamic]))

    return clean_list


###############################################
# 4) 실행부 (경로 저장 수정됨!)
###############################################
input_files = glob.glob("*.csv") + glob.glob("*.txt")

aug_dir = Path("augmented_real_data")
clean_dir = Path("clean_synthetic_data")

aug_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)

print(f"\n=== START AUGMENTATION ===")
print(f"Found {len(input_files)} files.\n")

processed = 0

for f in input_files:
    data = load_data_robust(f)
    if data is None:
        continue

    base = Path(f).stem

    # AUGMENTED
    aug_list = augment_trajectory_static_aware(data, n_variations=5)
    for i, aug in enumerate(aug_list):
        path = aug_dir / f"{base}_aug_{i+1}.csv"
        pd.DataFrame(aug).to_csv(path, header=False, index=False, sep='/')

    # CLEAN
    clean_list = generate_clean_version(data, n_variations=5)
    for i, c in enumerate(clean_list):
        path = clean_dir / f"{base}_clean_{i+1}.csv"
        pd.DataFrame(c).to_csv(path, header=False, index=False, sep='/')

    processed += 1
    print(f"Processed: {f}")

print(f"\n✔ DONE! {processed} files processed.")
print("  ▶ augmented_real_data/")
print("  ▶ clean_synthetic_data/")
