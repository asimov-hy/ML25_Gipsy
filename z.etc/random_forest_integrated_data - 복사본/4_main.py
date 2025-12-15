import importlib.util
import sys
import numpy as np

# ------------------------------------------------------
# Helper: Import module from a file
# ------------------------------------------------------
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------
# Import project modules
# ------------------------------------------------------
# 경로가 ML25_Gipsy 폴더 기준이라고 가정
preprocessing = import_module_from_file("1_preprocessing.py", "preprocessing")
feature_engineering = import_module_from_file("feature_engineering.py", "feature_engineering")
model_training = import_module_from_file("3_model_training.py", "model_training")

# Bind functions (17개 특징을 반환하도록 수정한 feature_engineering.extract_features를 사용)
load_and_preprocess_data = preprocessing.load_and_preprocess_data
extract_features_17 = feature_engineering.extract_features 
train_and_explain_model = model_training.train_and_explain_model


# ------------------------------------------------------
# Training Settings
# ------------------------------------------------------
DATA_ROOT = "../data/integrated_data"
TARGET_LENGTH = 100
N_ESTIMATORS = 100
RANDOM_STATE = 42


# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------
def main():
    print("="*60)
    print("TRAINING MOTION CLASSIFIER (17 Features)")
    print("="*60)

    # Step 1: Load
    print("\n[1/3] Loading and preprocessing data...")
    X_denoised, labels = load_and_preprocess_data(DATA_ROOT, TARGET_LENGTH, augment=True)

    if X_denoised is None or X_denoised.size == 0:
        print("❌ No data loaded!")
        return

    print(f"✅ Loaded {len(X_denoised)} samples")
    print(f"   Shape: {X_denoised.shape}")
    print(f"   Classes: {np.unique(labels)}")

    # Step 2: Feature extraction
    print("\n[2/3] Extracting features (17 Dims)...")
    # 17개 특징 추출 함수 호출
    X_feat, feat_names = extract_features_17(X_denoised)
    print(f"✅ Extracted features: {X_feat.shape[1]} dims")
    print(f"   Feature names: {feat_names}")

    # Step 3: Train
    print("\n[3/3] Training model...")
    results, mean_acc = train_and_explain_model(
        X_feat, labels, feat_names,
        N_ESTIMATORS, RANDOM_STATE
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total samples: {X_feat.shape[0]}")
    print(f"Mean Cross-Validation Accuracy: {mean_acc:.4f}")
    print("\nFeature Importance:")
    print(results)
    print("\nModel saved as 'trained_model.pkl'")
    print("="*60)


if __name__ == "__main__":
    main()