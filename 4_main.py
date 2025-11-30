import importlib.util
import sys
import numpy as np

# Helper function to import modules with numeric prefixes
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import the numbered modules
preprocessing = import_module_from_file("1_preprocessing.py", "preprocessing")
feature_engineering = import_module_from_file("2_feature_engineering.py", "feature_engineering")
model_training = import_module_from_file("3_model_training.py", "model_training")

# Now use the functions
load_and_preprocess_data = preprocessing.load_and_preprocess_data
extract_features = feature_engineering.extract_features
train_and_explain_model = model_training.train_and_explain_model

DATA_ROOT = "./csv_data_7"
TARGET_LENGTH = 100
N_ESTIMATORS = 100
RANDOM_STATE = 42


def main():
    print("="*60)
    print("TRAINING MOTION CLASSIFIER")
    print("="*60)
    
    # Step 1: Load + Denoise
    print("\n[1/3] Loading and preprocessing data...")
    X_denoised, labels = load_and_preprocess_data(DATA_ROOT, TARGET_LENGTH, augment=True)
    if X_denoised.size == 0:
        print("❌ Failed to load data.")
        return

    print(f"✅ Loaded {len(X_denoised)} samples")
    print(f"   Shape: {X_denoised.shape}")
    print(f"   Classes: {np.unique(labels)}")

    # Step 2: Feature extraction
    print("\n[2/3] Extracting features...")
    X_feat, feat_names = extract_features(X_denoised, labels)
    print(f"✅ Extracted features shape: {X_feat.shape}")
    print(f"   Features: {feat_names}")

    # Step 3: Train and explain
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
    print("\n" + "="*60)
    print("Model saved as 'trained_model.pkl'")
    print("Use '5_classifier_test.py' to classify new data")
    print("="*60)


if __name__ == "__main__":
    main()