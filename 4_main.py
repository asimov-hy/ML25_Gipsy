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
    # Step 1: Load + Denoise
    X_denoised, labels = load_and_preprocess_data(DATA_ROOT, TARGET_LENGTH)
    if X_denoised.size == 0:
        print("Failed to load data.")
        return

    print(f"Loaded {len(X_denoised)} samples")
    print(f"Shape: {X_denoised.shape}")
    print(f"Classes: {np.unique(labels)}")

    # Step 2: Feature extraction
    X_feat, feat_names = extract_features(X_denoised, labels)
    print(f"Extracted features shape: {X_feat.shape}")

    # Step 3: Train and explain
    results = train_and_explain_model(
        X_feat, labels, feat_names,
        N_ESTIMATORS, RANDOM_STATE
    )

    print("\nTotal samples:", X_feat.shape[0])
    print("\nFeature Importance:")
    print(results)


if __name__ == "__main__":
    main()