import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np

from preprocessing import load_data_from_folder, SensorDataset, collate_fn
from model_definition import SensorLSTM
from model_training import evaluate

# ==========================================
# CONFIGURATION - External Test Data Path
# ==========================================
# Path to the folder containing 2800+ augmented test samples
TEST_DATA_DIR = "../data/testdata_generated_ver2(100perclass)" 
MODEL_PATH = 'best_model.pt'
BATCH_SIZE = 16
HIDDEN_SIZE = 64
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Test Data
    # We use ALL data in this folder for testing (No splitting)
    print(f"Loading Test Data from: {TEST_DATA_DIR}")
    sequences, labels, label_map = load_data_from_folder(TEST_DATA_DIR)
    
    if len(sequences) == 0:
        print("Error: No data loaded.")
        return

    # 2. Create Dataset (No Augmentation for Testing)
    test_dataset = SensorDataset(sequences, labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Train the model first!")
        return

    # Note: Ensure the model's output size matches the number of classes in the test data
    model = SensorLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=len(label_map))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print("Error loading model: Class count mismatch?")
        print("The number of classes in the trained model might differ from the test dataset.")
        print(e)
        return

    model.to(device)
    print(f"Model loaded from {MODEL_PATH}")

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_targets, test_preds = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # 5. Detailed Report
    # Sort target names by index to align with confusion matrix
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_targets, test_preds))

if __name__ == '__main__':
    main()