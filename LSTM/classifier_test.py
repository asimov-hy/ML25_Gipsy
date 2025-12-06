import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import os

from preprocessing import load_data, SensorDataset, collate_fn
from model_definition import SensorLSTM
from model_training import evaluate

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "../data/processed/dataset1+2_raw.pkl"
MODEL_PATH = 'best_model.pt'
BATCH_SIZE = 16
HIDDEN_SIZE = 64
SEED = 42
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data from Pickle
    sequences, labels, label_map = load_data(DATA_PATH)
    
    if len(sequences) == 0:
        print("Error: No data loaded.")
        return
    
    # 2. Re-create Split (Must use the SAME seed as training to isolate the Test set)
    _, temp_seqs, _, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    _, test_seqs, _, test_labels = train_test_split(
        temp_seqs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    # 3. Create Test Dataset
    test_dataset = SensorDataset(test_seqs, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Train the model first!")
        return

    model = SensorLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}")

    # 5. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_targets, test_preds = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Detailed Report
    # Sort target names by index to match the confusion matrix
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names))

if __name__ == '__main__':
    main()