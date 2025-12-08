import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

from preprocessing import load_data_from_folder, SensorDataset, collate_fn
from model_definition import SensorLSTM
from model_training import train_one_epoch, evaluate

# ==========================================
# CONFIGURATION - Updated for Generalization
# ==========================================
TRAIN_DATA_DIR = "../data/integrated_data"  
EPOCHS = 100            # Reduced (200 -> 100) to prevent memorization
BATCH_SIZE = 16
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32        # Reduced (64 -> 32) to force feature learning
SEED = 42
SAVE_PATH = 'best_model.pt'
WEIGHT_DECAY = 1e-4     # L2 Regularization
# ==========================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Training Data
    print(f"Loading Training Data from: {TRAIN_DATA_DIR}")
    sequences, labels, label_map = load_data_from_folder(TRAIN_DATA_DIR)
    
    if len(sequences) == 0:
        print("Error: No data loaded. Check the directory path.")
        return

    # 2. Split Data
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    
    print(f"Train samples: {len(train_seqs)} | Val samples: {len(val_seqs)}")

    # 3. Create Datasets (Train set uses heavy augmentation)
    train_dataset = SensorDataset(train_seqs, train_labels, augment=True)
    val_dataset = SensorDataset(val_seqs, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model
    model = SensorLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=len(label_map))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with Weight Decay (L2 Regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 5. Training Loop
    best_acc = 0.0
    print("\nStarting Training...")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dev_loader, criterion, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Model saved to {SAVE_PATH} (Acc: {best_acc:.4f})")

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()