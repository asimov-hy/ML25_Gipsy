import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

from preprocessing import load_data, SensorDataset, collate_fn
from model_definition import SensorLSTM
from model_training import train_one_epoch, evaluate

DATA_DIR = 'csv_data_7'
EPOCHS = 100            
BATCH_SIZE = 16         
LEARNING_RATE = 0.001   
HIDDEN_SIZE = 64        
SEED = 42               
SAVE_PATH = 'best_model.pt'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Apply Setting
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Hidden={HIDDEN_SIZE}")

    # 1. Load Data
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return

    sequences, labels, label_map = load_data(DATA_DIR)
    print(f"Total sequences: {len(sequences)}")
    print(f"Labels: {label_map}")
    
    if len(sequences) == 0:
        print("Error: No data loaded. Please check your csv files.")
        return

    # 2. Split Data
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    dev_seqs, test_seqs, dev_labels, test_labels = train_test_split(
        temp_seqs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    # 3. Create Datasets
    train_dataset = SensorDataset(train_seqs, train_labels, augment=True)
    dev_dataset = SensorDataset(dev_seqs, dev_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model
    model = SensorLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=len(label_map))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_acc = 0.0
    print("\nStarting Training...")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dev_loader, criterion, device)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Model saved to {SAVE_PATH}")

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()