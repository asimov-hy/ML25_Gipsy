import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import os

# Import modules from fixed filenames
from preprocessing import load_data, SensorDataset, collate_fn
from model_definition import SensorLSTM
from model_training import evaluate

def main():
    parser = argparse.ArgumentParser(description="Human Motion Classifier Testing")
    parser.add_argument('--data_dir', type=str, default='csv_data_7', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split consistency')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data (Load all to reconstruct split)
    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' not found.")
        return

    sequences, labels, label_map = load_data(args.data_dir)
    
    # 2. Re-create Split (Must use same seed as training!)
    _, temp_seqs, _, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, stratify=labels, random_state=args.seed
    )
    _, test_seqs, _, test_labels = train_test_split(
        temp_seqs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )

    # 3. Create Test Dataset
    test_dataset = SensorDataset(test_seqs, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Load Model
    if not os.path.exists(args.model_path):
        print(f"Model file '{args.model_path}' not found. Train the model first!")
        return

    model = SensorLSTM(input_size=3, hidden_size=args.hidden_size, num_layers=2, num_classes=len(label_map))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {args.model_path}")

    # 5. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_targets, test_preds = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Detailed Report
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names))

if __name__ == '__main__':
    main()