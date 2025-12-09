import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re

from preprocessing import butter_lowpass_filter
from model_definition import SensorLSTM

# ==========================================
# CONFIGURATION
# ==========================================
DEMO_DATA_DIR = "../data/demo_data"  # Folder containing 1.txt, 2.txt ...
MODEL_PATH = 'best_model.pt'
HIDDEN_SIZE = 32                     # Must match training config
# Class mapping (Based on alphabetical order during training)
IDX_TO_CLASS = {
    0: 'circle',
    1: 'diagonal_left',
    2: 'diagonal_right',
    3: 'horizontal',
    4: 'vertical'
}
# ==========================================

def parse_raw_txt(filepath):
    """
    Parses the raw sensor text file provided for the demo.
    Format example: r,160209,3307276,-2/36/76/43,0/0/0/-1,0/0/0/0,70/497/-166,...
    Target: Index 6 (70/497/-166) -> x, y, z
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Only process lines starting with 'r,' (valid sensor data)
                if line.startswith('r,'):
                    parts = line.split(',')
                    # Index 6 contains coordinates like "70/497/-166"
                    if len(parts) > 6:
                        xyz_part = parts[6]
                        coords = xyz_part.split('/')
                        if len(coords) == 3:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                z = float(coords[2])
                                data.append([x, y, z])
                            except ValueError:
                                continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if len(data) == 0:
        return None
    
    return np.array(data, dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Reading files from: {DEMO_DATA_DIR}")
    print("-" * 40)

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found!")
        return

    # Initialize model
    num_classes = len(IDX_TO_CLASS)
    model = SensorLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # 2. Get list of .txt files and sort them numerically (1.txt, 2.txt, ... 10.txt)
    files = glob.glob(os.path.join(DEMO_DATA_DIR, "*.txt"))
    
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    files.sort(key=alphanum_key)

    if not files:
        print("No .txt files found in the demo directory.")
        return

    # 3. Predict loop
    print(f"{'File Name':<15} | {'Prediction':<15}")
    print("-" * 40)

    # Initialize counters for summary
    class_counts = {name: 0 for name in IDX_TO_CLASS.values()}
    total_files = 0

    with torch.no_grad():
        for file_path in files:
            filename = os.path.basename(file_path)
            
            # Parse raw data
            raw_data = parse_raw_txt(file_path)
            
            if raw_data is None or len(raw_data) == 0:
                print(f"{filename:<15} | [Error] Invalid Data")
                continue

            # Preprocessing (Filter)
            filtered_data = butter_lowpass_filter(raw_data)
            
            # Convert to Tensor (Batch size 1)
            # [Fix] Added .copy() to solve negative stride issue
            tensor_seq = torch.tensor(filtered_data.copy(), dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(filtered_data)]).to(device)

            # Inference
            outputs = model(tensor_seq, lengths)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = IDX_TO_CLASS[predicted_idx.item()]

            # Update stats
            class_counts[predicted_label] += 1
            total_files += 1

            # Output Format
            file_num = filename.replace('.txt', '')
            print(f"{file_num:<15} : {predicted_label}")

    # 4. Print Summary
    print("-" * 40)
    print(" [ Summary of Results ]")
    print("-" * 40)
    print(f" Total Files Processed: {total_files}")
    print("-" * 40)
    
    # Print counts for each class
    for class_name, count in class_counts.items():
        print(f" {class_name:<15}: {count}")
    
    print("-" * 40)
    print("Demo Finished.")

if __name__ == '__main__':
    main()