import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import glob
from scipy.signal import butter, filtfilt

# ==========================================
# Data Augmentation (Jittering & Scaling)
# ==========================================
class DataAugmenter:
    """
    Applies augmentation techniques to time-series data.
    Techniques: Jittering (Noise) & Scaling
    """
    def __init__(self, jitter_sigma=0.03, scale_sigma=0.05):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma

    def jitter(self, x):
        # Add Gaussian noise
        noise = np.random.normal(loc=0, scale=self.jitter_sigma, size=x.shape)
        return x + noise

    def scale(self, x):
        # Scale the signal magnitude
        factor = np.random.normal(loc=1.0, scale=self.scale_sigma, size=(1, x.shape[1])) 
        return x * factor

    def augment(self, x):
        # Apply random augmentation with 50% probability
        if np.random.rand() < 0.5:
            x = self.jitter(x)
        if np.random.rand() < 0.5:
            x = self.scale(x)
        return x

# ==========================================
# Filtering Logic
# ==========================================
def butter_lowpass_filter(data, cutoff=3.0, fs=30.0, order=5):
    """
    Applies a Butterworth Low-pass filter to smooth sensor noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    try:
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        # axis=0: Time dimension
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        # Return original data if filtering fails (e.g., sequence too short)
        return data

# ==========================================
# Dataset & Loader
# ==========================================
class SensorDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmenter = DataAugmenter()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation only if enabled (usually for Training set)
        if self.augment:
            seq_np = seq.numpy()
            seq_np = self.augmenter.augment(seq_np)
            seq = torch.tensor(seq_np, dtype=torch.float32)
            
        return seq, label

def load_data_from_folder(root_dir, apply_filter=True):
    """
    Reads all CSV files recursively from a directory structure:
    root_dir/
      class_A/
        1.csv
      class_B/
        ...
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    sequences = []
    labels = []
    
    # 1. Read and Sort Class Directories
    # CRITICAL: Sorting ensures that Class ID mapping (0, 1, 2...) is consistent between Train and Test sets.
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Loading from: {root_dir}")
    print(f"Detected Classes: {label_map}")

    total_files = 0
    
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        # Find all CSV files in the class folder
        csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
        
        for file_path in csv_files:
            seq = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check for header (x,y,z) and skip it
                start_idx = 0
                if len(lines) > 0 and "x" in lines[0].lower() and "y" in lines[0].lower():
                    start_idx = 1
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line: continue
                    
                    # Parse comma-separated values
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            seq.append([x, y, z])
                        except ValueError:
                            continue
                
                if len(seq) > 0:
                    seq_np = np.array(seq, dtype=np.float32)
                    
                    # Apply Filter
                    if apply_filter:
                        seq_np = butter_lowpass_filter(seq_np)

                    sequences.append(torch.tensor(seq_np.copy(), dtype=torch.float32))
                    labels.append(label_map[class_name])
                    total_files += 1
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    print(f"Loaded {total_files} sequences from {len(classes)} classes.")
    return sequences, labels, label_map

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Sorts sequences by length and pads them.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Sort by length (descending) required for pack_padded_sequence
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    sorted_sequences = [sequences[i] for i in sorted_idx]
    sorted_labels = torch.tensor([labels[i] for i in sorted_idx], dtype=torch.long)
    
    # Pad sequences
    padded_seqs = pad_sequence(sorted_sequences, batch_first=True)
    return padded_seqs, sorted_labels, sorted_lengths