import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import glob
import random
from scipy.signal import butter, filtfilt

# ==========================================
# Data Augmentation (Jittering & Scaling)
# ==========================================
class DataAugmenter:
    def __init__(self, jitter_sigma=0.03, scale_sigma=0.05):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma

    def jitter(self, x):
        noise = np.random.normal(loc=0, scale=self.jitter_sigma, size=x.shape)
        return x + noise

    def scale(self, x):
        factor = np.random.normal(loc=1.0, scale=self.scale_sigma, size=(1, x.shape[1])) 
        return x * factor

    def augment(self, x):
        if np.random.rand() < 0.5:
            x = self.jitter(x)
        if np.random.rand() < 0.5:
            x = self.scale(x)
        return x

# ==========================================
# Filtering Logic
# ==========================================
def butter_lowpass_filter(data, cutoff=3.0, fs=30.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    try:
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data, axis=0)
    except Exception as e:
        print(f"Filter Error: {e}")
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
        
        if self.augment:
            seq_np = seq.numpy()
            seq_np = self.augmenter.augment(seq_np)
            seq = torch.tensor(seq_np, dtype=torch.float32)
            
        return seq, label

def load_data(root_dir, apply_filter=True):
    sequences = []
    labels = []
    label_map = {} 
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for idx, class_name in enumerate(classes):
        label_map[class_name] = idx
        class_dir = os.path.join(root_dir, class_name)
        csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
        
        print(f"Loading Class '{class_name}' (Label: {idx}): {len(csv_files)} files found.")
        
        for file_path in csv_files:
            seq = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split("\n")
                    
                for line in lines:
                    line = line.strip()
                    if not line or line == '""': continue
                    
                    parts = line.split('/')
                    if len(parts) == 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            seq.append([x, y, z])
                        except ValueError:
                            continue
                
                if len(seq) > 0:
                    seq_np = np.array(seq, dtype=np.float32)
                    
                    if apply_filter:
                        seq_np = butter_lowpass_filter(seq_np)

                    # [수정됨] .copy()를 추가하여 메모리 stride 문제 해결
                    sequences.append(torch.tensor(seq_np.copy(), dtype=torch.float32))
                    labels.append(idx)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return sequences, labels, label_map

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    sorted_sequences = [sequences[i] for i in sorted_idx]
    sorted_labels = torch.tensor([labels[i] for i in sorted_idx], dtype=torch.long)
    padded_seqs = pad_sequence(sorted_sequences, batch_first=True)
    return padded_seqs, sorted_labels, sorted_lengths