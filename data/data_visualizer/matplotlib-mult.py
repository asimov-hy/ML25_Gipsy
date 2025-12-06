import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob

def plot_all_trajectories_by_class(data_folder):
    """Color by class with Circle-1 style labels"""
    
    filepaths = glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)
    
    if not filepaths:
        print(f"No CSV files found in {data_folder}")
        return
    
    print(f"Found {len(filepaths)} files")
    
    # Get unique classes and assign colors
    classes = list(set(os.path.basename(os.path.dirname(f)) for f in filepaths))
    class_colors = {cls: plt.cm.tab10(i) for i, cls in enumerate(classes)}
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    plotted_classes = set()
    
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        x, y, z = df['x'], df['y'], df['z']
        
        # Get class and filename
        cls = os.path.basename(os.path.dirname(filepath))
        filename = os.path.splitext(os.path.basename(filepath))[0]
        color = class_colors[cls]
        
        # Label: Circle-1 format, but only show class name once in legend
        label = cls.capitalize() if cls not in plotted_classes else None
        plotted_classes.add(cls)
        
        ax.plot(x, y, z, color=color, linewidth=2, label=label, alpha=0.7)
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=color, s=80, marker='^')
        
        # Optional: print what's being loaded
        print(f"  Loaded: {cls.capitalize()}-{filename}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectories by Class')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Usage
plot_all_trajectories_by_class('csv_data-filtered')