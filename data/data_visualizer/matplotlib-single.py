import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_trajectory(filepath):
    """Load trajectory from CSV with x,y,z columns"""
    df = pd.read_csv(filepath)
    return df['x'], df['y'], df['z']

def plot_trajectory(filepath):
    x, y, z = load_trajectory(filepath)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by time
    colors = np.linspace(0, 1, len(x))
    
    # Plot points
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=10)
    
    # Connect with line
    ax.plot(x, y, z, alpha=0.3, color='gray', linewidth=1)
    
    # Mark start and end
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='green', s=100, label='Start', marker='^')
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='red', s=100, label='End', marker='s')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory: {filepath}')
    ax.legend()
    
    plt.colorbar(scatter, label='Time progression')
    plt.tight_layout()
    plt.show()

# Usage
plot_trajectory('data/data1-filtered/circle/1.csv')