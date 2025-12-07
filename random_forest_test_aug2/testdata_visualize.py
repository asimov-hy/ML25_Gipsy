import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------
# CSV Loader
# -----------------------------------------
def load_csv_positions(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(r"-?\d+\.?\d*", line)
            if len(nums) >= 3:
                rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
    return np.array(rows)


# -----------------------------------------
# 3D Plotter
# -----------------------------------------
def visualize_original_and_synthetic(original_path, synthetic_dir, base_name):
    original = load_csv_positions(original_path)

    # synthetic 파일들 찾기
    synth_files = [
        f for f in os.listdir(synthetic_dir)
        if f.startswith(base_name) and f.endswith(".csv")
    ]

    if len(synth_files) == 0:
        print("⚠ No synthetic files found.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- 원본 플롯 ---
    ax.plot(
        original[:, 0], original[:, 1], original[:, 2],
        color="black", linewidth=3, label="Original"
    )

    # --- Synthetic versions 플롯 ---
    colors = plt.cm.tab20(np.linspace(0, 1, len(synth_files)))

    for c, f in zip(colors, synth_files):
        data = load_csv_positions(os.path.join(synthetic_dir, f))
        ax.plot(data[:, 0], data[:, 1], data[:, 2], color=c, alpha=0.6)

    ax.set_title(f"Original vs Synthetic Variants ({base_name})", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

