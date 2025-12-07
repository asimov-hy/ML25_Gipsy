import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
INPUT_ROOT = "../data/data2-filtered"  # 원본 데이터
OUTPUT_ROOT = "../data/testdata_generated_ver2(10perclass)"

TARGET_LENGTH = 100
NUM_SYNTH = 10  # 각 클래스당 생성할 testdata 개수

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------------
# CSV Loader
# ---------------------------------------------
def load_csv_positions(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(r"-?\d+\.?\d*", line)
            if len(nums) >= 3:
                rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
    return np.array(rows)


# ---------------------------------------------
# Random Transformations
# ---------------------------------------------
def add_noise(data, noise_scale=3):
    return data + np.random.normal(0, noise_scale, data.shape)


def random_shift(data, shift_range=20):
    shift = np.random.uniform(-shift_range, shift_range, size=(1, 3))
    return data + shift


def random_scale(data, scale_range=0.15):
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    return data * scale


def random_rotation_z(data, max_angle_deg=30):
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle),  0],
        [0,             0,              1]
    ])
    return data @ R.T


def resample_to_fixed_length(data, target_length=100):
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, data, axis=0)
    return f(x_new)


# ---------------------------------------------
# Testdata Generator
# ---------------------------------------------
def generate_synthetic_versions(data):
    versions = []

    for _ in range(NUM_SYNTH):
        d = data.copy()

        # 여러 변형 적용
        d = add_noise(d, noise_scale=np.random.uniform(1, 4))
        d = random_shift(d, shift_range=np.random.uniform(5, 25))
        d = random_scale(d, scale_range=np.random.uniform(0.05, 0.25))
        d = random_rotation_z(d, max_angle_deg=np.random.uniform(10, 40))

        d = resample_to_fixed_length(d, TARGET_LENGTH)
        versions.append(d)

    return versions


# ---------------------------------------------
# Main
# ---------------------------------------------
motions = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]

def main():
    print("\n=== TEST DATA GENERATOR ===\n")

    for motion in motions:
        input_dir = os.path.join(INPUT_ROOT, motion)
        output_dir = os.path.join(OUTPUT_ROOT, motion)

        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        print(f"▶ {motion}: {len(files)} source files found")

        for file in files:
            path = os.path.join(input_dir, file)
            data = load_csv_positions(path)

            if len(data) < 10:
                continue

            syn_versions = generate_synthetic_versions(data)

            base_name = os.path.splitext(file)[0]
            for i, syn in enumerate(syn_versions, 1):
                out_path = os.path.join(output_dir, f"{base_name}_synth_{i}.csv")
                np.savetxt(out_path, syn, fmt="%.3f", delimiter=",")
    
    print("\n🎉 Test dataset created at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
