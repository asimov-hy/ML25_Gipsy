# ============================================
# CONFIGURATION - Change these paths as needed
# ============================================
INPUT_DIR = "data/data1-RAW"
OUTPUT_DIR = "data/data1-filtered"
# ============================================

import os

def extract_xyz(line):
    """Extract x,y,z from line like: r,39534,...,392/-440/-84,...,#"""
    line = line.strip()
    
    if not line or not line.startswith("r,"):
        return None
    
    fields = line.split(",")
    
    if len(fields) < 7:
        return None
    
    xyz_field = fields[6]  # "392/-440/-84"
    parts = xyz_field.split("/")
    
    if len(parts) != 3:
        return None
    
    try:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return x, y, z
    except ValueError:
        return None


def process_file(input_path, output_path):
    """Process single txt file → csv"""
    data = []
    skipped = 0
    
    with open(input_path, "r") as f:
        for line in f:
            result = extract_xyz(line)
            if result:
                data.append(result)
            else:
                skipped += 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write CSV
    with open(output_path, "w") as f:
        f.write("x,y,z\n")
        for x, y, z in data:
            f.write(f"{x},{y},{z}\n")
    
    return len(data), skipped


def main():
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 40)
    
    total_files = 0
    total_samples = 0
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            
            # Build paths
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_path, INPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, relative_path.replace(".txt", ".csv"))
            
            # Process
            samples, skipped = process_file(input_path, output_path)
            total_files += 1
            total_samples += samples
            
            print(f"{relative_path} → {samples} samples ({skipped} skipped)")
    
    print("-" * 40)
    print(f"Done: {total_files} files, {total_samples} total samples")


if __name__ == "__main__":
    main()