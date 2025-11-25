# Explainable Motion Classification

## 0. Setup and Run Manual (Quickstart)

### Step 1. Activate Conda Environment

```
conda activate gipsyml
```

### Step 2. Ensure Project Structure

```
ML25_Gipsy/
    1_preprocessing.py
    2_feature_engineering.py
    3_model_training.py
    4_main.py
    csv_data_7/
        circle/
        diagonal_left/
        diagonal_right/
        horizontal/
        vertical/
```

### Step 3. Check CSV Format

Must be comma-separated:

```
392,-440,-84
```

If your data has slashes (`392/-440/-84`), convert them or update loader.

### Step 4. Update Data Path

Inside `4_main.py`:

```
DATA_ROOT = "./csv_data_7"
```

### Step 5. Run Pipeline

```
python 4_main.py
```

This will:

* load CSV files
* resample
* denoise
* extract features
* train Random Forest
* print top features (explainability)

### Step 6. Troubleshooting

* **No valid CSV loaded:** check separators and folder names.
* **Failed to load data:** confirm csv_data_7 exists and contains subfolders.
* **0 samples processed:** CSV files must contain 3 numeric columns.

---

### ML25 Gipsy Project

This repository implements a lightweight and explainable machine learning pipeline for classifying human arm motion using 3D End Effector Position time series data. The system preprocesses noisy data, extracts interpretable features, trains a Random Forest classifier, and outputs feature importance as the explanation.

---

## 1. Dataset

**Important format note:**
Your raw motion files must contain **three numeric columns (X, Y, Z)** separated by commas.
Example:

```
392,-440,-84
```

If your files use slashes instead of commas:

```
392/-440/-84
```

you must either:

* convert `/` to `,`, or

* modify the loader to accept slashes as separators.

* Input: End Effector Position (X, Y, Z in millimeters)

* Origin at shoulder joint (0, 0, 0)

* First batch: 34 good samples

* Second batch: Good plus noisy samples

* Data is limited, so augmentation is recommended

Expected structure:

```
csv_data_7/
    motion_1/*.csv
    motion_2/*.csv
```

---

## 2. Pipeline Overview

### Step 1. Load and Preprocess

* Reads CSV time series
* Resamples to fixed length (TARGET_LENGTH)
* Applies Butterworth low pass filter
* Output: (n_samples, time_length, 3)

### Step 2. Feature Extraction

Current features (placeholders):

* Mean of first dimension
* Maximum velocity magnitude
  Future features:
* Statistical metrics
* Dynamic metrics
* DTW distances

### Step 3. Model Training and Explainability

* StandardScaler normalization
* Stratified 5 fold cross validation
* Random Forest classifier
* Aggregated feature importance

### Step 4. Summary Output

* Total samples
* Top features ranked by importance

---

## 3. Usage

Install requirements:

```
numpy
pandas
scipy
scikit-learn
tslearn
```

Set dataset path in code:

```python
DATA_ROOT = 'path/to/csv_data_7'
```

Run:

```bash
python main.py
```

---

## 4. Code Structure

* load_and_preprocess_data()
  Loads, resamples, and denoises time series
* extract_features()
  Converts each time series into a feature vector
* train_and_explain_model()
  Trains classifier and computes feature importance
* main_program()
  Full pipeline runner

---

## 5. Example Output

```
--- Model Summary and Explanation ---
Total samples processed: 34

Top Features:
| Feature        | Importance |
| Mean_DIM_0     | 0.45       |
| Max_Velocity   | 0.29       |
```

---

## 6. ML25 Gipsy Requirements

### Timeline

* Weeks 11 to 14: Implementation
* Dec 16: Final code plus report plus presentation
* Dec 18: Peer review report

### Deliverables

1. Source code
2. Final report (max 10 pages)
3. Public repository with contributions
4. Lightning talk slides

### Team

* 3 to 4 members
* Must document individual contributions

---

## 6. Running the Code (Manual)

### Step 1. Activate the Conda Environment

```
conda activate gipsyml
```

### Step 2. Confirm Project Structure

Your repository should contain:

```
ML25_Gipsy/
    1_preprocessing.py
    2_feature_engineering.py
    3_model_training.py
    4_main.py
    csv_data_7/
        circle/
        diagonal_left/
        diagonal_right/
        horizontal/
        vertical/
```

Make sure your data uses comma-separated values:

```
392,-440,-84
```

If using `/` separators, update the loader or convert the files.

### Step 3. Update DATA_ROOT Path

Inside **4_main.py**:

```
DATA_ROOT = "./csv_data_7"
```

This must point to your motion folders.

### Step 4. Run the Full Pipeline

```
python 4_main.py
```

If everything is correct, the output will show:

* number of samples loaded
* extracted features
* feature importance table (explainability)

### Step 5. Troubleshooting

* If you see **"No valid CSV loaded"**, check separators and folder names.
* If you see **"Failed to load data"**, confirm dataset folder structure.
* If the program runs but outputs 0 samples, confirm that each file has 3 numeric columns.

---

## 7. Future Work

* Full DTW features
* Data augmentation routines
* Additional dynamic features
* SHAP or LIME explainability

---
