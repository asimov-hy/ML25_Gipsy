Human Motion Trajectory Classifier (LSTM)

0. Setup and Run Manual (Quickstart)

Step 1. Environment Setup

It is recommended to use a virtual environment (Conda or venv) to avoid dependency conflicts.

conda create -n gipsy_lstm python=3.8
conda activate gipsy_lstm
pip install torch numpy scikit-learn scipy matplotlib


Step 2. Ensure Project Structure

Ensure your project directory is structured as follows for the imports to work correctly:

ML25_Project/
    preprocessing.py         # Data Loading, Filtering, Augmentation logic
    model_definition.py      # LSTM Model Architecture
    model_training.py        # Training & Evaluation Functions
    main.py                  # Main Execution Script (Training)
    classifier_test.py       # Testing Script
    csv_data_7/              # Dataset Directory
        circle/
        diagonal_left/
        diagonal_right/
        horizontal/
        vertical/


Step 3. Check CSV Format

The dataset files should contain 3D coordinates (X, Y, Z).
Our custom parser supports the slash-separated format found in the raw data:

392/-440/-84


(Note: Empty lines or lines with "" are automatically handled by the loader.)

Step 4. Run Training Pipeline

To train the model and save the best weights:

python main.py


This will:

Load and filter the data (Low-pass Filter).

Apply Online Data Augmentation (Jittering/Scaling).

Train the Bidirectional LSTM model.

Save the best model weights as best_model.pt.

Step 5. Run Testing

To evaluate the trained model on the held-out test set:

python classifier_test.py --model_path best_model.pt


1. Dataset

Input: End Effector Position (X, Y, Z in millimeters).

Format: Text files with X/Y/Z format (e.g., 392/-440/-84).

Structure: Each class (motion type) must be in its own subdirectory inside csv_data_7.

Preprocessing:

Filtering: A Butterworth Low-pass Filter (Cutoff 3.0Hz) is applied to remove high-frequency sensor noise.

Padding: Since motion trajectories have variable lengths, sequences are padded to match the longest sequence in the batch using PyTorch's pad_sequence.

2. Pipeline Overview

Step 1. Load & Preprocess

Reads CSV files from the directory structure.

Applies signal filtering to smooth the trajectory.

Splits data into Train (70%), Validation (15%), and Test (15%) sets.

Step 2. Data Augmentation (Online)

To address the small dataset size challenge, we apply augmentation on-the-fly during the training phase. This ensures the model sees a slightly different variation of the trajectory every epoch.

Jittering: Adds Gaussian noise ($\mu=0, \sigma=0.03$) to simulate sensor instability.

Scaling: Multiplies trajectories by a random factor ($0.95 \sim 1.05$) to simulate different arm sizes/ranges.

Step 3. Feature Engineering (LSTM)

Model: Bidirectional LSTM (Long Short-Term Memory).

Input: Raw 3D coordinates (x, y, z).

Logic: The model captures temporal dependencies and sequential patterns in both forward and backward directions, which is crucial for distinguishing geometric shapes.

Step 4. Classification

The final hidden states of the LSTM are passed through a Fully Connected Layer to classify the motion into one of the target categories (e.g., Circle, Vertical).

3. Usage

Training Arguments

You can customize hyperparameters in main.py or via command line arguments:

python main.py --epochs 100 --batch_size 16 --lr 0.001 --hidden_size 64


epochs: Number of training iterations (Default: 50)

batch_size: Number of samples per batch (Default: 16)

data_dir: Path to the dataset (Default: 'csv_data_7')

Testing

Ensure best_model.pt exists before running the test script.

python classifier_test.py --model_path best_model.pt


4. Code Structure

The project is modularized for explainability and maintenance:

preprocessing.py

load_data(): Loads and parses CSV files.

DataAugmenter: Implements Jittering and Scaling logic.

butter_lowpass_filter: Signal processing for noise removal.

SensorDataset: Custom PyTorch Dataset class.

model_definition.py

SensorLSTM: Defines the neural network architecture (Bi-LSTM + Dropout + FC).

model_training.py

train_one_epoch(): Handles the training loop and backpropagation.

evaluate(): Computes loss and accuracy for validation/testing.

main.py

Orchestrates the entire training pipeline.

Handles argument parsing and model saving.

classifier_test.py

Loads the saved model.

Performs final evaluation on the separated Test Set.

Outputs a detailed classification report.

5. Example Output

Training:

Loading Class 'circle' (Label: 0): 10 files found.
...
Starting Training...
Epoch 1/50 | Train Loss: 1.6021 Acc: 0.2500 | Val Loss: 1.5880 Acc: 0.3300
...
Epoch 50/50 | Train Loss: 0.1023 Acc: 0.9800 | Val Loss: 0.1500 Acc: 0.9500
Training Finished. Best Validation Accuracy: 0.9500


Testing:

Final Test Accuracy: 0.9600

Classification Report:
                precision    recall  f1-score   support

        circle       1.00      1.00      1.00         5
    horizontal       0.92      1.00      0.96         4
      vertical       1.00      0.90      0.95         5
      ...


6. Why LSTM? (Methodology Explanation)

1. Robustness to Sequential Data

Unlike traditional machine learning models (e.g., Random Forest or Decision Trees) that often require fixed-length feature vectors or resampling, LSTMs naturally handle variable-length time-series data. This allows us to preserve the temporal integrity of the motion without lossy aggregation.

2. Bidirectional Context

Motion trajectories have a geometric shape where the "start" and "end" context matters significantly. We utilize a Bidirectional LSTM, which processes the data from start-to-end and end-to-start simultaneously. This improves the model's ability to distinguish similar shapes (e.g., distinguishing a 'Vertical' line drawn up-to-down vs down-to-up) compared to unidirectional models.

3. Noise Robustness

By combining Low-pass Filtering (Preprocessing) and Online Data Augmentation (Training), our model is designed to be robust against sensor noise. Even with significant jitter or speed variations, the LSTM focuses on the global pattern of the movement rather than local outliers.

7. Troubleshooting

FileNotFoundError: Directory not found: Check if the csv_data_7 folder exists in the same directory as the python scripts.

ModuleNotFoundError: Ensure all 5 python files are in the same directory and filenames are correct (preprocessing.py, model_definition.py, etc.).

Low Accuracy: Try increasing epochs in main.py or checking if the dataset classes are balanced.

Slash Separator Error: If you encounter parsing errors, ensure your data uses / or update the load_data function in preprocessing.py to match your delimiter.