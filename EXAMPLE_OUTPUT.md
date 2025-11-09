# Example Output After Running Experiment

This document shows what files and output you can expect after running the experiment framework.

## Command Run

```bash
python experiment.py \
  --model simple_cnn \
  --data .data/04_samples/test_cwt.pkl \
  --experiment cnn_test_exp \
  --device gpu \
  --epochs 20 \
  --batch-size 16
```

## Directory Structure After Run

```
.
├── experiment_results/
│   └── cnn_test_exp/
│       ├── experiment_results.json      # Complete experiment metadata
│       ├── training_history.png         # Accuracy/loss plots
│       ├── training_log.csv            # Epoch-by-epoch training data
│       └── tuner/
│           └── cnn_test_exp/
│               ├── oracle.json         # Hyperband state
│               ├── trial_*/            # Individual trial results
│               └── tuner*.json         # Tuner configuration
│
└── models/
    └── cnn_test_exp/
        └── best_model.keras            # Best trained model
```

## Terminal Output Example

```
2025-01-15 10:30:23 - INFO - Device: GPU (1 GPU(s) available, auto-selected)
2025-01-15 10:30:23 - INFO - Loaded model builder from: model_defs/simple_cnn.py
2025-01-15 10:30:23 - INFO - Loaded hyperparameters from: model_defs/simple_cnn_params.py
2025-01-15 10:30:24 - INFO - Loading data from: .data/04_samples/test_cwt.pkl
2025-01-15 10:30:25 - INFO - Data loaded successfully
2025-01-15 10:30:25 - INFO - ============================================================
2025-01-15 10:30:25 - INFO - Starting Hyperband hyperparameter optimization
2025-01-15 10:30:25 - INFO - ============================================================
2025-01-15 10:30:25 - INFO - Search space summary:

Search space summary
Default search space size: 13
c1_filters (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 128, 'step': 32, 'sampling': 'linear'}
c1_kernel_size (Choice)
{'default': 3, 'conditions': [], 'values': [3, 5], 'ordered': True}
dropout_1 (Float)
{'default': 0.0, 'conditions': [], 'min_value': 0.0, 'max_value': 0.5, 'step': 0.1, 'sampling': 'linear'}
c2_filters (Int)
{'default': None, 'conditions': [], 'min_value': 64, 'max_value': 256, 'step': 64, 'sampling': 'linear'}
c2_kernel_size (Choice)
{'default': 3, 'conditions': [], 'values': [3, 5, 7], 'ordered': True}
...

2025-01-15 10:30:26 - INFO - Running Hyperband search...

Trial 10 Complete [00h 02m 15s]
val_accuracy: 0.8234

Best val_accuracy So Far: 0.8945
Total elapsed time: 00h 23m 42s

2025-01-15 10:54:08 - INFO - Best hyperparameters found:
2025-01-15 10:54:08 - INFO - ------------------------------------------------------------
2025-01-15 10:54:08 - INFO -   c1_filters: 64
2025-01-15 10:54:08 - INFO -   c1_kernel_size: 3
2025-01-15 10:54:08 - INFO -   dropout_1: 0.2
2025-01-15 10:54:08 - INFO -   c2_filters: 128
2025-01-15 10:54:08 - INFO -   c2_kernel_size: 5
2025-01-15 10:54:08 - INFO -   dropout_2: 0.3
2025-01-15 10:54:08 - INFO -   use_third_conv_block: False
2025-01-15 10:54:08 - INFO -   d1_units: 256
2025-01-15 10:54:08 - INFO -   dropout_3: 0.4
2025-01-15 10:54:08 - INFO -   use_second_dense: True
2025-01-15 10:54:08 - INFO -   d2_units: 128
2025-01-15 10:54:08 - INFO -   dropout_4: 0.3
2025-01-15 10:54:08 - INFO -   optimizer: adam
2025-01-15 10:54:08 - INFO -   learning_rate: 0.001234
2025-01-15 10:54:08 - INFO - ------------------------------------------------------------
2025-01-15 10:54:09 - INFO - ============================================================
2025-01-15 10:54:09 - INFO - Training final model for 40 epochs
2025-01-15 10:54:09 - INFO - ============================================================

Epoch 1/40
100/100 ━━━━━━━━━━━━━━━━━━━━ 12s 98ms/step - accuracy: 0.4523 - loss: 1.2345 - val_accuracy: 0.5234 - val_loss: 1.0123

Epoch 2/40
100/100 ━━━━━━━━━━━━━━━━━━━━ 10s 95ms/step - accuracy: 0.5678 - loss: 0.9876 - val_accuracy: 0.6123 - val_loss: 0.8567

...

Epoch 35/40
100/100 ━━━━━━━━━━━━━━━━━━━━ 10s 94ms/step - accuracy: 0.9234 - loss: 0.2145 - val_accuracy: 0.8956 - val_loss: 0.3421

Epoch 35: ReduceLROnPlateau reducing learning rate to 0.00015432.

...

Epoch 38/40
100/100 ━━━━━━━━━━━━━━━━━━━━ 10s 95ms/step - accuracy: 0.9345 - loss: 0.1987 - val_accuracy: 0.8967 - val_loss: 0.3389

Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.

2025-01-15 11:32:45 - INFO - Training complete. Model saved to: models/cnn_test_exp/best_model.keras
2025-01-15 11:32:46 - INFO - ============================================================
2025-01-15 11:32:46 - INFO - Evaluating model on test set
2025-01-15 11:32:46 - INFO - ============================================================

25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 78ms/step - accuracy: 0.8934 - loss: 0.3456

2025-01-15 11:32:48 - INFO - Test Results:
2025-01-15 11:32:48 - INFO -   loss: 0.3421
2025-01-15 11:32:48 - INFO -   accuracy: 0.8956
2025-01-15 11:32:49 - INFO - Training plots saved to: experiment_results/cnn_test_exp/training_history.png
2025-01-15 11:32:49 - INFO - Results saved to: experiment_results/cnn_test_exp/experiment_results.json

============================================================
EXPERIMENT SUMMARY
============================================================
Experiment: cnn_test_exp
Model: simple_cnn
Timestamp: 2025-01-15T10:30:23.456789
------------------------------------------------------------
Best Hyperparameters:
  c1_filters: 64
  c1_kernel_size: 3
  dropout_1: 0.2
  c2_filters: 128
  c2_kernel_size: 5
  dropout_2: 0.3
  use_third_conv_block: False
  d1_units: 256
  dropout_3: 0.4
  use_second_dense: True
  d2_units: 128
  dropout_4: 0.3
  optimizer: adam
  learning_rate: 0.001234
------------------------------------------------------------
Test Metrics:
  loss: 0.3421
  accuracy: 0.8956
------------------------------------------------------------
Results directory: experiment_results/cnn_test_exp
Model saved to: models/cnn_test_exp/best_model.keras
============================================================
```

## Example experiment_results.json

```json
{
  "experiment_name": "cnn_test_exp",
  "model_name": "simple_cnn",
  "timestamp": "2025-01-15T10:30:23.456789",
  "config": {
    "max_epochs": 20,
    "hyperband_iterations": 1,
    "batch_size": 16,
    "validation_split": 0.2
  },
  "best_hyperparameters": {
    "c1_filters": 64,
    "c1_kernel_size": 3,
    "dropout_1": 0.2,
    "c2_filters": 128,
    "c2_kernel_size": 5,
    "dropout_2": 0.3,
    "use_third_conv_block": false,
    "d1_units": 256,
    "dropout_3": 0.4,
    "use_second_dense": true,
    "d2_units": 128,
    "dropout_4": 0.3,
    "optimizer": "adam",
    "learning_rate": 0.001234
  },
  "test_metrics": {
    "loss": 0.3421,
    "accuracy": 0.8956
  }
}
```

## Example training_log.csv

```csv
epoch,accuracy,loss,val_accuracy,val_loss,lr
0,0.4523,1.2345,0.5234,1.0123,0.001234
1,0.5678,0.9876,0.6123,0.8567,0.001234
2,0.6234,0.8234,0.6789,0.7234,0.001234
...
35,0.9234,0.2145,0.8956,0.3421,0.00015432
36,0.9267,0.2089,0.8945,0.3445,0.00015432
37,0.9345,0.1987,0.8967,0.3389,0.00015432
```

## Training History Plots

The `training_history.png` file contains two subplots:

1. **Model Accuracy**: Train vs Validation accuracy over epochs
2. **Model Loss**: Train vs Validation loss over epochs

Both plots show:
- X-axis: Epoch number
- Y-axis: Metric value
- Blue line: Training
- Orange line: Validation
- Grid for easy reading

## Loading and Using Saved Model

```python
import tensorflow as tf

# Load the best model
model = tf.keras.models.load_model('models/cnn_test_exp/best_model.keras')

# Make predictions
predictions = model.predict(new_data)

# Evaluate
results = model.evaluate(test_data)
print(f"Test accuracy: {results[1]:.4f}")
```

## Accessing Hyperband Results

```python
import json

# Load experiment results
with open('experiment_results/cnn_test_exp/experiment_results.json', 'r') as f:
    results = json.load(f)

# Access best hyperparameters
best_params = results['best_hyperparameters']
print(f"Best learning rate: {best_params['learning_rate']}")

# Access test metrics
test_acc = results['test_metrics']['accuracy']
print(f"Test accuracy: {test_acc:.4f}")
```

## Re-running or Resuming Experiments

The framework supports resuming hyperparameter search:

```python
# In experiment.py, the tuner is initialized with overwrite=False
tuner = kt.Hyperband(
    ...
    overwrite=False  # Set to True to start fresh
)
```

To start a fresh search:
- Delete the `experiment_results/<experiment_name>/tuner/` directory, or
- Use a different experiment name, or
- Modify `experiment.py` to set `overwrite=True`
