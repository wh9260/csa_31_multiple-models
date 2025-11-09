# Modular ML Experiment Framework

A flexible, modular framework for machine learning experimentation with automatic hyperparameter optimization using Hyperband.

## Features

- **Modular Architecture**: Separate model definitions, hyperparameters, and experiment logic
- **Hyperband Optimization**: Efficient hyperparameter search using Keras Tuner
- **Device Management**: Automatic CPU/GPU selection and memory management
- **Result Tracking**: Automatic saving of models, metrics, plots, and logs
- **Command-line Interface**: Easy-to-use CLI with argparse
- **Extensible**: Simple to add new models and customize hyperparameters

## Directory Structure

```
.
├── experiment.py                    # Main experiment runner
├── model_defs/                      # Model definitions and configs
│   ├── simple_cnn.py               # Example CNN model
│   ├── simple_cnn_params.py        # CNN hyperparameters
│   ├── simple_mlp.py               # Example MLP model
│   └── simple_mlp_params.py        # MLP hyperparameters
├── experiment_results/              # Experiment outputs (auto-created)
│   └── <experiment_name>/
│       ├── experiment_results.json # Metrics and config
│       ├── training_history.png    # Training plots
│       ├── training_log.csv        # Epoch-by-epoch logs
│       └── tuner/                  # Hyperband search results
└── models/                          # Saved models (auto-created)
    └── <experiment_name>/
        └── best_model.keras        # Best model checkpoint
```

## Quick Start

### 1. Basic Usage

Run an experiment with default settings:

```bash
python experiment.py \
  --model simple_cnn \
  --data .data/04_samples/test_cwt.pkl \
  --experiment my_first_experiment
```

### 2. Custom Configuration

Run with custom hyperparameter search settings:

```bash
python experiment.py \
  --model simple_cnn \
  --data path/to/your/data.pkl \
  --experiment cnn_exp_v1 \
  --device gpu \
  --epochs 30 \
  --iterations 2 \
  --batch-size 16
```

### 3. Using MLP Model

```bash
python experiment.py \
  --model simple_mlp \
  --data path/to/your/data.pkl \
  --experiment mlp_exp_v1 \
  --device auto
```

## Command-line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | str | Yes | - | Model name (without .py) |
| `--data` | str | Yes | - | Path to dataset file |
| `--experiment` | str | Yes | - | Experiment name |
| `--device` | str | No | auto | Device: cpu, gpu, or auto |
| `--epochs` | int | No | 50 | Max epochs for search |
| `--iterations` | int | No | 1 | Hyperband iterations |
| `--batch-size` | int | No | 32 | Training batch size |
| `--validation-split` | float | No | 0.2 | Validation split ratio |

## Creating Custom Models

### Step 1: Create Model Definition

Create a new file in `model_defs/` (e.g., `my_model.py`):

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt

def build_model(hp, input_shape, num_classes, hyperparameters):
    """
    Build your custom model.

    Args:
        hp: Keras Tuner HyperParameters object
        input_shape: Input data shape
        num_classes: Number of output classes
        hyperparameters: Dict from your params file

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()

    # Define your architecture with tunable hyperparameters
    hp_units = hp.Int('units',
                      min_value=hyperparameters['units_min'],
                      max_value=hyperparameters['units_max'])

    model.add(layers.Dense(hp_units, activation='relu',
                          input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

### Step 2: Create Hyperparameter Config

Create `model_defs/my_model_params.py`:

```python
HYPERPARAMETERS = {
    'units_min': 64,
    'units_max': 512,
    'units_step': 64,
}
```

### Step 3: Run Experiment

```bash
python experiment.py --model my_model --data data.pkl --experiment test
```

## Example Output

### Terminal Summary

After running an experiment, you'll see:

```
============================================================
EXPERIMENT SUMMARY
============================================================
Experiment: my_first_experiment
Model: simple_cnn
Timestamp: 2025-01-15T10:30:45.123456
------------------------------------------------------------
Best Hyperparameters:
  c1_filters: 64
  c1_kernel_size: 3
  c2_filters: 128
  c2_kernel_size: 5
  dropout_1: 0.2
  dropout_2: 0.3
  optimizer: adam
  learning_rate: 0.001
------------------------------------------------------------
Test Metrics:
  loss: 0.3421
  accuracy: 0.8956
------------------------------------------------------------
Results directory: experiment_results/my_first_experiment
Model saved to: models/my_first_experiment/best_model.keras
============================================================
```

### Generated Files

After running, you'll find:

1. **experiment_results/my_first_experiment/**
   - `experiment_results.json` - Complete metrics and config
   - `training_history.png` - Accuracy and loss plots
   - `training_log.csv` - Epoch-by-epoch training data
   - `tuner/` - Hyperband search state

2. **models/my_first_experiment/**
   - `best_model.keras` - Best model checkpoint

## Workflow

The framework executes the following pipeline:

1. **Initialization**: Load model builder and hyperparameters
2. **Device Setup**: Configure GPU/CPU with memory management
3. **Data Loading**: Load train/val/test datasets
4. **Hyperband Search**: Find optimal hyperparameters
5. **Final Training**: Train best model with extended epochs
6. **Evaluation**: Test on held-out test set
7. **Save Results**: Save model, plots, and metrics

## Advanced Features

### GPU Memory Management

The framework automatically configures GPU memory growth to prevent OOM errors:

```python
# Automatically enabled when using GPU
tf.config.experimental.set_memory_growth(gpu, True)
```

### Custom Data Loaders

To use custom data formats, modify the `load_data()` method in `ExperimentFramework`:

```python
def load_data(self):
    # Add your custom data loading logic here
    # Return: train_data, val_data, test_data
    pass
```

### TensorBoard Logging

TensorBoard logs are automatically saved:

```bash
tensorboard --logdir experiment_results/<experiment_name>/tuner
```

### Hyperband Configuration

Adjust the Hyperband algorithm parameters:

- `--epochs`: Maximum epochs per trial (higher = longer search)
- `--iterations`: Number of Hyperband iterations (higher = more thorough)
- `factor`: Reduction factor (default: 3, hardcoded in framework)

## Tips for Better Results

1. **Start Small**: Begin with `--epochs 10 --iterations 1` for testing
2. **GPU Memory**: Use `--batch-size 8` or `--batch-size 16` to avoid OOM
3. **Hyperparameter Ranges**: Keep search space reasonable in params files
4. **Quick Tests**: Create a `*_params_quick.py` file with narrow ranges
5. **Monitor Progress**: Watch TensorBoard for real-time training metrics

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution**: Reduce batch size or limit model complexity
```bash
python experiment.py --model simple_cnn --data data.pkl \
  --experiment test --batch-size 8
```

### Slow Hyperparameter Search

**Solution**: Reduce search space in params file or decrease epochs
- Edit `model_defs/<model>_params.py`
- Use larger `step` values
- Reduce max values or limit optimizer choices

### Data Loading Errors

**Solution**: Customize `load_data()` method in `experiment.py` for your data format

## Extending the Framework

### Adding New Callbacks

Edit `train_final_model()` or `run_hyperband_search()` in `experiment.py`:

```python
callbacks = [
    # Add custom callbacks here
    tf.keras.callbacks.LearningRateScheduler(schedule_fn),
    # ... existing callbacks
]
```

### Custom Metrics

Modify the model compilation in your model definition:

```python
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']  # Add custom metrics
)
```

## Requirements

- Python >= 3.12
- TensorFlow >= 2.x
- Keras Tuner
- NumPy
- Matplotlib
- Pickle (for data loading)

Install with:
```bash
pip install tensorflow keras-tuner numpy matplotlib
```

## License

This framework is part of the csa_31_multiple_models project.

## Support

For issues or questions:
1. Check this README
2. Review example models in `model_defs/`
3. Examine logs in `experiment_results/`
