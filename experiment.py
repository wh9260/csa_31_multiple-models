"""
Modular Machine Learning Experiment Framework

This module provides a flexible framework for running ML experiments with:
- Hyperparameter optimization using Hyperband
- Automatic model saving and result logging
- Support for custom model definitions
- CPU/GPU device management
"""

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

# Import existing data loading functions
from model_training import load_data, prepare_training_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ExperimentFramework:
    """Main experiment framework class for training, validation, and testing."""

    def __init__(
        self,
        model_name: str,
        data_path: str,
        experiment_name: str,
        device: str = 'auto',
        max_epochs: int = 50,
        hyperband_iterations: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Initialize the experiment framework.

        Args:
            model_name: Name of the model definition file (without .py extension)
            data_path: Path to the dataset
            experiment_name: Name for this experiment (used for saving results)
            device: Device to use ('cpu', 'gpu', or 'auto')
            max_epochs: Maximum epochs for hyperparameter search
            hyperband_iterations: Number of Hyperband iterations
            batch_size: Batch size for training
            validation_split: Validation split ratio
        """
        self.model_name = model_name
        self.data_path = Path(data_path)
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.hyperband_iterations = hyperband_iterations
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Set up directories
        self.results_dir = Path("experiment_results") / experiment_name
        self.models_dir = Path("models") / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Configure device
        self._configure_device(device)

        # Load model builder and hyperparameters
        self.model_builder = self._load_model_builder()
        self.hyperparameters = self._load_hyperparameters()

        # Initialize results storage
        self.results = {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_epochs': max_epochs,
                'hyperband_iterations': hyperband_iterations,
                'batch_size': batch_size,
                'validation_split': validation_split
            }
        }

    def _configure_device(self, device: str) -> None:
        """Configure GPU/CPU device for TensorFlow."""
        gpus = tf.config.list_physical_devices('GPU')

        if device == 'cpu':
            # Force CPU usage
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Device: CPU (forced)")
        elif device == 'gpu':
            if gpus:
                try:
                    # Enable memory growth to avoid OOM errors
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Device: GPU ({len(gpus)} GPU(s) available)")
                except RuntimeError as e:
                    logger.warning(f"GPU configuration failed: {e}. Using CPU.")
            else:
                logger.warning("GPU requested but not available. Using CPU.")
        else:  # auto
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Device: GPU ({len(gpus)} GPU(s) available, auto-selected)")
                except RuntimeError as e:
                    logger.warning(f"GPU configuration failed: {e}. Using CPU.")
            else:
                logger.info("Device: CPU (no GPU available)")

    def _load_model_builder(self) -> callable:
        """Dynamically load the model builder function from model_defs directory."""
        model_file = Path("model_defs") / f"{self.model_name}.py"

        if not model_file.exists():
            raise FileNotFoundError(f"Model definition not found: {model_file}")

        spec = importlib.util.spec_from_file_location(self.model_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'build_model'):
            raise AttributeError(
                f"Model file {model_file} must contain a 'build_model' function"
            )

        logger.info(f"Loaded model builder from: {model_file}")
        return module.build_model

    def _load_hyperparameters(self) -> Dict[str, Any]:
        """Load hyperparameter configuration from model_defs directory."""
        params_file = Path("model_defs") / f"{self.model_name}_params.py"

        if not params_file.exists():
            logger.warning(
                f"Hyperparameter file not found: {params_file}. Using defaults."
            )
            return {}

        spec = importlib.util.spec_from_file_location(
            f"{self.model_name}_params", params_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'HYPERPARAMETERS'):
            raise AttributeError(
                f"Params file {params_file} must contain a 'HYPERPARAMETERS' dict"
            )

        logger.info(f"Loaded hyperparameters from: {params_file}")
        return module.HYPERPARAMETERS


    def run_hyperband_search(
        self,
        train_data: Any,
        val_data: Any,
        input_shape: tuple,
        num_classes: int
    ) -> Tuple[keras.Model, kt.HyperParameters, kt.Hyperband]:
        """
        Run Hyperband hyperparameter optimization.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            input_shape: Input shape for the model
            num_classes: Number of output classes

        Returns:
            Tuple of (best_model, best_hyperparameters, tuner)
        """
        logger.info("="*60)
        logger.info("Starting Hyperband hyperparameter optimization")
        logger.info("="*60)

        # Create model builder wrapper that uses loaded hyperparameters
        def model_builder_wrapper(hp):
            return self.model_builder(
                hp, input_shape, num_classes, self.hyperparameters
            )

        # Initialize Hyperband tuner
        tuner = kt.Hyperband(
            model_builder_wrapper,
            objective='val_accuracy',
            max_epochs=self.max_epochs,
            factor=3,
            hyperband_iterations=self.hyperband_iterations,
            directory=str(self.results_dir / "tuner"),
            project_name=self.experiment_name,
            overwrite=False
        )

        # Display search space
        logger.info("Search space summary:")
        tuner.search_space_summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.results_dir / "tensorboard")
            )
        ]

        # Run search
        logger.info("Running Hyperband search...")
        tuner.search(
            train_data,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )

        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        logger.info("Best hyperparameters found:")
        logger.info("-" * 60)
        for param, value in best_hps.values.items():
            logger.info(f"  {param}: {value}")
        logger.info("-" * 60)

        # Build best model
        best_model = tuner.hypermodel.build(best_hps)

        # Store hyperparameters in results
        self.results['best_hyperparameters'] = best_hps.values

        return best_model, best_hps, tuner

    def train_final_model(
        self,
        model: keras.Model,
        train_data: Any,
        val_data: Any,
        epochs: int = None
    ) -> keras.callbacks.History:
        """
        Train the final model with best hyperparameters.

        Args:
            model: Model to train
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs (defaults to max_epochs * 2)

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.max_epochs * 2

        logger.info("="*60)
        logger.info(f"Training final model for {epochs} epochs")
        logger.info("="*60)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=7,
                factor=0.5,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.models_dir / "best_model.keras"),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(self.results_dir / "training_log.csv")
            )
        ]

        # Train
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info(f"Training complete. Model saved to: {self.models_dir / 'best_model.keras'}")

        return history

    def evaluate_model(
        self,
        model: keras.Model,
        test_data: Any
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Args:
            model: Trained model
            test_data: Test dataset

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("="*60)
        logger.info("Evaluating model on test set")
        logger.info("="*60)

        results = model.evaluate(test_data, verbose=1, return_dict=True)

        logger.info("Test Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

        self.results['test_metrics'] = results

        return results

    def plot_training_history(self, history: keras.callbacks.History) -> None:
        """Plot and save training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Plot loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = self.results_dir / "training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training plots saved to: {plot_path}")

    def save_results(self) -> None:
        """Save experiment results to JSON file."""
        results_file = self.results_dir / "experiment_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

    def print_summary(self) -> None:
        """Print a concise summary of experiment results."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {self.model_name}")
        print(f"Timestamp: {self.results['timestamp']}")
        print("-"*60)

        if 'best_hyperparameters' in self.results:
            print("Best Hyperparameters:")
            for param, value in self.results['best_hyperparameters'].items():
                print(f"  {param}: {value}")
            print("-"*60)

        if 'test_metrics' in self.results:
            print("Test Metrics:")
            for metric, value in self.results['test_metrics'].items():
                print(f"  {metric}: {value:.4f}")

        print("-"*60)
        print(f"Results directory: {self.results_dir}")
        print(f"Model saved to: {self.models_dir / 'best_model.keras'}")
        print("="*60 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Modular ML Experiment Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python experiment.py --model simple_cnn --data .data/04_samples/test_cwt.pkl --experiment exp1
  python experiment.py --model simple_cnn --data path/to/data.pkl --experiment exp2 --device gpu --epochs 30
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of model definition file (without .py extension)'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset file'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name for this experiment'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help='Device to use for training (default: auto)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum epochs for hyperparameter search (default: 50)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Hyperband iterations (default: 1)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )

    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum'],
        help='Feature types to use (default: press_flow spo2 rip_abdomen rip_thorax rip_sum)'
    )

    parser.add_argument(
        '--target',
        type=str,
        default='score',
        help='Target column name (default: score)'
    )

    return parser.parse_args()


def main():
    """Main experiment execution function."""
    args = parse_arguments()

    # Initialize framework
    framework = ExperimentFramework(
        model_name=args.model,
        data_path=args.data,
        experiment_name=args.experiment,
        device=args.device,
        max_epochs=args.epochs,
        hyperband_iterations=args.iterations,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )

    try:
        # Load data using existing data loading functions
        logger.info(f"Loading data from: {args.data}")
        data_df = load_data(args.data)

        # Prepare training data
        logger.info(f"Preparing training data with features: {args.features}")
        train_data, val_data, test_data, X_test, y_test, class_map, num_classes = prepare_training_data(
            data_df=data_df,
            feature_types=args.features,
            target_column=args.target,
            batch_size=args.batch_size,
            test_size=0.4,
            val_split=0.5,
            random_state=42
        )

        # Determine input shape from the data
        # Get a batch to infer shape
        for batch_x, batch_y in train_data.take(1):
            input_shape = batch_x.shape[1:]  # Remove batch dimension
            logger.info(f"Detected input shape: {input_shape}")
            logger.info(f"Number of classes: {num_classes}")
            break

        # Run hyperparameter search
        best_model, best_hps, tuner = framework.run_hyperband_search(
            train_data, val_data, input_shape, num_classes
        )

        # Train final model
        history = framework.train_final_model(
            best_model, train_data, val_data
        )

        # Evaluate on test set
        test_metrics = framework.evaluate_model(best_model, test_data)

        # Plot training history
        framework.plot_training_history(history)

        # Save results (include class mapping)
        framework.results['class_map'] = {str(k): int(v) for k, v in class_map.items()}
        framework.save_results()

        # Print summary
        framework.print_summary()

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
