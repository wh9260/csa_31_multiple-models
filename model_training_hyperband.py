"""
Hyperband-based hyperparameter optimization for CNN model.

This module uses Keras Tuner's Hyperband algorithm to efficiently search
for optimal hyperparameters for the CNN classification model.
"""

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from pathlib import Path

# Import data loading and preparation from the base training module
from model_training import load_data, prepare_training_data, show_results
from hyperband_model import build_hyperband_model, build_simple_search_model


def run_hyperband_search(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    input_shape: tuple,
    num_classes: int,
    project_name: str = "cnn_hyperband",
    max_epochs: int = 50,
    factor: int = 3,
    hyperband_iterations: int = 1,
    directory: str = "hyperband_results",
    use_simple_model: bool = False
) -> tuple:
    """
    Run Hyperband hyperparameter search to find optimal model configuration.

    Hyperband is an efficient hyperparameter optimization algorithm that uses
    adaptive resource allocation and early-stopping to quickly find good
    hyperparameter configurations.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        project_name: Name for the tuning project
        max_epochs: Maximum training epochs per trial
        factor: Reduction factor for Hyperband (controls aggressive early stopping)
        hyperband_iterations: Number of times to iterate over the full Hyperband algorithm
        directory: Directory to save tuning results
        use_simple_model: If True, use simpler model with fewer hyperparameters

    Returns:
        Tuple of (best_model, best_hyperparameters, tuner)
    """

    print("\n" + "="*60)
    print("HYPERBAND HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Project: {project_name}")
    print(f"Max epochs per trial: {max_epochs}")
    print(f"Factor: {factor}")
    print(f"Hyperband iterations: {hyperband_iterations}")
    print(f"Results directory: {directory}")
    print("="*60 + "\n")

    # Select model builder
    model_builder = build_simple_search_model if use_simple_model else build_hyperband_model

    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        lambda hp: model_builder(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        directory=directory,
        project_name=project_name,
        overwrite=False  # Set to True to start fresh
    )

    # Display search space summary
    print("\nSearch space summary:")
    tuner.search_space_summary()

    # Define callbacks for training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Run the search
    print("\nStarting Hyperband search...")
    print("This may take a while depending on the search space and max_epochs.\n")

    tuner.search(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest hyperparameters found:")
    print("-" * 60)
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")
    print("-" * 60)

    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)

    return best_model, best_hps, tuner


def train_best_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str = "best_hyperband_model",
    epochs: int = 100,
    save_dir: str = "models"
) -> tuple:
    """
    Train the best model found by Hyperband with extended epochs.

    Args:
        model: Best model from hyperparameter search
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving the model
        epochs: Number of training epochs
        save_dir: Directory to save the best model

    Returns:
        Tuple of (trained model, training history)
    """
    print("\n" + "="*60)
    print("TRAINING BEST MODEL")
    print("="*60)
    print(f"Model name: {model_name}")
    print(f"Epochs: {epochs}")
    print("="*60 + "\n")

    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / f"{model_name}.keras"

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=7,
            factor=0.5,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_file),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nTraining complete. Best model saved to {model_file}")

    return model, history


def display_top_models(tuner: kt.Hyperband, top_n: int = 5) -> None:
    """
    Display the top N models from the Hyperband search.

    Args:
        tuner: Hyperband tuner object after search
        top_n: Number of top models to display
    """
    print("\n" + "="*60)
    print(f"TOP {top_n} MODELS FROM HYPERBAND SEARCH")
    print("="*60)

    top_models = tuner.get_best_models(num_models=top_n)
    top_hps = tuner.get_best_hyperparameters(num_trials=top_n)

    for i, (model, hps) in enumerate(zip(top_models, top_hps), 1):
        print(f"\n--- Model {i} ---")
        print("Hyperparameters:")
        for param, value in hps.values.items():
            print(f"  {param}: {value}")

        # Get validation accuracy from tuner oracle
        trial = tuner.oracle.get_trial(tuner.oracle.get_best_trials(top_n)[i-1].trial_id)
        if trial.metrics.exists('val_accuracy'):
            best_val_acc = trial.metrics.get_best_value('val_accuracy')
            print(f"Best validation accuracy: {best_val_acc:.4f}")

    print("="*60)


def main():
    """Main function to run Hyperband hyperparameter optimization pipeline."""

    # Configuration
    input_file_path = '.data/04_samples/test_cwt.pkl'
    feature_types = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']
    project_name = "cnn_hyperband_optimization"

    # Hyperband configuration
    # For quick testing, use smaller values:
    # max_epochs=10, hyperband_iterations=1, final_epochs=20
    # For thorough search, use larger values:
    # max_epochs=50, hyperband_iterations=2, final_epochs=100
    max_epochs = 30  # Maximum epochs for each trial in search
    hyperband_iterations = 1  # Number of Hyperband iterations
    final_training_epochs = 50  # Epochs for final training of best model

    # 1. Load data
    data_df = load_data(input_file_path)

    # 2. Prepare training data
    train_ds, val_ds, test_ds, X_test, y_test, class_map, num_classes = prepare_training_data(
        data_df=data_df,
        feature_types=feature_types,
        target_column='score',
        batch_size=16
    )

    # Determine input shape
    input_shape = (128, 128, 5)  # Based on the data structure

    # 3. Run Hyperband search
    # Set use_simple_model=True for faster experimentation
    best_model, best_hps, tuner = run_hyperband_search(
        train_ds=train_ds,
        val_ds=val_ds,
        input_shape=input_shape,
        num_classes=num_classes,
        project_name=project_name,
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=hyperband_iterations,
        directory="hyperband_results",
        use_simple_model=False  # Set to True for faster search with fewer hyperparameters
    )

    # 4. Display top models
    display_top_models(tuner, top_n=3)

    # 5. Train best model with extended epochs
    model_name = "cnn_hyperband_optimized"
    best_model, history = train_best_model(
        model=best_model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=model_name,
        epochs=final_training_epochs,
        save_dir="models"
    )

    # 6. Show results
    show_results(
        model=best_model,
        test_ds=test_ds,
        X_test=X_test,
        y_test=y_test,
        history=history,
        class_map=class_map,
        model_name=model_name,
        save_plots=True
    )

    # 7. Save best hyperparameters to file
    hp_file = Path("models") / f"{model_name}_hyperparameters.txt"
    with open(hp_file, 'w') as f:
        f.write("Best Hyperparameters from Hyperband Search\n")
        f.write("=" * 60 + "\n\n")
        for param, value in best_hps.values.items():
            f.write(f"{param}: {value}\n")

    print(f"\nBest hyperparameters saved to {hp_file}")

    print("\n" + "="*60)
    print("HYPERBAND OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best model saved to: models/{model_name}.keras")
    print(f"Hyperparameters saved to: {hp_file}")
    print(f"Hyperband search results in: hyperband_results/{project_name}/")
    print("="*60)


if __name__ == "__main__":
    main()
