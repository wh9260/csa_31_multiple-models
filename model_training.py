import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        DataFrame containing the loaded data
    """
    print(f"Loading data from {file_path}...")
    data_df = pd.read_pickle(file_path)
    print(f"Data loaded: {len(data_df)} samples")
    return data_df


def prepare_training_data(
    data_df: pd.DataFrame,
    feature_types: list[str],
    target_column: str = 'score',
    test_size: float = 0.4,
    val_split: float = 0.5,
    batch_size: int = 32,
    random_state: int = 42
) -> tuple:
    """
    Prepare data for training by normalizing, splitting, and creating TensorFlow datasets.

    Args:
        data_df: Input DataFrame with features and labels
        feature_types: List of feature column names to use
        target_column: Name of the target column
        test_size: Proportion of data for validation+test
        val_split: Proportion of test data to use for validation
        batch_size: Batch size for training
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - train_ds: Training dataset
        - val_ds: Validation dataset
        - test_ds: Test dataset
        - X_test: Test features (for evaluation)
        - y_test: Test labels (for evaluation)
        - class_map: Mapping from original to encoded classes
        - num_classes: Number of classes
    """
    print("Preparing training data...")

    # Stack features
    X = np.stack(data_df[feature_types].apply(
        lambda row: np.stack(row.values, axis=-1), axis=1
    ))
    X = np.squeeze(X)
    X = X.astype('float32')

    # Normalize
    X = (X - np.mean(X)) / np.std(X)
    print(f"Feature shape: {X.shape}")

    # Prepare labels
    y = data_df[target_column].to_numpy()
    unique_classes = np.unique(y)
    print(f"Unique classes: {unique_classes}")

    # Map classes to indices
    class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_mapped = np.array([class_map[label] for label in y])
    print(f"Mapped classes: {np.unique(y_mapped)}")

    # One-hot encode
    num_classes = len(unique_classes)
    y_cat = to_categorical(y_mapped, num_classes=num_classes)

    # Train/validation/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_cat, test_size=test_size, stratify=y_mapped, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=val_split,
        stratify=np.argmax(y_tmp, axis=1), random_state=random_state
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
        .shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, X_test, y_test, class_map, num_classes


def create_cnn_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    """
    Create a CNN model for classification.

    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras Sequential model
    """
    print(f"Creating CNN model with input shape {input_shape}...")

    model = models.Sequential([
        # C1 - First convolutional layer
        layers.Conv2D(
            64, (3, 3),
            activation='relu',
            strides=1,
            padding='same',
            input_shape=input_shape,
            name='C1_Conv2D'
        ),

        # S2 - First pooling layer
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            padding='same',
            name='S2_MaxPool'
        ),

        # C3 - Second convolutional layer
        layers.Conv2D(
            128,
            (5, 5),
            activation='relu',
            strides=1,
            padding='same',
            name='C3_Conv2D'
        ),

        # S4 - Second pooling layer
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            padding='same',
            name='S4_MaxPool'
        ),

        # Flatten before fully connected layers
        layers.Flatten(),

        # F5 - First dense layer
        layers.Dense(256, activation='relu', name='F5_Dense'),

        # F6 - Second dense layer
        layers.Dense(128, activation='relu', name='F6_Dense'),

        # F7 - Output layer
        layers.Dense(num_classes, activation='softmax', name='F7_Output')
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_model(
    model: models.Sequential,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str = "model",
    epochs: int = 50,
    save_dir: str = "models"
) -> tuple:
    """
    Train the model with callbacks for early stopping and learning rate reduction.

    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving the model
        epochs: Maximum number of training epochs
        save_dir: Directory to save the best model

    Returns:
        Tuple of (trained model, training history)
    """
    print(f"Training model for up to {epochs} epochs...")

    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / f"{model_name}.keras"

    callbacks = [
        EarlyStopping(
            patience=8,
            restore_best_weights=True,
            monitor="val_loss"
        ),
        ReduceLROnPlateau(
            patience=4,
            factor=0.5,
            monitor="val_loss"
        ),
        ModelCheckpoint(
            str(model_file),
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {model_file}")
    return model, history


def show_results(
    model: models.Sequential,
    test_ds: tf.data.Dataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history,
    class_map: dict,
    model_name: str = "model",
    save_plots: bool = True
) -> None:
    """
    Evaluate the model and display results including accuracy, loss curves, and confusion matrix.

    Args:
        model: Trained Keras model
        test_ds: Test dataset
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        history: Training history object
        class_map: Mapping from original to encoded classes
        model_name: Name for saving plots
        save_plots: Whether to save plots to disk
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Create inverse mapping
    inverse_map = {v: k for k, v in class_map.items()}

    # Get predictions
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Map back to original labels
    y_true_labels = np.array([inverse_map[i] for i in y_true])
    y_pred_labels = np.array([inverse_map[i] for i in y_pred])

    # Plot training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.plot(epochs, val_acc, 'r--', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{model_name}_training_history.png', dpi=150)
        print(f"Training history saved to {model_name}_training_history.png")
    plt.close()

    # Plot model architecture (requires system graphviz installation)
    try:
        plot_model(
            model,
            to_file=f'{model_name}_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=100
        )
        print(f"Model architecture saved to {model_name}_architecture.png")
    except ImportError:
        print("Warning: Graphviz not installed. Skipping model architecture plot.")
        print("To enable: Install graphviz system package (apt/brew install graphviz)")

    # Confusion matrix
    class_names = sorted(list(class_map.keys()))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {model_name}')

    if save_plots:
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150)
        print(f"Confusion matrix saved to {model_name}_confusion_matrix.png")
    plt.close()

    print("="*60)


def main():
    """Main function to run the complete training pipeline."""

    # Configuration
    input_file_path = '.data/04_samples/test_cwt.pkl'
    feature_types = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']
    model_name = "cnn_baseline"

    # 1. Load data
    data_df = load_data(input_file_path)

    # 2. Prepare training data
    train_ds, val_ds, test_ds, X_test, y_test, class_map, num_classes = prepare_training_data(
        data_df=data_df,
        feature_types=feature_types,
        target_column='score',
        batch_size=32
    )

    # Determine input shape from the data
    input_shape = (128, 128, 5)  # Based on the original code

    # 3. Create model
    model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)

    # 4. Train model
    model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=model_name,
        epochs=50,
        save_dir="models"
    )

    # 5. Show results
    show_results(
        model=model,
        test_ds=test_ds,
        X_test=X_test,
        y_test=y_test,
        history=history,
        class_map=class_map,
        model_name=model_name,
        save_plots=True
    )


if __name__ == "__main__":
    main()
