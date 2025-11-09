"""
Simple MLP (Multi-Layer Perceptron) Model Definition

This module defines a simple feedforward neural network architecture
with tunable hyperparameters for use with the experiment framework.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from typing import Dict, Any


def build_model(
    hp: kt.HyperParameters,
    input_shape: tuple,
    num_classes: int,
    hyperparameters: Dict[str, Any]
) -> models.Sequential:
    """
    Build an MLP model with tunable hyperparameters.

    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Shape of input data (height, width, channels) - will be flattened
        num_classes: Number of output classes
        hyperparameters: Dictionary containing hyperparameter ranges from config file

    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential()

    # Flatten input if needed
    model.add(layers.Flatten(input_shape=input_shape, name='Flatten'))

    # Get hyperparameter configurations
    hidden_params = hyperparameters.get('hidden_layers', {})
    optimizer_params = hyperparameters.get('optimizer', {})

    # Number of hidden layers
    hp_num_layers = hp.Int(
        'num_hidden_layers',
        min_value=hidden_params.get('num_layers_min', 1),
        max_value=hidden_params.get('num_layers_max', 4),
        step=1
    )

    # Build hidden layers
    for i in range(hp_num_layers):
        # Units for this layer
        hp_units = hp.Int(
            f'units_layer_{i}',
            min_value=hidden_params.get('units_min', 64),
            max_value=hidden_params.get('units_max', 512),
            step=hidden_params.get('units_step', 64)
        )

        # Activation function
        hp_activation = hp.Choice(
            f'activation_layer_{i}',
            values=hidden_params.get('activations', ['relu', 'tanh'])
        )

        model.add(layers.Dense(
            hp_units,
            activation=hp_activation,
            name=f'Hidden_{i+1}'
        ))

        # Dropout
        hp_dropout = hp.Float(
            f'dropout_layer_{i}',
            min_value=hidden_params.get('dropout_min', 0.0),
            max_value=hidden_params.get('dropout_max', 0.5),
            step=hidden_params.get('dropout_step', 0.1)
        )
        if hp_dropout > 0:
            model.add(layers.Dropout(hp_dropout, name=f'Dropout_{i+1}'))

        # Optional batch normalization
        if hidden_params.get('use_batch_norm', False):
            hp_use_bn = hp.Boolean(f'use_batch_norm_layer_{i}')
            if hp_use_bn:
                model.add(layers.BatchNormalization(name=f'BatchNorm_{i+1}'))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))

    # Optimizer configuration
    hp_optimizer = hp.Choice(
        'optimizer',
        values=optimizer_params.get('types', ['adam', 'rmsprop'])
    )
    hp_learning_rate = hp.Float(
        'learning_rate',
        min_value=optimizer_params.get('lr_min', 1e-4),
        max_value=optimizer_params.get('lr_max', 1e-2),
        sampling='log'
    )

    # Create optimizer
    if hp_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
    else:  # sgd
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_baseline_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    """
    Build a baseline MLP model without hyperparameter tuning.

    Args:
        input_shape: Shape of input data (will be flattened)
        num_classes: Number of output classes

    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape, name='Flatten'),
        layers.Dense(256, activation='relu', name='Hidden_1'),
        layers.Dropout(0.3, name='Dropout_1'),
        layers.Dense(128, activation='relu', name='Hidden_2'),
        layers.Dropout(0.3, name='Dropout_2'),
        layers.Dense(num_classes, activation='softmax', name='Output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
