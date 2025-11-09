"""
Simple CNN Model Definition

This module defines a simple CNN architecture for image classification
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
    Build a CNN model with tunable hyperparameters.

    This function is called by the experiment framework and Keras Tuner
    to create model instances with different hyperparameter configurations.

    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        hyperparameters: Dictionary containing hyperparameter ranges from config file

    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential()

    # Get hyperparameter configurations
    conv_params = hyperparameters.get('conv_layers', {})
    dense_params = hyperparameters.get('dense_layers', {})
    optimizer_params = hyperparameters.get('optimizer', {})

    # === First Convolutional Block ===
    hp_c1_filters = hp.Int(
        'c1_filters',
        min_value=conv_params.get('c1_filters_min', 32),
        max_value=conv_params.get('c1_filters_max', 128),
        step=conv_params.get('c1_filters_step', 32)
    )
    hp_c1_kernel = hp.Choice(
        'c1_kernel_size',
        values=conv_params.get('c1_kernel_sizes', [3, 5])
    )

    model.add(layers.Conv2D(
        hp_c1_filters,
        (hp_c1_kernel, hp_c1_kernel),
        activation='relu',
        padding='same',
        input_shape=input_shape,
        name='C1_Conv2D'
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        name='S2_MaxPool'
    ))

    # Optional dropout after first conv block
    hp_dropout_1 = hp.Float(
        'dropout_1',
        min_value=conv_params.get('dropout_min', 0.0),
        max_value=conv_params.get('dropout_max', 0.5),
        step=conv_params.get('dropout_step', 0.1)
    )
    if hp_dropout_1 > 0:
        model.add(layers.Dropout(hp_dropout_1, name='Dropout_1'))

    # === Second Convolutional Block ===
    hp_c2_filters = hp.Int(
        'c2_filters',
        min_value=conv_params.get('c2_filters_min', 64),
        max_value=conv_params.get('c2_filters_max', 256),
        step=conv_params.get('c2_filters_step', 64)
    )
    hp_c2_kernel = hp.Choice(
        'c2_kernel_size',
        values=conv_params.get('c2_kernel_sizes', [3, 5, 7])
    )

    model.add(layers.Conv2D(
        hp_c2_filters,
        (hp_c2_kernel, hp_c2_kernel),
        activation='relu',
        padding='same',
        name='C2_Conv2D'
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        name='S4_MaxPool'
    ))

    # Optional dropout after second conv block
    hp_dropout_2 = hp.Float(
        'dropout_2',
        min_value=conv_params.get('dropout_min', 0.0),
        max_value=conv_params.get('dropout_max', 0.5),
        step=conv_params.get('dropout_step', 0.1)
    )
    if hp_dropout_2 > 0:
        model.add(layers.Dropout(hp_dropout_2, name='Dropout_2'))

    # === Optional Third Convolutional Block ===
    hp_use_c3 = hp.Boolean('use_third_conv_block')
    if hp_use_c3:
        hp_c3_filters = hp.Int(
            'c3_filters',
            min_value=conv_params.get('c3_filters_min', 128),
            max_value=conv_params.get('c3_filters_max', 512),
            step=conv_params.get('c3_filters_step', 128)
        )
        model.add(layers.Conv2D(
            hp_c3_filters,
            (3, 3),
            activation='relu',
            padding='same',
            name='C3_Conv2D'
        ))
        model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same',
            name='S6_MaxPool'
        ))

    # === Flatten ===
    model.add(layers.Flatten())

    # === First Dense Layer ===
    hp_d1_units = hp.Int(
        'd1_units',
        min_value=dense_params.get('d1_units_min', 128),
        max_value=dense_params.get('d1_units_max', 512),
        step=dense_params.get('d1_units_step', 128)
    )
    model.add(layers.Dense(hp_d1_units, activation='relu', name='D1_Dense'))

    # Dropout after first dense
    hp_dropout_3 = hp.Float(
        'dropout_3',
        min_value=dense_params.get('dropout_min', 0.0),
        max_value=dense_params.get('dropout_max', 0.5),
        step=dense_params.get('dropout_step', 0.1)
    )
    if hp_dropout_3 > 0:
        model.add(layers.Dropout(hp_dropout_3, name='Dropout_3'))

    # === Optional Second Dense Layer ===
    hp_use_d2 = hp.Boolean('use_second_dense')
    if hp_use_d2:
        hp_d2_units = hp.Int(
            'd2_units',
            min_value=dense_params.get('d2_units_min', 64),
            max_value=dense_params.get('d2_units_max', 256),
            step=dense_params.get('d2_units_step', 64)
        )
        model.add(layers.Dense(hp_d2_units, activation='relu', name='D2_Dense'))

        hp_dropout_4 = hp.Float(
            'dropout_4',
            min_value=dense_params.get('dropout_min', 0.0),
            max_value=dense_params.get('dropout_max', 0.5),
            step=dense_params.get('dropout_step', 0.1)
        )
        if hp_dropout_4 > 0:
            model.add(layers.Dropout(hp_dropout_4, name='Dropout_4'))

    # === Output Layer ===
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))

    # === Optimizer Configuration ===
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

    # === Compile Model ===
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_baseline_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    """
    Build a baseline CNN model without hyperparameter tuning.

    This is useful for quick testing or as a reference point.

    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, name='C1_Conv2D'),
        layers.MaxPooling2D(pool_size=(2, 2), name='S2_MaxPool'),
        layers.Dropout(0.25, name='Dropout_1'),

        # Conv Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='C2_Conv2D'),
        layers.MaxPooling2D(pool_size=(2, 2), name='S4_MaxPool'),
        layers.Dropout(0.25, name='Dropout_2'),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='D1_Dense'),
        layers.Dropout(0.5, name='Dropout_3'),
        layers.Dense(num_classes, activation='softmax', name='Output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
