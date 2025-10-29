"""
Hyperband-optimized CNN model builder for hyperparameter tuning.

This module defines a flexible CNN architecture with tunable hyperparameters
for use with Keras Tuner's Hyperband algorithm.
"""

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models


def build_hyperband_model(hp: kt.HyperParameters, input_shape: tuple, num_classes: int) -> models.Sequential:
    """
    Build a CNN model with tunable hyperparameters for Hyperband optimization.

    Tunable hyperparameters:
    - Number of convolutional filters in each layer
    - Kernel sizes for convolutional layers
    - Number of units in dense layers
    - Dropout rates
    - Learning rate
    - Optimizer choice

    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras Sequential model with tuned hyperparameters
    """

    model = models.Sequential()

    # --- First Convolutional Block (C1 + S2) ---
    hp_c1_filters = hp.Int('c1_filters', min_value=32, max_value=128, step=32)
    hp_c1_kernel = hp.Choice('c1_kernel_size', values=[3, 5])

    model.add(layers.Conv2D(
        hp_c1_filters,
        (hp_c1_kernel, hp_c1_kernel),
        activation='relu',
        strides=1,
        padding='same',
        input_shape=input_shape,
        name='C1_Conv2D'
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S2_MaxPool'
    ))

    # Optional: Add dropout after first pooling
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    if hp_dropout_1 > 0:
        model.add(layers.Dropout(hp_dropout_1, name='Dropout_1'))

    # --- Second Convolutional Block (C3 + S4) ---
    hp_c3_filters = hp.Int('c3_filters', min_value=64, max_value=256, step=64)
    hp_c3_kernel = hp.Choice('c3_kernel_size', values=[3, 5, 7])

    model.add(layers.Conv2D(
        hp_c3_filters,
        (hp_c3_kernel, hp_c3_kernel),
        activation='relu',
        strides=1,
        padding='same',
        name='C3_Conv2D'
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S4_MaxPool'
    ))

    # Optional: Add dropout after second pooling
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
    if hp_dropout_2 > 0:
        model.add(layers.Dropout(hp_dropout_2, name='Dropout_2'))

    # --- Optional Third Convolutional Block ---
    hp_use_c5 = hp.Boolean('use_third_conv_block')
    if hp_use_c5:
        hp_c5_filters = hp.Int('c5_filters', min_value=128, max_value=512, step=128)
        model.add(layers.Conv2D(
            hp_c5_filters,
            (3, 3),
            activation='relu',
            strides=1,
            padding='same',
            name='C5_Conv2D'
        ))
        model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            padding='same',
            name='S6_MaxPool'
        ))

    # --- Flatten ---
    model.add(layers.Flatten())

    # --- First Dense Layer (F5) ---
    hp_f5_units = hp.Int('f5_units', min_value=128, max_value=512, step=128)
    model.add(layers.Dense(hp_f5_units, activation='relu', name='F5_Dense'))

    hp_dropout_3 = hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)
    if hp_dropout_3 > 0:
        model.add(layers.Dropout(hp_dropout_3, name='Dropout_3'))

    # --- Optional Second Dense Layer (F6) ---
    hp_use_f6 = hp.Boolean('use_second_dense')
    if hp_use_f6:
        hp_f6_units = hp.Int('f6_units', min_value=64, max_value=256, step=64)
        model.add(layers.Dense(hp_f6_units, activation='relu', name='F6_Dense'))

        hp_dropout_4 = hp.Float('dropout_4', min_value=0.0, max_value=0.5, step=0.1)
        if hp_dropout_4 > 0:
            model.add(layers.Dropout(hp_dropout_4, name='Dropout_4'))

    # --- Output Layer (F7) ---
    model.add(layers.Dense(num_classes, activation='softmax', name='F7_Output'))

    # --- Optimizer Selection ---
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if hp_optimizer == 'adam':
        optimizer = 'adam'
    elif hp_optimizer == 'rmsprop':
        optimizer = 'rmsprop'
    else:  # sgd
        optimizer = 'sgd'

    # Compile with learning rate
    if hp_optimizer == 'adam':
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'rmsprop':
        from tensorflow.keras.optimizers import RMSprop
        optimizer = RMSprop(learning_rate=hp_learning_rate)
    else:
        from tensorflow.keras.optimizers import SGD
        optimizer = SGD(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_simple_search_model(hp: kt.HyperParameters, input_shape: tuple, num_classes: int) -> models.Sequential:
    """
    Build a simpler CNN model with fewer tunable hyperparameters for faster searches.

    This is useful for quick experiments or when computational resources are limited.

    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras Sequential model with tuned hyperparameters
    """

    model = models.Sequential([
        # C1
        layers.Conv2D(
            hp.Choice('c1_filters', values=[32, 64, 128]),
            (3, 3),
            activation='relu',
            strides=1,
            padding='same',
            input_shape=input_shape,
            name='C1_Conv2D'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='S2_MaxPool'),

        # C3
        layers.Conv2D(
            hp.Choice('c3_filters', values=[64, 128, 256]),
            (5, 5),
            activation='relu',
            strides=1,
            padding='same',
            name='C3_Conv2D'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='S4_MaxPool'),

        # Dense layers
        layers.Flatten(),
        layers.Dense(
            hp.Choice('dense_units', values=[128, 256, 512]),
            activation='relu',
            name='F5_Dense'
        ),
        layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)),
        layers.Dense(num_classes, activation='softmax', name='F7_Output')
    ])

    # Learning rate tuning
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
