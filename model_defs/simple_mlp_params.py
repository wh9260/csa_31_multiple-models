"""
Hyperparameter Configuration for Simple MLP Model

This file defines the search space for hyperparameter optimization
of the Multi-Layer Perceptron architecture.
"""

# Hyperparameter configuration dictionary
HYPERPARAMETERS = {
    # Hidden layer hyperparameters
    'hidden_layers': {
        # Number of hidden layers
        'num_layers_min': 1,
        'num_layers_max': 4,

        # Units per hidden layer
        'units_min': 64,
        'units_max': 512,
        'units_step': 64,

        # Activation functions to try
        'activations': ['relu', 'tanh'],

        # Dropout
        'dropout_min': 0.0,
        'dropout_max': 0.5,
        'dropout_step': 0.1,

        # Batch normalization
        'use_batch_norm': False,  # Set to True to enable batch norm tuning
    },

    # Optimizer configuration
    'optimizer': {
        'types': ['adam', 'rmsprop'],
        'lr_min': 1e-4,
        'lr_max': 1e-2,
    },
}


# Quick test configuration
QUICK_TEST_PARAMS = {
    'hidden_layers': {
        'num_layers_min': 2,
        'num_layers_max': 3,
        'units_min': 128,
        'units_max': 256,
        'units_step': 128,
        'activations': ['relu'],
        'dropout_min': 0.2,
        'dropout_max': 0.4,
        'dropout_step': 0.2,
        'use_batch_norm': False,
    },
    'optimizer': {
        'types': ['adam'],
        'lr_min': 1e-3,
        'lr_max': 1e-2,
    },
}


# Thorough search configuration
THOROUGH_SEARCH_PARAMS = {
    'hidden_layers': {
        'num_layers_min': 1,
        'num_layers_max': 5,
        'units_min': 32,
        'units_max': 1024,
        'units_step': 32,
        'activations': ['relu', 'tanh', 'elu'],
        'dropout_min': 0.0,
        'dropout_max': 0.6,
        'dropout_step': 0.1,
        'use_batch_norm': True,
    },
    'optimizer': {
        'types': ['adam', 'rmsprop', 'sgd'],
        'lr_min': 1e-5,
        'lr_max': 1e-1,
    },
}
