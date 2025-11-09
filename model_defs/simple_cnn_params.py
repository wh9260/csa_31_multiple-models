"""
Hyperparameter Configuration for Simple CNN Model

This file defines the search space for hyperparameter optimization.
Modify these values to customize the hyperparameter search for your experiments.
"""

# Hyperparameter configuration dictionary
HYPERPARAMETERS = {
    # Convolutional layer hyperparameters
    'conv_layers': {
        # First convolutional block
        'c1_filters_min': 32,
        'c1_filters_max': 128,
        'c1_filters_step': 32,
        'c1_kernel_sizes': [3, 5],

        # Second convolutional block
        'c2_filters_min': 64,
        'c2_filters_max': 256,
        'c2_filters_step': 64,
        'c2_kernel_sizes': [3, 5, 7],

        # Third convolutional block (optional)
        'c3_filters_min': 128,
        'c3_filters_max': 512,
        'c3_filters_step': 128,

        # Dropout for convolutional blocks
        'dropout_min': 0.0,
        'dropout_max': 0.5,
        'dropout_step': 0.1,
    },

    # Dense layer hyperparameters
    'dense_layers': {
        # First dense layer
        'd1_units_min': 128,
        'd1_units_max': 512,
        'd1_units_step': 128,

        # Second dense layer (optional)
        'd2_units_min': 64,
        'd2_units_max': 256,
        'd2_units_step': 64,

        # Dropout for dense layers
        'dropout_min': 0.0,
        'dropout_max': 0.5,
        'dropout_step': 0.1,
    },

    # Optimizer configuration
    'optimizer': {
        'types': ['adam', 'rmsprop'],  # Available optimizers to try
        'lr_min': 1e-4,  # Minimum learning rate
        'lr_max': 1e-2,  # Maximum learning rate
    },
}


# Optional: Define specific configurations for quick testing
QUICK_TEST_PARAMS = {
    """Use this for quick testing with reduced search space."""
    'conv_layers': {
        'c1_filters_min': 32,
        'c1_filters_max': 64,
        'c1_filters_step': 32,
        'c1_kernel_sizes': [3],

        'c2_filters_min': 64,
        'c2_filters_max': 128,
        'c2_filters_step': 64,
        'c2_kernel_sizes': [3],

        'c3_filters_min': 128,
        'c3_filters_max': 256,
        'c3_filters_step': 128,

        'dropout_min': 0.2,
        'dropout_max': 0.3,
        'dropout_step': 0.1,
    },

    'dense_layers': {
        'd1_units_min': 128,
        'd1_units_max': 256,
        'd1_units_step': 128,

        'd2_units_min': 64,
        'd2_units_max': 128,
        'd2_units_step': 64,

        'dropout_min': 0.3,
        'dropout_max': 0.5,
        'dropout_step': 0.2,
    },

    'optimizer': {
        'types': ['adam'],
        'lr_min': 1e-3,
        'lr_max': 1e-2,
    },
}


# Optional: Define specific configurations for thorough search
THOROUGH_SEARCH_PARAMS = {
    """Use this for comprehensive hyperparameter search."""
    'conv_layers': {
        'c1_filters_min': 16,
        'c1_filters_max': 128,
        'c1_filters_step': 16,
        'c1_kernel_sizes': [3, 5, 7],

        'c2_filters_min': 32,
        'c2_filters_max': 256,
        'c2_filters_step': 32,
        'c2_kernel_sizes': [3, 5, 7],

        'c3_filters_min': 64,
        'c3_filters_max': 512,
        'c3_filters_step': 64,

        'dropout_min': 0.0,
        'dropout_max': 0.6,
        'dropout_step': 0.1,
    },

    'dense_layers': {
        'd1_units_min': 64,
        'd1_units_max': 1024,
        'd1_units_step': 64,

        'd2_units_min': 32,
        'd2_units_max': 512,
        'd2_units_step': 32,

        'dropout_min': 0.0,
        'dropout_max': 0.6,
        'dropout_step': 0.1,
    },

    'optimizer': {
        'types': ['adam', 'rmsprop', 'sgd'],
        'lr_min': 1e-5,
        'lr_max': 1e-1,
    },
}


# Configuration notes:
# - Adjust min/max values based on your dataset size and complexity
# - Smaller step sizes increase search space but take longer
# - More optimizer types increase search time
# - Use QUICK_TEST_PARAMS for initial experiments
# - Use THOROUGH_SEARCH_PARAMS when you have compute resources
# - Use HYPERPARAMETERS as a balanced middle ground
