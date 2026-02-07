"""
Configuration file for the Geocentric Orbit Prediction project.

Contains constants, hyperparameters, and project settings.
"""
from datetime import timedelta

# ===== FILE PATHS =====
DEFAULT_EPHEMERIS_FILE = 'de421.bsp'
DATA_DIR = 'data'
MODELS_DIR = 'models'
VISUALIZATIONS_DIR = 'visualizations'
SUMMARY_DIR = 'summary'

# ===== DATA GENERATION =====
DEFAULT_TIME_STEP = timedelta(days=1)
DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2025

# ===== TRAINING HYPERPARAMETERS =====
TRAIN_TEST_SPLIT_RATIO = 0.8
BATCH_SIZE = 64
MAX_EPOCHS = 1000
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 50

# Model Architecture
LAYER_SIZES = [256, 128, 64]
OUTPUT_SIZE = 3  # X, Y, Z residuals
ACTIVATION = 'relu'

# ===== PHYSICS CONSTANTS =====
# Tolerances
KEPLER_TOLERANCE = 1e-6
KEPLER_MAX_ITERATIONS = 50

# Numerical stability
EPSILON = 1e-9  # Small value to prevent division by zero

# ===== VISUALIZATION =====
VIZ_SAMPLE_SIZE = 2000  # Number of points to plot in 3D visualizations
PLOT_STYLE = 'plotly_dark'

# ===== SUPPORTED PLANETS =====
INNER_PLANETS = ['mercury', 'venus', 'mars']
OUTER_PLANETS = ['jupiter', 'saturn', 'uranus', 'neptune']
ALL_PLANETS = INNER_PLANETS + OUTER_PLANETS

# ===== FEATURES =====
# Base time features
TIME_FEATURES = ['Time_Index', 'Time_Index_2']

# Lag features (for momentum/velocity)
LAG_STEPS = [1, 2]

# Perturbers to consider
PERTURBERS = ['Jupiter', 'Saturn', 'Venus', 'Earth']

# ===== TARGET COLUMNS =====
GEOCENTRIC_COORDS = ['X_au', 'Y_au', 'Z_au']
KEPLERIAN_COORDS = ['Kepler_X', 'Kepler_Y', 'Kepler_Z']
RESIDUAL_TARGETS = ['Res_X', 'Res_Y', 'Res_Z']

# ===== LOGGING =====
LOG_FREQUENCY = 10  # Log every N epochs during training
VERBOSE_TRAINING = False  # Set to True for detailed training output
