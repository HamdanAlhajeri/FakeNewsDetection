"""
Configuration and constants for Fake News Detection project.
"""

from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
SRC_DIR = PROJECT_ROOT / 'src'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Random seed for reproducibility
RANDOM_SEED = 42

# Truthfulness labels (6-class classification)
TRUTHFULNESS_LABELS = {
    0: 'true',
    1: 'mostly-true',
    2: 'half-true',
    3: 'barely-true',
    4: 'false',
    5: 'pants-on-fire'
}

# Reverse mapping
LABEL_TO_IDX = {v: k for k, v in TRUTHFULNESS_LABELS.items()}

# Data processing parameters
DATA_CONFIG = {
    'text_column': 'statement',
    'label_column': 'label',
    'remove_stopwords': True,
    'min_word_freq': 1,
    'max_sequence_length': 100,
    'vocab_size': None,  # Will be determined from data
}

# Model parameters
MODEL_CONFIG = {
    'embedding_dim': 100,
    'max_sequence_length': 100,
    'num_classes': 6,
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
}

# File paths
FILE_PATHS = {
    'processed_data': ARTIFACTS_DIR / 'processed_data.csv',
    'encoded_data': ARTIFACTS_DIR / 'encoded_data.npz',
    'processor': ARTIFACTS_DIR / 'processor.pkl',
    'vocab': ARTIFACTS_DIR / 'vocabulary.pkl',
    'model': ARTIFACTS_DIR / 'model.pkl',
    'model_weights': ARTIFACTS_DIR / 'model_weights.h5',
}

# LIAR2 dataset info
LIAR2_INFO = {
    'url': 'https://www.cs.ucsb.edu/~william/data/liar_plus_dataset.zip',
    'num_samples': 23000,
    'num_classes': 6,
    'source': 'PolitiFact',
}

if __name__ == '__main__':
    print("Fake News Detection - Configuration")
    print("\nTruthfulness Labels:")
    for idx, label in enumerate(TRUTHFULNESS_LABELS):
        print(f"  {idx}: {label}")
    
    print("\nProject Directories:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print(f"  Source: {SRC_DIR}")
