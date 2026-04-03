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
    0: 'barely-true',
    1: 'false',
    2: 'half-true',
    3: 'mostly-true',
    4: 'pants-fire',
    5: 'true'
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

# Tinker LLM configuration

TINKER_EXPERIMENTS = {
    'v1': {
        'note': 'baseline — overfit: train collapsed to 0.015, val rose to 0.56',
        'base_model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        'lora_rank': 16,
        'learning_rate': 1e-4,
        'batch_size': 8,
        'epochs': 3,
        'max_inference_tokens': 10,
        'temperature': 0.0,
    },
    'v2': {
        'note': 'fix overfitting via lr only — keep rank for more headroom for middle ranks',
        'base_model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        'lora_rank': 16,
        'learning_rate': 2e-5,
        'batch_size': 8,
        'epochs': 6,
        'max_inference_tokens': 10,
        'temperature': 0.0,
    },
    'v3': {
        'note': 'isolate rank effect — same lr as v2 but drop rank 16→8',
        'base_model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        'lora_rank': 8,
        'learning_rate': 2e-5,
        'batch_size': 8,
        'epochs': 3,
        'max_inference_tokens': 10,
        'temperature': 0.0,
    },
    'v4': {
        'note': 'smaller batch (8→4) for noisier gradients — may help middle-class separation',
        'base_model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        'lora_rank': 8,
        'learning_rate': 2e-5,
        'batch_size': 4,
        'epochs': 6,
        'max_inference_tokens': 10,
        'temperature': 0.0,
    },
}

ACTIVE_EXPERIMENT = 'v4'
TINKER_CONFIG = TINKER_EXPERIMENTS[ACTIVE_EXPERIMENT]

# Prompt template for classification — completion is " {label}"
PROMPT_TEMPLATE = (
    "Classify the truthfulness of the following political statement.\n\n"
    "Statement: {text}\n\n"
    "Choose exactly one label: barely-true, false, half-true, mostly-true, pants-fire, true\n\n"
    "Label:"
)

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
