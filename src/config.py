"""
Configuration and constants for Fake News Detection project.
Tinker API — Nemotron 30B + LoRA fine-tuning on LIAR2 dataset.
"""

from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
SRC_DIR      = PROJECT_ROOT / 'src'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Random seed
RANDOM_SEED = 42

# ── Labels ─────────────────────────────────────────────────────────────────────
# Alphabetical order — used consistently for encoding / decoding
NUM_LABELS   = 6
LABEL_NAMES  = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABEL_NAMES)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}

# ── Data ───────────────────────────────────────────────────────────────────────
# TSV column names (LIAR2 files have no header row)
TSV_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'job_title', 'state_info', 'party_affiliation',
    'barely_true_count', 'false_count', 'half_true_count',
    'mostly_true_count', 'pants_on_fire_count', 'context'
]

# Feature columns concatenated into model input text
FEATURE_COLS = ['statement', 'speaker', 'job_title', 'party_affiliation', 'subject', 'context']
LABEL_COL    = 'label'

# ── Tinker / LoRA ──────────────────────────────────────────────────────────────
TINKER_CONFIG = {
    'base_model':           'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    'lora_rank':            16,
    'learning_rate':        1e-5,
    'batch_size':           8,
    'epochs':               5,
    'early_stopping_patience': 2,
    'max_inference_tokens': 10,
    'temperature':          0.0,
}

# Prompt template — {text} is the full combined feature string
PROMPT_TEMPLATE = (
    "Classify the truthfulness of the following political statement "
    "and its context.\n\n"
    "{text}\n\n"
    "Choose exactly one label: barely-true, false, half-true, "
    "mostly-true, pants-fire, true\n\n"
    "Label:"
)

# ── Artifact paths ─────────────────────────────────────────────────────────────
PREPROCESSED_PATH       = ARTIFACTS_DIR / 'preprocessed.pkl'
TINKER_WEIGHTS_URI_PATH = ARTIFACTS_DIR / 'tinker_weights_uri.txt'
RESULTS_PATH            = ARTIFACTS_DIR / 'training_results.json'
LOSS_CURVE_PATH         = ARTIFACTS_DIR / 'loss_curve.png'
CONFUSION_MATRIX_PATH   = ARTIFACTS_DIR / 'confusion_matrix.png'

if __name__ == '__main__':
    print("Fake News Detection — Tinker Configuration")
    print(f"\nBase model: {TINKER_CONFIG['base_model']}")
    print(f"LoRA rank:  {TINKER_CONFIG['lora_rank']}")
    print(f"LR / Batch / Epochs: "
          f"{TINKER_CONFIG['learning_rate']} / "
          f"{TINKER_CONFIG['batch_size']} / "
          f"{TINKER_CONFIG['epochs']}")
    print(f"\nLabels: {LABEL_NAMES}")
    print(f"Data:   {DATA_DIR}")
