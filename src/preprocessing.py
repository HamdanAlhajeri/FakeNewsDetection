"""
Preprocessing script for Fake News Detection.

Run this ONCE to clean and combine all features. Outputs three TSV files
that train.py reads directly — no reprocessing needed on each training run.

Usage:
    python src/preprocessing.py

Output:
    artifacts/preprocessed_train.tsv
    artifacts/preprocessed_val.tsv
    artifacts/preprocessed_test.tsv

Each file has two tab-separated columns with a header row:
    text    — cleaned and combined feature string
    label   — truthfulness label string (e.g. 'false')
"""

import sys
import logging
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, ARTIFACTS_DIR
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PREPROCESSED_TRAIN = ARTIFACTS_DIR / 'preprocessed_train.tsv'
PREPROCESSED_VAL   = ARTIFACTS_DIR / 'preprocessed_val.tsv'
PREPROCESSED_TEST  = ARTIFACTS_DIR / 'preprocessed_test.tsv'


def _process_and_save(raw_path: Path, out_path: Path):
    """Load a raw TSV, apply full preprocessing, save as a two-column TSV."""
    df = DataProcessor.load_data(raw_path)
    df = df.dropna(subset=['label', 'statement']).copy()
    df['label'] = df['label'].str.strip().str.lower()

    rows = []
    for _, row in df.iterrows():
        text = DataProcessor.build_input_text(row)
        if text.strip():
            rows.append({'text': text, 'label': row['label']})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, sep='\t', index=False)
    logger.info(f"  {out_path.name}: {len(out)} rows")


def run_and_save():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Preprocessing data splits...")
    _process_and_save(DATA_DIR / 'train.tsv', PREPROCESSED_TRAIN)
    _process_and_save(DATA_DIR / 'valid.tsv', PREPROCESSED_VAL)
    _process_and_save(DATA_DIR / 'test.tsv',  PREPROCESSED_TEST)
    logger.info("Done.")


if __name__ == '__main__':
    run_and_save()
