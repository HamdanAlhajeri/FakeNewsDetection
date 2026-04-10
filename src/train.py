"""
Training entry point for Fake News Detection — Tinker API.

Run preprocessing first (once):
    python src/preprocessing.py

Then train (as many times as needed):
    python src/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from config import (
    ARTIFACTS_DIR, TINKER_CONFIG, PROMPT_TEMPLATE,
    TINKER_WEIGHTS_URI_PATH, LABEL_TO_IDX
)

RUNS_DIR      = ARTIFACTS_DIR / 'runs'
RUNS_REGISTRY = ARTIFACTS_DIR / 'runs_registry.json'
from preprocessing import PREPROCESSED_TRAIN, PREPROCESSED_VAL, PREPROCESSED_TEST
from data_processor import DataProcessor
from models import TinkerClassifier
from training_utils import (
    TinkerTrainer, MetricsCalculator,
    plot_curves, plot_confusion_matrix, save_results
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_split(path: Path):
    """Load a preprocessed TSV and return (texts, label_strings, label_indices)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}\n"
            "Run preprocessing first:  python src/preprocessing.py"
        )
    df = pd.read_csv(path, sep='\t')
    texts      = df['text'].tolist()
    label_strs = df['label'].tolist()
    label_idx  = [LABEL_TO_IDX[l] for l in label_strs]
    return texts, label_strs, label_idx


def save_run(run_id: str, uri: str, config: dict, metrics: dict, history: dict):
    """Save this run's URI and results to its own file and append to the registry."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run = {
        'run_id':    run_id,
        'timestamp': datetime.now().isoformat(),
        'uri':       uri,
        'config':    config,
        'metrics': {
            'accuracy':    round(metrics['accuracy'],    4),
            'macro_f1':    round(metrics['macro_f1'],    4),
            'weighted_f1': round(metrics['weighted_f1'], 4),
        },
        'history': history,
    }

    # Per-run file
    run_file = RUNS_DIR / f'{run_id}.json'
    run_file.write_text(json.dumps(run, indent=2))

    # Registry — list of all runs, latest first
    registry = []
    if RUNS_REGISTRY.exists():
        registry = json.loads(RUNS_REGISTRY.read_text())
    registry.insert(0, {
        'run_id':    run['run_id'],
        'timestamp': run['timestamp'],
        'uri':       run['uri'],
        'accuracy':  run['metrics']['accuracy'],
        'macro_f1':  run['metrics']['macro_f1'],
    })
    RUNS_REGISTRY.write_text(json.dumps(registry, indent=2))

    # Keep tinker_weights_uri.txt pointing to the latest run
    TINKER_WEIGHTS_URI_PATH.write_text(uri)

    print(f"Run saved → {run_file}")
    print(f"Registry  → {RUNS_REGISTRY}")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    # ── 1. Load preprocessed TSVs ──────────────────────────────────────────────
    logger.info("Loading preprocessed splits...")
    train_texts, train_label_strs, _           = load_split(PREPROCESSED_TRAIN)
    val_texts,   val_label_strs,   _           = load_split(PREPROCESSED_VAL)
    test_texts,  _,                test_labels = load_split(PREPROCESSED_TEST)
    logger.info(f"train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # ── 2. Connect to Tinker ───────────────────────────────────────────────────
    print("Connecting to Tinker...")
    classifier = TinkerClassifier()
    classifier.connect()
    classifier.create_training_client(lora_rank=TINKER_CONFIG['lora_rank'])
    tokenizer = classifier.get_tokenizer()

    # ── 3. Build Tinker Datums (tokenization only) ─────────────────────────────
    print("Building Tinker Datums...")
    class_weights = DataProcessor.compute_class_weights(train_label_strs)
    logger.info(f"Class weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    train_datums = DataProcessor.prepare_tinker_dataset(
        train_texts, train_label_strs, tokenizer, PROMPT_TEMPLATE,
        class_weights=class_weights
    )
    val_datums = DataProcessor.prepare_tinker_dataset(
        val_texts, val_label_strs, tokenizer, PROMPT_TEMPLATE
    )

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer = TinkerTrainer(
        classifier=classifier,
        learning_rate=TINKER_CONFIG['learning_rate'],
        batch_size=TINKER_CONFIG['batch_size'],
    )
    history = trainer.train(
        train_datums, val_datums,
        epochs=TINKER_CONFIG['epochs'],
        patience=TINKER_CONFIG['early_stopping_patience'],
    )

    # ── 5. Save weights ────────────────────────────────────────────────────────
    print("\nSaving weights...")
    uri = classifier.save_for_inference(run_id)
    print(f"Weights URI: {uri}")

    # ── 6. Evaluate on test set ────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_preds_str = classifier.predict_batch(test_texts)
    test_preds_idx = [LABEL_TO_IDX.get(p, 0) for p in test_preds_str]
    test_metrics   = MetricsCalculator.calculate(test_labels, test_preds_idx)

    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['classification_report']}")

    # ── 7. Save artifacts ──────────────────────────────────────────────────────
    plot_curves(history)
    plot_confusion_matrix(test_labels, test_preds_idx)
    save_results(history, test_metrics)
    save_run(run_id, uri, TINKER_CONFIG, test_metrics, history)

    print(f"\nDone. Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == '__main__':
    main()
