"""
Training entry point for Fake News Detection — Tinker API.

Run preprocessing first (once):
    python src/preprocessing.py

Then train (as many times as needed):
    python src/train.py

To evaluate an existing run without retraining:
    python src/train.py --eval-only
    python src/train.py --eval-only --run run_20240101_120000
    python src/train.py --eval-only --uri tinker://...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fake News Detection — Tinker training & evaluation")
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Skip training and evaluate an existing saved run'
    )
    parser.add_argument(
        '--uri', type=str, default=None,
        help='Explicit Tinker weights URI to evaluate (overrides --run and latest run)'
    )
    parser.add_argument(
        '--run', type=str, default=None,
        help='Run ID to evaluate, e.g. run_20240101_120000'
    )
    return parser.parse_args()


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


def resolve_uri(args) -> str:
    """Resolve the weights URI from CLI args or saved run registry."""
    if args.uri:
        return args.uri
    if args.run:
        run_file = RUNS_DIR / f'{args.run}.json'
        if not run_file.exists():
            raise FileNotFoundError(
                f"Run '{args.run}' not found in {RUNS_DIR}.\n"
                "Check artifacts/runs/ for available run IDs."
            )
        return json.loads(run_file.read_text())['uri']
    if TINKER_WEIGHTS_URI_PATH.exists():
        uri = TINKER_WEIGHTS_URI_PATH.read_text().strip()
        logger.info(f"Using latest saved URI from {TINKER_WEIGHTS_URI_PATH}")
        return uri
    raise FileNotFoundError(
        "No saved weights found. Run training first or pass --uri / --run."
    )


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


def run_evaluation(classifier: TinkerClassifier, test_texts, test_labels,
                   uri: str, run_id: str, history: dict = None):
    """Evaluate classifier on test set and save all artifacts."""
    print("\nEvaluating on test set...")
    test_preds_str = classifier.predict_batch(test_texts)
    test_preds_idx = [LABEL_TO_IDX.get(p, 0) for p in test_preds_str]
    test_metrics   = MetricsCalculator.calculate(test_labels, test_preds_idx)

    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['classification_report']}")

    # Only plot curves if we have training history
    if history:
        plot_curves(history)
    plot_confusion_matrix(test_labels, test_preds_idx)
    save_results(history or {}, test_metrics)
    save_run(run_id, uri, TINKER_CONFIG, test_metrics, history or {})

    print(f"\nDone. Artifacts saved to: {ARTIFACTS_DIR}")
    return test_metrics


def main():
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test split (always needed) ───────────────────────────────────────
    logger.info("Loading preprocessed splits...")
    test_texts, _, test_labels = load_split(PREPROCESSED_TEST)

    # ══════════════════════════════════════════════════════════════════════════
    # EVAL-ONLY PATH — skip training, load existing weights
    # ══════════════════════════════════════════════════════════════════════════
    if args.eval_only:
        uri    = resolve_uri(args)
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Eval-only mode. Loading weights from: {uri}")

        classifier = TinkerClassifier()
        classifier.connect()
        classifier.load_sampling_client(uri)

        run_evaluation(classifier, test_texts, test_labels, uri, run_id, history=None)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # FULL TRAINING PATH
    # ══════════════════════════════════════════════════════════════════════════
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    # ── 1. Load all splits ─────────────────────────────────────────────────────
    train_texts, train_label_strs, _ = load_split(PREPROCESSED_TRAIN)
    val_texts,   val_label_strs,   _ = load_split(PREPROCESSED_VAL)
    logger.info(f"train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # ── 2. Connect to Tinker ───────────────────────────────────────────────────
    print("Connecting to Tinker...")
    classifier = TinkerClassifier()
    classifier.connect()
    classifier.create_training_client(lora_rank=TINKER_CONFIG['lora_rank'])
    tokenizer = classifier.get_tokenizer()

    # ── 3. Build Tinker Datums ─────────────────────────────────────────────────
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

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    run_evaluation(classifier, test_texts, test_labels, uri, run_id, history)


if __name__ == '__main__':
    main()