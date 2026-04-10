"""
Inference script for Fake News Detection — Tinker API.

Usage:
    python src/predict.py --text "The unemployment rate is 3 percent."
    python src/predict.py --text "..." --proba
    python src/predict.py --file statements.csv --col statement
    python src/predict.py --file statements.tsv --col text --out results.tsv
    python src/predict.py --run run_20260407_143022 --text "..."
    python src/predict.py --list-runs
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import ARTIFACTS_DIR, TINKER_WEIGHTS_URI_PATH

RUNS_DIR      = ARTIFACTS_DIR / 'runs'
RUNS_REGISTRY = ARTIFACTS_DIR / 'runs_registry.json'
from models import TinkerClassifier


class TinkerPredictor:
    """
    Load a trained Tinker checkpoint and run inference.

    Usage:
        predictor = TinkerPredictor()                    # reads URI from artifacts/
        predictor = TinkerPredictor(uri="tinker://...")  # explicit URI
        label = predictor.predict("Some political statement")
        label, logprobs = predictor.predict("...", return_proba=True)
    """

    def __init__(self, uri: str = None, run_id: str = None):
        if uri is None:
            if run_id is not None:
                # Load URI from a specific named run
                run_file = RUNS_DIR / f'{run_id}.json'
                if not run_file.exists():
                    raise FileNotFoundError(
                        f"Run '{run_id}' not found. "
                        "Use --list-runs to see available runs."
                    )
                uri = json.loads(run_file.read_text())['uri']
            elif TINKER_WEIGHTS_URI_PATH.exists():
                # Default: load the latest run
                uri = TINKER_WEIGHTS_URI_PATH.read_text().strip()
            else:
                raise FileNotFoundError(
                    "No saved runs found. "
                    "Run `python src/train.py` first, or pass --uri / --run."
                )

        self.classifier = TinkerClassifier()
        self.classifier.connect()
        self.classifier.load_sampling_client(uri)
        print(f"Loaded model from URI: {uri}")

    def predict(self, text: str, return_proba: bool = False):
        return self.classifier.predict(text, return_proba=return_proba)

    def predict_batch(self, texts: list) -> list:
        return self.classifier.predict_batch(texts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_tabular(path: Path, col: str) -> tuple:
    """
    Load a CSV or TSV file and return (DataFrame, list of texts from col).
    Separator is inferred from the file extension.
    """
    sep = '\t' if path.suffix.lower() == '.tsv' else ','
    df  = pd.read_csv(path, sep=sep, header=None if col.isdigit() else 'infer')
    if col.isdigit():
        col_key = int(col)
        if col_key >= len(df.columns):
            raise ValueError(
                f"Column index {col_key} out of range. "
                f"File has {len(df.columns)} columns (0–{len(df.columns)-1})."
            )
    else:
        if col not in df.columns:
            available = ', '.join(df.columns.astype(str).tolist())
            raise ValueError(
                f"Column '{col}' not found in {path.name}. "
                f"Available columns: {available}\n"
                f"Use --col to specify the correct column name or a numeric index."
            )
        col_key = col
    texts = df[col_key].fillna('').astype(str).tolist()
    return df, texts


def main():
    parser = argparse.ArgumentParser(description='Fake news label prediction (Tinker)')
    parser.add_argument('--uri',        default=None,
                        help='Explicit Tinker weights URI')
    parser.add_argument('--run',        default=None,
                        help='Run ID to load (e.g. run_20260407_143022). Defaults to latest.')
    parser.add_argument('--list-runs',  action='store_true',
                        help='List all saved runs and exit')
    parser.add_argument('--text',       default=None,
                        help='Single statement to classify')
    parser.add_argument('--file',       default=None,
                        help='CSV or TSV file to classify (separator inferred from extension)')
    parser.add_argument('--col',        default='text',
                        help='Column containing the statements (default: text)')
    parser.add_argument('--out',        default=None,
                        help='Output file to save results (CSV or TSV, inferred from extension)')
    parser.add_argument('--proba',      action='store_true',
                        help='Print per-label log-probabilities')
    args = parser.parse_args()

    if args.list_runs:
        if not RUNS_REGISTRY.exists():
            print("No runs found. Train a model first: python src/train.py")
        else:
            runs = json.loads(RUNS_REGISTRY.read_text())
            print(f"\n{'Run ID':<30} {'Timestamp':<22} {'Accuracy':>9} {'Macro F1':>9}")
            print('-' * 75)
            for r in runs:
                print(f"{r['run_id']:<30} {r['timestamp'][:19]:<22} "
                      f"{r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}")
            print(f"\nLatest: {runs[0]['run_id']}")
        return

    predictor = TinkerPredictor(uri=args.uri, run_id=args.run)

    if args.text:
        if args.proba:
            label, logprobs = predictor.predict(args.text, return_proba=True)
            print(f"\nText:       {args.text}")
            print(f"Prediction: {label}")
            print("\nPer-label log-probabilities:")
            for lbl, lp in sorted(logprobs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {lbl:<14s} {lp:.4f}")
        else:
            label = predictor.predict(args.text)
            print(f"\nText:       {args.text}")
            print(f"Prediction: {label}")

    elif args.file:
        path = Path(args.file)
        df, texts = _load_tabular(path, args.col)
        print(f"\nPredicting {len(texts)} rows from {path.name} (column: '{args.col}')...")

        predictions = predictor.predict_batch(texts)
        df['prediction'] = predictions

        # Print to console
        print(f"\n{'#':<5} {'Text':<70}  Prediction")
        print('-' * 90)
        for i, (text, pred) in enumerate(zip(texts, predictions), 1):
            print(f"{i:<5} {str(text)[:70]:<70}  {pred}")

        # Optionally save to file
        if args.out:
            out_path = Path(args.out)
            sep = '\t' if out_path.suffix.lower() == '.tsv' else ','
            df.to_csv(out_path, sep=sep, index=False)
            print(f"\nResults saved to {out_path}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
