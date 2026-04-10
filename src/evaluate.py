"""
Compare predictions in results.tsv against ground-truth labels in test.tsv.

Usage:
    python src/evaluate.py
    python src/evaluate.py --test src/test.tsv --results results.tsv
    python src/evaluate.py --test src/test.tsv --results results.tsv --out eval_report.txt
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from config import LABEL_NAMES, TSV_COLUMNS

PROJECT_ROOT = Path(__file__).parent.parent


def load_test(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep='\t', header=None, names=TSV_COLUMNS)
    return df['label'].str.strip().str.lower()


def load_results(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep='\t')
    if 'prediction' not in df.columns:
        raise ValueError(
            f"'prediction' column not found in {path.name}. "
            f"Available columns: {', '.join(df.columns)}"
        )
    return df['prediction'].str.strip().str.lower()


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> str:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Row count mismatch: test has {len(y_true)} rows, "
            f"results has {len(y_pred)} rows."
        )

    # Normalise any unrecognised predictions to 'unknown'
    known = set(LABEL_NAMES)
    y_pred_clean = y_pred.apply(lambda x: x if x in known else 'unknown')

    acc         = accuracy_score(y_true, y_pred_clean)
    macro_f1    = f1_score(y_true, y_pred_clean, average='macro',    labels=LABEL_NAMES, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_clean, average='weighted', labels=LABEL_NAMES, zero_division=0)
    report      = classification_report(
        y_true, y_pred_clean, labels=LABEL_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred_clean, labels=LABEL_NAMES)

    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"  Samples:      {len(y_true)}",
        f"  Accuracy:     {acc:.4f}  ({acc*100:.2f}%)",
        f"  Macro F1:     {macro_f1:.4f}",
        f"  Weighted F1:  {weighted_f1:.4f}",
        "",
        "Per-class report:",
        "-" * 60,
        report,
        "Confusion matrix  (rows=true, cols=predicted):",
        "-" * 60,
        _format_cm(cm, LABEL_NAMES),
    ]

    # Warn about unrecognised predictions
    unknown_mask = ~y_pred.isin(known)
    if unknown_mask.any():
        n = unknown_mask.sum()
        examples = y_pred[unknown_mask].unique()[:5].tolist()
        lines += [
            "",
            f"WARNING: {n} prediction(s) not in label set — treated as 'unknown'.",
            f"  Examples: {examples}",
        ]

    return "\n".join(lines)


def _format_cm(cm, labels) -> str:
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    rows = [header]
    for label, row in zip(labels, cm):
        row_str = f"{label:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        rows.append(row_str)
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fake news predictions against ground truth')
    parser.add_argument('--test',    default='src/test.tsv',   help='Ground-truth test TSV (default: src/test.tsv)')
    parser.add_argument('--results', default='results.tsv',    help='Predictions TSV from predict.py (default: results.tsv)')
    parser.add_argument('--out',     default=None,             help='Optional file to save the report')
    args = parser.parse_args()

    test_path    = Path(args.test)
    results_path = Path(args.results)

    if not test_path.exists():
        print(f"Error: test file not found: {test_path}")
        sys.exit(1)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    y_true = load_test(test_path)
    y_pred = load_results(results_path)

    report = evaluate(y_true, y_pred)
    print(report)

    if args.out:
        Path(args.out).write_text(report)
        print(f"\nReport saved to {args.out}")


if __name__ == '__main__':
    main()
