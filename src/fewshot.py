"""
Few-shot baseline for Fake News Detection.

Uses the base Qwen3-8B model (no LoRA) with K labelled examples per class
prepended to the prompt. No training — pure in-context learning.

The few-shot prompt format:

    Here are some examples of truthfulness classification:

    Statement: ...  Speaker: ...  ...
    Label: false

    Statement: ...  Speaker: ...  ...
    Label: mostly-true

    ... (K examples per class = K*6 total)

    Now classify this statement:
    Statement: ...
    Choose exactly one label: barely-true, false, ...
    Label:

Usage:
    python src/fewshot.py              # 2 examples per class (12 total)
    python src/fewshot.py --shots 1    # 1 example per class (6 total)
    python src/fewshot.py --shots 3    # 3 examples per class (18 total)
    python src/fewshot.py --uri tinker://...   # reuse an existing Tinker URI

Output:
    artifacts/fewshot_results.json
    artifacts/fewshot_report.txt
"""

import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import (
    ARTIFACTS_DIR, LABEL_NAMES, LABEL_TO_IDX, TINKER_CONFIG
)
from training_utils import MetricsCalculator

PREPROCESSED_TRAIN = ARTIFACTS_DIR / 'preprocessed_train.tsv'
PREPROCESSED_TEST  = ARTIFACTS_DIR / 'preprocessed_test.tsv'
RUNS_DIR           = ARTIFACTS_DIR / 'runs'
FEWSHOT_RESULTS    = ARTIFACTS_DIR / 'fewshot_results.json'
FEWSHOT_REPORT     = ARTIFACTS_DIR / 'fewshot_report.txt'

RANDOM_SEED = 42


# ── prompt building ───────────────────────────────────────────────────────────

FEWSHOT_HEADER = (
    "Here are some examples of political statement truthfulness classification:\n\n"
)

FEWSHOT_EXAMPLE_TEMPLATE = (
    "{text}\n"
    "Label: {label}\n\n"
)

FEWSHOT_QUERY_TEMPLATE = (
    "<|im_start|>user\n"
    "{header}"
    "{examples}"
    "Now classify this statement:\n\n"
    "{text}\n\n"
    "Choose exactly one label: barely-true, false, half-true, "
    "mostly-true, pants-fire, true\n\n"
    "Label:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def select_examples(train_df: pd.DataFrame, shots: int, seed: int = RANDOM_SEED) -> dict[str, list[str]]:
    """
    Select `shots` examples per class from the training set.
    Returns a dict mapping label → list of text strings.
    """
    rng = random.Random(seed)
    examples: dict[str, list[str]] = {}
    for label in LABEL_NAMES:
        pool = train_df[train_df['label'] == label]['text'].tolist()
        if len(pool) < shots:
            raise ValueError(
                f"Label '{label}' has only {len(pool)} training examples "
                f"but --shots={shots} was requested."
            )
        examples[label] = rng.sample(pool, shots)
    return examples


def build_fewshot_prompt(query_text: str, examples: dict[str, list[str]], shots: int) -> str:
    """
    Build the full few-shot prompt for one test sample.
    Examples are interleaved across classes so no single class appears in a block.
    """
    # Interleave: [label0_ex0, label1_ex0, ..., label0_ex1, label1_ex1, ...]
    interleaved = []
    for shot_idx in range(shots):
        for label in LABEL_NAMES:
            interleaved.append((label, examples[label][shot_idx]))

    example_strs = "".join(
        FEWSHOT_EXAMPLE_TEMPLATE.format(text=text, label=label)
        for label, text in interleaved
    )

    return FEWSHOT_QUERY_TEMPLATE.format(
        header=FEWSHOT_HEADER,
        examples=example_strs,
        text=query_text,
    )


# ── Tinker connection ─────────────────────────────────────────────────────────

def connect_tinker(uri: str | None) -> tuple:
    """
    Return (sampling_client, tokenizer).
    If uri is None, creates an untrained LoRA session (base model, no gradient steps).
    """
    from tinker import ServiceClient
    from transformers import AutoTokenizer

    svc = ServiceClient()

    if uri is None:
        print("  Creating untrained LoRA session (base model, no gradient steps)...")
        training_client = svc.create_lora_training_client(
            base_model=TINKER_CONFIG['base_model'],
            rank=TINKER_CONFIG['lora_rank'],
        )
        uri = training_client.save_weights_for_sampler('fewshot-baseline').result().path
        print(f"  URI: {uri}")
    else:
        print(f"  Reusing URI: {uri}")

    sampling_client = svc.create_sampling_client(
        model_path=uri,
        base_model=TINKER_CONFIG['base_model'],
    )
    tokenizer = AutoTokenizer.from_pretrained(TINKER_CONFIG['base_model'])
    return sampling_client, tokenizer, uri


# ── scoring ───────────────────────────────────────────────────────────────────

def score_labels(prompt: str, sampling_client, tokenizer) -> str:
    """Score all 6 labels via logprobs and return the argmax."""
    from tinker import types

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_input  = types.ModelInput.from_ints(tokens=prompt_tokens)

    logprob_dict = {}
    for label in LABEL_NAMES:
        completion_tokens = tokenizer.encode(f" {label}", add_special_tokens=False)
        full_input = types.ModelInput.from_ints(
            tokens=prompt_input.to_ints() + completion_tokens
        )
        logprobs = sampling_client.compute_logprobs(full_input).result()
        logprob_dict[label] = float(sum(
            lp for lp in logprobs[-len(completion_tokens):] if lp is not None
        ))
    return max(logprob_dict, key=logprob_dict.get)


def predict_all(test_texts: list[str], examples: dict[str, list[str]],
                shots: int, sampling_client, tokenizer) -> list[str]:
    preds = []
    total = len(test_texts)
    for i, text in enumerate(test_texts):
        prompt = build_fewshot_prompt(text, examples, shots)
        preds.append(score_labels(prompt, sampling_client, tokenizer))
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} scored")
    return preds


# ── metrics & reporting ───────────────────────────────────────────────────────

def compute_metrics(name: str, pred_labels: list[str], true_labels: list[str]) -> dict:
    pred_idx = [LABEL_TO_IDX.get(p, 0) for p in pred_labels]
    true_idx = [LABEL_TO_IDX.get(t, 0) for t in true_labels]
    m = MetricsCalculator.calculate(true_idx, pred_idx)
    return {
        'name':        name,
        'accuracy':    round(m['accuracy'],    4),
        'macro_f1':    round(m['macro_f1'],    4),
        'weighted_f1': round(m['weighted_f1'], 4),
        'report':      m['classification_report'],
    }


def load_best_finetuned() -> dict | None:
    if not RUNS_DIR.exists():
        return None
    runs = [json.loads(p.read_text()) for p in RUNS_DIR.glob('run_*.json')]
    runs = [r for r in runs if r.get('metrics')]
    if not runs:
        return None
    best = max(runs, key=lambda r: r['metrics']['accuracy'])
    return {
        'name':        f"Fine-tuned LoRA ({best['run_id']})",
        'accuracy':    best['metrics']['accuracy'],
        'macro_f1':    best['metrics']['macro_f1'],
        'weighted_f1': best['metrics']['weighted_f1'],
        'report':      None,
    }


def print_and_build_report(results: list[dict], shots: int, n: int) -> str:
    col_w = 40
    lines = []

    header_line = f"FEW-SHOT BASELINE — {shots} example(s) per class ({shots*6} total)"
    lines.append(header_line)
    lines.append(f"Test samples: {n}   |   Model: {TINKER_CONFIG['base_model']}")
    lines.append("")
    lines.append("=" * 75)
    lines.append(f"{'Model':<{col_w}} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10}")
    lines.append("-" * 75)
    for r in results:
        acc = f"{r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)"
        lines.append(
            f"{r['name']:<{col_w}} {acc:>17} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f}"
        )
    lines.append("=" * 75)
    lines.append("  Random chance (uniform): ~16.7%")
    lines.append("")
    lines.append("PER-CLASS BREAKDOWN")
    lines.append("")
    for r in results:
        if r.get('report'):
            lines.append(r['name'])
            lines.append("-" * 40)
            lines.append(r['report'])
            lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Few-shot baseline — Fake News Detection')
    parser.add_argument('--shots', type=int, default=2,
                        help='Number of examples per class (default: 2, total = shots*6)')
    parser.add_argument('--uri', type=str, default=None,
                        help='Reuse an existing Tinker URI instead of creating a new session')
    args = parser.parse_args()

    if args.shots < 1:
        print("Error: --shots must be at least 1")
        sys.exit(1)

    # ── load data ─────────────────────────────────────────────────────────────
    for path in (PREPROCESSED_TRAIN, PREPROCESSED_TEST):
        if not path.exists():
            print(f"Error: {path} not found. Run: python src/preprocessing.py")
            sys.exit(1)

    print("Loading data...")
    train_df   = pd.read_csv(PREPROCESSED_TRAIN, sep='\t')
    test_df    = pd.read_csv(PREPROCESSED_TEST,  sep='\t')
    test_texts  = test_df['text'].tolist()
    test_labels = test_df['label'].str.strip().str.lower().tolist()
    n = len(test_labels)
    print(f"  Train: {len(train_df)} rows  |  Test: {n} rows")

    # ── select examples ───────────────────────────────────────────────────────
    print(f"\nSelecting {args.shots} example(s) per class ({args.shots*6} total)...")
    examples = select_examples(train_df, shots=args.shots)
    for label, exs in examples.items():
        preview = exs[0][:80].replace('\n', ' ')
        print(f"  {label:<14}: \"{preview}...\"")

    # ── connect to Tinker ─────────────────────────────────────────────────────
    print("\nConnecting to Tinker...")
    sampling_client, tokenizer, uri = connect_tinker(args.uri)
    print("Connected.\n")

    # ── run few-shot predictions ──────────────────────────────────────────────
    print(f"Scoring {n} test samples ({args.shots} shot(s) per class)...")
    preds = predict_all(test_texts, examples, args.shots, sampling_client, tokenizer)

    # ── compute metrics ───────────────────────────────────────────────────────
    results = [compute_metrics(
        f"Few-Shot ({args.shots}-shot, {args.shots*6} examples, base Qwen3-8B)",
        preds, test_labels
    )]

    best = load_best_finetuned()
    if best:
        results.append(best)

    # ── display & save ────────────────────────────────────────────────────────
    report_text = print_and_build_report(results, args.shots, n)

    FEWSHOT_REPORT.write_text(report_text)
    print(f"\nFull report saved to {FEWSHOT_REPORT}")

    save_data = [
        {
            **{k: v for k, v in r.items() if k != 'report'},
            'shots': args.shots,
            'total_examples': args.shots * 6,
            'tinker_uri': uri,
        }
        for r in results
    ]
    FEWSHOT_RESULTS.write_text(json.dumps(save_data, indent=2))
    print(f"Metrics saved to {FEWSHOT_RESULTS}")


if __name__ == '__main__':
    main()
