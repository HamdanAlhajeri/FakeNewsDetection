"""
Baseline models for Fake News Detection.

Three statistical baselines that require no training or Tinker connection,
plus an optional zero-shot baseline using the raw Qwen3-8B base model
(same prompting as the fine-tuned model but without LoRA adapters).

Usage:
    # Statistical baselines only (fast, no GPU/API needed)
    python src/baseline.py

    # Include zero-shot Qwen3-8B via Tinker (slow, requires API key)
    python src/baseline.py --zero-shot

Output:
    Console table comparing all baselines against the best fine-tuned run.
    artifacts/baseline_results.json
"""

import sys
import json
import argparse
import logging
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config import (
    ARTIFACTS_DIR, LABEL_NAMES, LABEL_TO_IDX, TINKER_CONFIG, PROMPT_TEMPLATE
)
from training_utils import MetricsCalculator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RUNS_DIR            = ARTIFACTS_DIR / 'runs'
PREPROCESSED_TRAIN  = ARTIFACTS_DIR / 'preprocessed_train.tsv'
PREPROCESSED_TEST   = ARTIFACTS_DIR / 'preprocessed_test.tsv'
BASELINE_RESULTS    = ARTIFACTS_DIR / 'baseline_results.json'
BASELINE_REPORT     = ARTIFACTS_DIR / 'baseline_report.txt'

RANDOM_SEED = 42


# ── data loading ──────────────────────────────────────────────────────────────

def load_labels(path: Path) -> list[str]:
    df = pd.read_csv(path, sep='\t')
    return df['label'].str.strip().str.lower().tolist()


def load_texts_and_labels(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, sep='\t')
    texts  = df['text'].tolist()
    labels = df['label'].str.strip().str.lower().tolist()
    return texts, labels


# ── statistical baselines ─────────────────────────────────────────────────────

class MajorityClassBaseline:
    """Always predicts the single most frequent class in the training set."""

    def __init__(self):
        self.majority_label: str = None

    def fit(self, train_labels: list[str]):
        counts = Counter(train_labels)
        self.majority_label = counts.most_common(1)[0][0]

    def predict(self, n: int) -> list[str]:
        return [self.majority_label] * n

    def __str__(self):
        return f"MajorityClass(always predicts '{self.majority_label}')"


class StratifiedRandomBaseline:
    """Samples predictions from the training label distribution."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed   = seed
        self.labels: list[str] = []
        self.probs:  list[float] = []

    def fit(self, train_labels: list[str]):
        counts = Counter(train_labels)
        total  = sum(counts.values())
        self.labels = list(counts.keys())
        self.probs  = [counts[l] / total for l in self.labels]

    def predict(self, n: int) -> list[str]:
        rng = np.random.default_rng(self.seed)
        return list(rng.choice(self.labels, size=n, p=self.probs))

    def __str__(self):
        dist = ', '.join(f'{l}={p:.1%}' for l, p in zip(self.labels, self.probs))
        return f"StratifiedRandom({dist})"


class UniformRandomBaseline:
    """Samples predictions uniformly across all 6 labels."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def predict(self, n: int) -> list[str]:
        rng = np.random.default_rng(self.seed)
        return list(rng.choice(LABEL_NAMES, size=n))

    def __str__(self):
        return f"UniformRandom(1/6 per class ≈ 16.7%)"


# ── zero-shot baseline ────────────────────────────────────────────────────────

class ZeroShotBaseline:
    """
    Base Qwen3-8B with NO LoRA adapters — same prompt as fine-tuned model,
    same logprob-based scoring. Requires Tinker API key.
    """

    def __init__(self):
        self.sampling_client = None
        self.tokenizer       = None

    def connect(self):
        from tinker import ServiceClient
        from transformers import AutoTokenizer

        print("Connecting to Tinker (zero-shot, no LoRA)...")
        svc = ServiceClient()

        # Tinker requires a tinker:// URI for sampling — there is no "raw base model"
        # sampling path. We create a LoRA training client, save immediately without
        # any training steps, then load that URI. This gives the base model weights
        # with rank-32 adapters initialised at zero — equivalent to zero-shot.
        training_client = svc.create_lora_training_client(
            base_model=TINKER_CONFIG['base_model'],
            rank=TINKER_CONFIG['lora_rank'],
        )
        print("  Saving untrained weights (no gradient steps)...")
        uri = training_client.save_weights_for_sampler('zero-shot-baseline').result().path
        print(f"  URI: {uri}")

        self.sampling_client = svc.create_sampling_client(
            model_path=uri,
            base_model=TINKER_CONFIG['base_model'],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TINKER_CONFIG['base_model'])
        print("Connected.")

    def _score_labels(self, text: str) -> str:
        from tinker import types

        prompt        = PROMPT_TEMPLATE.format(text=text)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_input  = types.ModelInput.from_ints(tokens=prompt_tokens)

        logprob_dict = {}
        for label in LABEL_NAMES:
            completion_tokens = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            full_input        = types.ModelInput.from_ints(
                tokens=prompt_input.to_ints() + completion_tokens
            )
            logprobs = self.sampling_client.compute_logprobs(full_input).result()
            logprob_dict[label] = float(sum(
                lp for lp in logprobs[-len(completion_tokens):] if lp is not None
            ))
        return max(logprob_dict, key=logprob_dict.get)

    def predict(self, texts: list[str]) -> list[str]:
        preds = []
        for i, text in enumerate(texts):
            pred = self._score_labels(text)
            preds.append(pred)
            if (i + 1) % 50 == 0:
                print(f"  Zero-shot: {i+1}/{len(texts)}")
        return preds

    def __str__(self):
        return f"ZeroShot(Qwen3-8B base, no LoRA, same prompt)"


# ── metrics helper ────────────────────────────────────────────────────────────

def evaluate(name: str, pred_labels: list[str], true_labels: list[str]) -> dict:
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


# ── load best fine-tuned run for comparison ───────────────────────────────────

def load_best_finetuned() -> dict | None:
    if not RUNS_DIR.exists():
        return None
    runs = []
    for p in RUNS_DIR.glob('run_*.json'):
        data = json.loads(p.read_text())
        if data.get('metrics'):
            runs.append(data)
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


# ── display ───────────────────────────────────────────────────────────────────

def print_comparison(results: list[dict]):
    col_w = 38
    print()
    print("=" * 75)
    print("BASELINE COMPARISON")
    print("=" * 75)
    header = f"{'Model':<{col_w}} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10}"
    print(header)
    print("-" * 75)

    for r in results:
        acc  = f"{r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)"
        mf1  = f"{r['macro_f1']:.4f}"
        wf1  = f"{r['weighted_f1']:.4f}"
        print(f"{r['name']:<{col_w}} {acc:>17} {mf1:>10} {wf1:>10}")

    print("=" * 75)
    print(f"  Random chance (uniform): ~16.7%")
    print()

    if len(results) >= 2:
        last   = results[-1]
        first  = results[0]
        gain   = last['accuracy'] - first['accuracy']
        print(f"  Gain of fine-tuned model over majority baseline: "
              f"+{gain*100:.1f} pp accuracy")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection — Baseline Evaluation')
    parser.add_argument('--zero-shot', action='store_true',
                        help='Include zero-shot Qwen3-8B baseline (requires Tinker API key, very slow)')
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    if not PREPROCESSED_TRAIN.exists() or not PREPROCESSED_TEST.exists():
        print("Error: preprocessed TSVs not found.")
        print("Run:  python src/preprocessing.py")
        sys.exit(1)

    print("Loading data...")
    train_labels          = load_labels(PREPROCESSED_TRAIN)
    test_texts, test_labels = load_texts_and_labels(PREPROCESSED_TEST)
    n = len(test_labels)
    print(f"  Train labels: {len(train_labels)}  |  Test samples: {n}")

    train_dist = Counter(train_labels)
    print("\nTraining label distribution:")
    for label in LABEL_NAMES:
        count = train_dist.get(label, 0)
        pct   = count / len(train_labels) * 100
        bar   = '#' * int(pct / 2)
        print(f"  {label:<14} {count:>5}  ({pct:5.1f}%)  {bar}")

    results = []

    # ── majority class ────────────────────────────────────────────────────────
    print("\nRunning majority-class baseline...")
    mc = MajorityClassBaseline()
    mc.fit(train_labels)
    results.append(evaluate("Majority Class", mc.predict(n), test_labels))
    print(f"  {mc}")

    # ── uniform random ────────────────────────────────────────────────────────
    print("Running uniform-random baseline...")
    ur = UniformRandomBaseline()
    results.append(evaluate("Uniform Random (1/6 per class)", ur.predict(n), test_labels))

    # ── stratified random ─────────────────────────────────────────────────────
    print("Running stratified-random baseline...")
    sr = StratifiedRandomBaseline()
    sr.fit(train_labels)
    results.append(evaluate("Stratified Random", sr.predict(n), test_labels))

    # ── zero-shot Qwen3-8B ────────────────────────────────────────────────────
    if args.zero_shot:
        print("\nRunning zero-shot Qwen3-8B baseline (this will take a long time)...")
        zs = ZeroShotBaseline()
        zs.connect()
        zs_preds = zs.predict(test_texts)
        results.append(evaluate("Zero-Shot Qwen3-8B (no LoRA)", zs_preds, test_labels))

    # ── fine-tuned comparison ─────────────────────────────────────────────────
    best = load_best_finetuned()
    if best:
        results.append(best)

    # ── display & save ────────────────────────────────────────────────────────
    print_comparison(results)

    print("Per-class breakdown:\n")
    for r in results:
        if r.get('report'):
            print(f"  {r['name']}")
            print(f"  {'-' * 40}")
            for line in r['report'].splitlines():
                print(f"  {line}")
            print()

    # JSON — metrics only (no report string)
    save_data = [
        {k: v for k, v in r.items() if k != 'report'}
        for r in results
    ]
    BASELINE_RESULTS.write_text(json.dumps(save_data, indent=2))

    # Plain-text report — full output including per-class breakdown
    lines = []
    lines.append("BASELINE COMPARISON REPORT")
    lines.append(f"Test samples: {n}")
    lines.append("")

    col_w = 38
    lines.append("=" * 75)
    lines.append(f"{'Model':<{col_w}} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10}")
    lines.append("-" * 75)
    for r in results:
        acc = f"{r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)"
        lines.append(f"{r['name']:<{col_w}} {acc:>17} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f}")
    lines.append("=" * 75)
    lines.append(f"  Random chance (uniform): ~16.7%")
    if len(results) >= 2:
        gain = results[-1]['accuracy'] - results[0]['accuracy']
        lines.append(f"  Gain of fine-tuned model over majority baseline: +{gain*100:.1f} pp accuracy")
    lines.append("")

    lines.append("PER-CLASS BREAKDOWN")
    lines.append("")
    for r in results:
        if r.get('report'):
            lines.append(r['name'])
            lines.append("-" * 40)
            lines.append(r['report'])
            lines.append("")

    BASELINE_REPORT.write_text("\n".join(lines))

    print(f"Results saved to {BASELINE_RESULTS}")
    print(f"Full report saved to {BASELINE_REPORT}")


if __name__ == '__main__':
    main()
