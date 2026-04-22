"""
Data processing for Fake News Detection.
Loads LIAR2 TSV files, cleans text, combines all features,
and prepares Tinker Datum objects for LoRA fine-tuning.
"""

import re
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

from config import TSV_COLUMNS, LABEL_TO_IDX, LABEL_NAMES

try:
    _STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOPWORDS = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Load, clean, and prepare LIAR2 data for Tinker fine-tuning."""

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def load_data(filepath) -> pd.DataFrame:
        df = pd.read_csv(
            filepath, sep='\t', header=None,
            names=TSV_COLUMNS, quoting=3, on_bad_lines='skip'
        )
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    # ── Cleaning ──────────────────────────────────────────────────────────────

    @staticmethod
    def clean_text(text) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        tokens = [t for t in text.split() if t not in _STOPWORDS]
        return ' '.join(tokens)

    # Count columns → human-readable label names
    _COUNT_COLS = [
        ('barely_true_count',    'barely-true'),
        ('false_count',          'false'),
        ('half_true_count',      'half-true'),
        ('mostly_true_count',    'mostly-true'),
        ('pants_on_fire_count',  'pants-fire'),
    ]

    @staticmethod
    def _speaker_history(row: pd.Series) -> str:
        """
        Derive speaker credibility stats from historical count columns.

        Returns a string like:
            "Speaker history: 45 prior claims — mostly-true: 35%, false: 22%, ..."
        or empty string if no history is available.
        """
        counts = {}
        for col, label in DataProcessor._COUNT_COLS:
            try:
                val = int(float(row.get(col, 0) or 0))
            except (ValueError, TypeError):
                val = 0
            counts[label] = val

        total = sum(counts.values())
        if total == 0:
            return ""

        # Sort by frequency descending, show all non-zero rates
        rates = sorted(
            [(label, cnt / total) for label, cnt in counts.items() if cnt > 0],
            key=lambda x: x[1], reverse=True
        )
        rate_str = ', '.join(f"{lbl}: {pct:.0%}" for lbl, pct in rates)
        dominant = rates[0][0]
        return (
            f"Speaker history: {total} prior claims — {rate_str}. "
            f"Most common rating: {dominant}"
        )

    @staticmethod
    def build_input_text(row: pd.Series) -> str:
        """
        Concatenate all feature fields plus engineered credibility stats
        into a single model-input string.
        """
        parts = {
            'Statement': row.get('statement', ''),
            'Speaker':   row.get('speaker', ''),
            'Party':     row.get('party_affiliation', ''),
            'Job':       row.get('job_title', ''),
            'Subject':   row.get('subject', ''),
            'Context':   row.get('context', ''),
        }
        segments = []
        for key, val in parts.items():
            cleaned = DataProcessor.clean_text(val)
            if cleaned:
                segments.append(f"{key}: {cleaned}")

        # Append speaker credibility history (engineered feature)
        history = DataProcessor._speaker_history(row)
        if history:
            segments.append(history)

        return ' '.join(segments)

    # ── Label encoding ────────────────────────────────────────────────────────

    @staticmethod
    def encode_labels(labels: pd.Series) -> List[int]:
        encoded = []
        for lbl in labels:
            lbl = str(lbl).strip().lower()
            if lbl not in LABEL_TO_IDX:
                raise ValueError(f"Unknown label: {repr(lbl)}")
            encoded.append(LABEL_TO_IDX[lbl])
        return encoded

    # ── Full pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def process(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Clean, combine features, and encode labels.

        Returns:
            texts  — combined feature strings, one per sample
            labels — integer label indices (0–5)
        """
        df = df.dropna(subset=['label', 'statement']).copy()
        texts  = [DataProcessor.build_input_text(row) for _, row in df.iterrows()]
        labels = DataProcessor.encode_labels(df['label'])
        # Drop rows whose combined text is empty after cleaning
        filtered = [(t, l) for t, l in zip(texts, labels) if t.strip()]
        if not filtered:
            raise ValueError("All texts are empty after cleaning.")
        texts, labels = zip(*filtered)
        return list(texts), list(labels)

    # ── Tinker datum preparation ───────────────────────────────────────────────

    @staticmethod
    def prepare_tinker_datum(text: str, label: str, tokenizer: Any,
                             prompt_template: str,
                             class_weight: float = 1.0) -> Any:
        """
        Convert a single (text, label) pair into a Tinker Datum.

        Args:
            text:            Combined feature string from build_input_text()
            label:           Label string, e.g. 'false'
            tokenizer:       Tinker tokenizer from training_client.get_tokenizer()
            prompt_template: Format string with {text} placeholder
            class_weight:    Multiplier applied to completion token weights (for class balancing)

        Returns:
            types.Datum ready for forward_backward()
        """
        from tinker import types, TensorData

        prompt     = prompt_template.format(text=text)
        completion = f" {label}"

        prompt_tokens     = tokenizer.encode(prompt,      add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion,  add_special_tokens=False)

        # Weight 0 = don't train on prompt tokens; class_weight = train on completion tokens
        weights    = [0.0] * len(prompt_tokens) + [class_weight] * len(completion_tokens)
        all_tokens = prompt_tokens + completion_tokens

        # Shift for next-token prediction
        input_tokens  = all_tokens[:-1]
        target_tokens = all_tokens[1:]
        weights       = weights[1:]

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(
                weights=TensorData(data=weights, dtype='float32'),
                target_tokens=TensorData(data=target_tokens, dtype='int64'),
            )
        )

    @staticmethod
    def compute_class_weights(labels: List[str]) -> Dict[str, float]:
        """
        Compute inverse-frequency class weights so each class contributes equally to loss.

        Returns a dict mapping label string → weight multiplier.
        """
        from collections import Counter
        counts  = Counter(labels)
        n_total = len(labels)
        n_classes = len(counts)
        return {
            label: (n_total / (n_classes * count))
            for label, count in counts.items()
        }

    @staticmethod
    def prepare_tinker_dataset(texts: List[str], labels: List[str],
                               tokenizer: Any, prompt_template: str,
                               class_weights: Dict[str, float] = None) -> List[Any]:
        """
        Build a list of Tinker Datums from combined-feature texts and string labels.

        Args:
            texts:           Combined feature strings (from process())
            labels:          String label names, e.g. ['false', 'half-true', ...]
            tokenizer:       Tinker tokenizer
            prompt_template: Format string with {text} placeholder
            class_weights:   Optional dict of label → weight multiplier for class balancing

        Returns:
            List of types.Datum objects
        """
        datums = []
        for text, label in zip(texts, labels):
            weight = class_weights.get(label, 1.0) if class_weights else 1.0
            datum  = DataProcessor.prepare_tinker_datum(
                text, label, tokenizer, prompt_template, class_weight=weight
            )
            datums.append(datum)
        logger.info(f"Prepared {len(datums)} Tinker datums")
        return datums
