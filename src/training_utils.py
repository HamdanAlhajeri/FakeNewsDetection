"""
Training utilities for Fake News Detection — Tinker API.
Includes TinkerTrainer, MetricsCalculator, plotting, and result saving.
"""

import json
import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    LABEL_NAMES, RESULTS_PATH, LOSS_CURVE_PATH, CONFUSION_MATRIX_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Compute classification metrics from true and predicted label arrays."""

    @staticmethod
    def calculate(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        acc         = accuracy_score(y_true, y_pred)
        macro_f1    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm          = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
        report      = classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
        )
        return {
            'accuracy':              acc,
            'macro_f1':              macro_f1,
            'weighted_f1':           weighted_f1,
            'confusion_matrix':      cm.tolist(),
            'classification_report': report,
        }


class TinkerTrainer:
    """
    Training loop for TinkerClassifier.

    Usage:
        trainer = TinkerTrainer(classifier, learning_rate, batch_size)
        history = trainer.train(train_datums, val_datums, epochs, patience=2)
    """

    def __init__(self, classifier, learning_rate: float, batch_size: int):
        self.classifier    = classifier
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        self.best_epoch: int = 1

    def train_epoch(self, datums: list) -> float:
        """One full pass over training datums in mini-batches. Returns mean loss."""
        np.random.shuffle(datums)
        batch_losses = []
        num_batches  = math.ceil(len(datums) / self.batch_size)

        for i in range(0, len(datums), self.batch_size):
            batch     = datums[i: i + self.batch_size]
            loss      = self.classifier.train_step(batch, learning_rate=self.learning_rate)
            batch_num = i // self.batch_size + 1
            batch_losses.append(loss)
            if batch_num % 50 == 0 or batch_num == num_batches:
                logger.info(f"  Batch {batch_num}/{num_batches}  loss={loss:.4f}")

        epoch_loss = float(np.mean(batch_losses))
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss

    def evaluate_loss(self, datums: list) -> float:
        """Forward-only pass over validation datums. Returns mean loss."""
        all_logprobs, all_weights = [], []

        for i in range(0, len(datums), self.batch_size):
            batch  = datums[i: i + self.batch_size]
            result = self.classifier.training_client.forward(
                data=batch, loss_fn="cross_entropy"
            ).result()

            for output, datum in zip(result.loss_fn_outputs, batch):
                all_logprobs.extend(output['logprobs'].to_numpy().tolist())
                all_weights.extend(datum.loss_fn_inputs['weights'].to_numpy().tolist())

        all_logprobs = np.array(all_logprobs)
        all_weights  = np.array(all_weights)
        weight_sum   = all_weights.sum()
        val_loss     = float(-np.dot(all_logprobs, all_weights) / weight_sum) if weight_sum > 0 else 0.0
        self.history['val_loss'].append(val_loss)
        return val_loss

    def train(self, train_datums: list, val_datums: list, epochs: int,
              patience: int = 2) -> Dict[str, List[float]]:
        logger.info(
            f"Starting training — {epochs} epochs (patience={patience}), "
            f"{len(train_datums)} train / {len(val_datums)} val datums, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}"
        )
        best_val_loss  = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_datums)
            val_loss   = self.evaluate_loss(val_datums)
            logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                self.best_epoch   = epoch
                epochs_no_improve = 0
                logger.info(f"  New best val loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(
                    f"  No improvement ({epochs_no_improve}/{patience}). "
                    f"Best so far: {best_val_loss:.4f} at epoch {self.best_epoch}"
                )
                if epochs_no_improve >= patience:
                    logger.info(f"  Early stopping triggered at epoch {epoch}.")
                    break

        return self.history


# ── Plotting & saving ──────────────────────────────────────────────────────────

def plot_curves(history: Dict[str, List[float]]):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.title('Training / Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=120)
    plt.close()
    logger.info(f"Loss curve saved to {LOSS_CURVE_PATH}")


def plot_confusion_matrix(y_true: List[int], y_pred: List[int]):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES
    )
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close()
    logger.info(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


def save_results(history: Dict[str, List[float]], test_metrics: Dict[str, Any]):
    results = {
        'history': {k: [round(v, 6) for v in vals] for k, vals in history.items()},
        'test': {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")
