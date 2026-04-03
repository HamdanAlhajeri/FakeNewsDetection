"""
Training utilities and helpers for model training.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Utility for splitting data into train/validation/test sets."""
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, 
                   train_size: float = 0.7, 
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Label array
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        logger.info(f"Splitting data: train={train_size}, val={val_size}, test={test_size}")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: list = None) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names for reporting
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report
        )
        
        logger.info("Calculating metrics...")
        
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }
        
        if labels:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, 
                target_names=labels,
                output_dict=True
            )
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Weighted F1: {weighted_f1:.4f}")
        
        return metrics


class ModelTrainer:
    """Base trainer for models."""
    
    def __init__(self, model: Any, 
                 loss_fn: str = 'crossentropy',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.00005):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimizer name
            learning_rate: Learning rate
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        logger.info(f"Initialized ModelTrainer with {optimizer} optimizer")
    
    def compile_model(self):
        """Compile model (placeholder for subclasses)."""
        logger.info("Model compiled")
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, 
                    batch_size: int = 32) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Batch size
            
        Returns:
            Dictionary with loss and metrics
        """
        logger.info(f"Training epoch with batch_size={batch_size}")
        # Placeholder implementation
        return {'loss': 0.5, 'accuracy': 0.8}
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating model")
        # Placeholder implementation
        return {'loss': 0.6, 'accuracy': 0.78}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10, batch_size: int = 32) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(X_train, y_train, batch_size)
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['train_acc'].append(train_metrics.get('accuracy', 0))
            
            # Validate
            val_metrics = self.validate(X_val, y_val)
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            
            logger.info(f"Train Loss: {train_metrics.get('loss'):.4f}, "
                       f"Val Loss: {val_metrics.get('loss'):.4f}")
        
        return self.history


class TinkerTrainer:
    """Training loop for TinkerClassifier."""

    def __init__(self, classifier, learning_rate: float = 1e-4, batch_size: int = 8):
        """
        Args:
            classifier: TinkerClassifier instance with an active training_client
            learning_rate: AdamW learning rate
            batch_size: Number of datums per forward-backward call
        """
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.history: Dict[str, list] = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, datums: list) -> float:
        """
        Run one full pass over the training datums in mini-batches.

        Args:
            datums: All training Datum objects for this epoch

        Returns:
            Mean loss over all batches
        """
        import math
        np.random.shuffle(datums)

        batch_losses = []
        num_batches = math.ceil(len(datums) / self.batch_size)

        for i in range(0, len(datums), self.batch_size):
            batch = datums[i: i + self.batch_size]
            loss = self.classifier.train_step(batch, learning_rate=self.learning_rate)
            batch_losses.append(loss)
            batch_num = i // self.batch_size + 1
            if batch_num % 10 == 0 or batch_num == num_batches:
                logger.info(f"  Batch {batch_num}/{num_batches}  loss={loss:.4f}")

        epoch_loss = float(np.mean(batch_losses))
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss

    def evaluate_loss(self, datums: list) -> float:
        """
        Compute validation loss using forward-only passes (no gradient computation).

        Args:
            datums: Validation Datum objects

        Returns:
            Mean cross-entropy loss per completion token
        """
        import math

        all_logprobs = []
        all_weights = []

        for i in range(0, len(datums), self.batch_size):
            batch = datums[i: i + self.batch_size]
            result = self.classifier.training_client.forward(
                data=batch,
                loss_fn="cross_entropy",
            ).result()

            for output, datum in zip(result.loss_fn_outputs, batch):
                all_logprobs.extend(output['logprobs'].to_numpy().tolist())
                all_weights.extend(datum.loss_fn_inputs['weights'].to_numpy().tolist())

        all_logprobs = np.array(all_logprobs)
        all_weights = np.array(all_weights)
        weight_sum = all_weights.sum()
        val_loss = float(-np.dot(all_logprobs, all_weights) / weight_sum) if weight_sum > 0 else 0.0
        self.history['val_loss'].append(val_loss)
        return val_loss

    def train(self, train_datums: list, val_datums: list, epochs: int) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_datums: Training Datum objects
            val_datums: Validation Datum objects
            epochs: Number of epochs

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists
        """
        logger.info(f"Starting Tinker training: {epochs} epochs, "
                    f"{len(train_datums)} train / {len(val_datums)} val datums, "
                    f"batch_size={self.batch_size}, lr={self.learning_rate}")

        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")

            train_loss = self.train_epoch(train_datums)
            val_loss = self.evaluate_loss(val_datums)

            logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        return self.history


if __name__ == '__main__':
    print("Training utilities loaded successfully")
