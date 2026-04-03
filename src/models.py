"""
Model definitions for Fake News Detection.
Uses Tinker SDK for LoRA fine-tuning of a supported LLM.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class ClassBalancer:
    """Handle class imbalance in dataset."""

    @staticmethod
    def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
        """
        Compute class weights for imbalanced dataset.

        Args:
            y: Encoded labels array
            num_classes: Number of classes

        Returns:
            Dictionary mapping class index to weight
        """
        logger.info("Computing class weights...")
        classes = np.arange(num_classes)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = {i: w for i, w in enumerate(weights)}
        logger.info("Class weights computed:")
        for cls, weight in class_weights.items():
            logger.info(f"  Class {cls}: {weight:.4f}")
        return class_weights

    @staticmethod
    def oversample_minority(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority classes to balance dataset.

        Args:
            X: Padded sequences
            y: Labels

        Returns:
            Oversampled X and y
        """
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError:
            logger.warning("imbalanced-learn not installed. Install via: pip install imbalanced-learn")
            return X, y

        logger.info("Oversampling minority classes...")
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_flat, y)
        X_resampled = X_resampled.reshape(-1, original_shape[1])
        logger.info(f"Original: {len(y)} → Resampled: {len(y_resampled)}")
        return X_resampled, y_resampled

    @staticmethod
    def undersample_majority(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersample majority classes to balance dataset.

        Args:
            X: Padded sequences
            y: Labels

        Returns:
            Undersampled X and y
        """
        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            logger.warning("imbalanced-learn not installed. Install via: pip install imbalanced-learn")
            return X, y

        logger.info("Undersampling majority classes...")
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_flat, y)
        X_resampled = X_resampled.reshape(-1, original_shape[1])
        logger.info(f"Original: {len(y)} → Resampled: {len(y_resampled)}")
        return X_resampled, y_resampled


class TinkerClassifier:
    """
    Fake news classifier using Tinker LoRA fine-tuning on a supported LLM.

    Training flow:
        classifier = TinkerClassifier(base_model='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
        classifier.connect()
        classifier.create_training_client()
        tokenizer = classifier.get_tokenizer()
        # ... build datums from data_processor.prepare_tinker_dataset() ...
        for epoch in range(epochs):
            loss = classifier.train_step(batch)
        classifier.save_for_inference()
        label = classifier.predict("Some political statement")

    Inference flow (from saved weights):
        classifier = TinkerClassifier(base_model='...')
        classifier.connect()
        classifier.load_sampling_client(tinker_uri)
        label = classifier.predict("Some political statement")
    """

    # Labels sorted alphabetically — must match LabelEncoder order
    LABELS = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']

    def __init__(self, base_model: str, prompt_template: Optional[str] = None):
        """
        Args:
            base_model: Tinker model identifier (e.g. 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
            prompt_template: Classification prompt with {text} placeholder.
                             Defaults to PROMPT_TEMPLATE from config.
        """
        self.base_model = base_model
        if prompt_template is None:
            from config import PROMPT_TEMPLATE
            self.prompt_template = PROMPT_TEMPLATE
        else:
            self.prompt_template = prompt_template

        self.service_client = None
        self.training_client = None
        self.sampling_client = None
        self._tokenizer = None
        logger.info(f"TinkerClassifier initialized with base_model={base_model}")

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        """
        Connect to the Tinker service.
        Reads TINKER_API_KEY from the environment automatically.
        """
        try:
            from tinker import ServiceClient
        except ImportError:
            raise ImportError("tinker SDK not installed. Run: pip install tinker")

        logger.info("Connecting to Tinker...")
        self.service_client = ServiceClient()
        logger.info("Connected to Tinker service")

    # ── Training ──────────────────────────────────────────────────────────────

    def create_training_client(self, lora_rank: int = 16):
        """
        Initialize LoRA training on the remote model.
        Blocks until Tinker has allocated resources (usually a few seconds).

        Args:
            lora_rank: LoRA rank. Higher = more parameters, slower but potentially
                       better quality. 16 is a good default.
        """
        if self.service_client is None:
            self.connect()

        try:
            from tinker import types
        except ImportError:
            raise ImportError("tinker SDK not installed. Run: pip install tinker")

        logger.info(f"Creating LoRA training client (rank={lora_rank}, model={self.base_model})...")
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=lora_rank,
        )
        logger.info("Training client ready")
        return self.training_client

    def get_tokenizer(self):
        """Return the tokenizer for the loaded model (cached after first call)."""
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")
        if self._tokenizer is None:
            self._tokenizer = self.training_client.get_tokenizer()
        return self._tokenizer

    def train_step(self, datums: List[Any], learning_rate: float = 1e-4) -> float:
        """
        Run one forward-backward pass and optimizer step on a batch of datums.

        Futures are submitted before calling .result() so that the optim step
        can be pipelined with the fwdbwd computation on Tinker's side.

        Args:
            datums: List of types.Datum objects for this batch
            learning_rate: AdamW learning rate for this step

        Returns:
            Cross-entropy loss per completion token (float)
        """
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")

        from tinker import types

        # Submit both requests before blocking on either result
        fwdbwd_future = self.training_client.forward_backward(
            data=datums,
            loss_fn="cross_entropy",
        )
        optim_future = self.training_client.optim_step(
            adam_params=types.AdamParams(learning_rate=learning_rate),
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        # Compute mean cross-entropy loss over completion tokens
        logprobs = np.concatenate([
            output['logprobs'].to_numpy()
            for output in fwdbwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            datum.loss_fn_inputs['weights'].to_numpy()
            for datum in datums
        ])
        weight_sum = weights.sum()
        loss = float(-np.dot(logprobs, weights) / weight_sum) if weight_sum > 0 else 0.0
        return loss

    def save_for_inference(self, name: str = 'fake-news-classifier') -> str:
        """
        Save LoRA weights and create a sampling client for inference.

        Args:
            name: Human-readable name for this checkpoint

        Returns:
            Tinker URI of the saved weights (store this to reload later)
        """
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")

        logger.info(f"Saving weights as '{name}'...")
        # Returns SamplingClient directly (not a Future)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=name
        )
        uri = self.training_client.holder.get_session_id() + ':train:0'
        logger.info(f"Weights saved. URI: {uri}")
        return uri

    # ── Inference ─────────────────────────────────────────────────────────────

    def load_sampling_client(self, tinker_uri: str):
        """
        Load a previously saved checkpoint for inference without retraining.

        Args:
            tinker_uri: tinker:// URI returned by save_for_inference()
        """
        if self.service_client is None:
            self.connect()

        logger.info(f"Loading sampling client from {tinker_uri}...")
        self.sampling_client = self.service_client.create_sampling_client(
            base_model=tinker_uri
        )
        # get_tokenizer() on the sampling client tries to resolve the URI as a HF
        # model ID — load from the known base model name instead
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        logger.info("Sampling client loaded")

    def predict(self, text: str, return_proba: bool = False):
        """
        Predict the truthfulness label for a single statement.

        Args:
            text: The political statement to classify
            return_proba: If True, also return per-label log-probabilities

        Returns:
            Predicted label string, or (label, logprob_dict) if return_proba=True
        """
        if self.sampling_client is None:
            raise RuntimeError("No sampling client. Call save_for_inference() or load_sampling_client() first")

        from tinker import types

        tokenizer = self._tokenizer or self.sampling_client.get_tokenizer()
        prompt = self.prompt_template.format(text=text)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        result = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=10,
                temperature=0.0,
                stop=["\n", ".", ","],
            ),
        ).result()

        generated = tokenizer.decode(result.sequences[0].tokens).strip().lower()

        # Match to the nearest known label
        predicted = self._match_label(generated)

        if return_proba:
            logprob_dict = self._compute_label_logprobs(model_input, tokenizer)
            return predicted, logprob_dict

        return predicted

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict labels for a list of statements."""
        return [self.predict(text) for text in texts]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _match_label(self, generated: str) -> str:
        """Match raw generated text to the closest known label."""
        generated = generated.strip().lower()
        for label in self.LABELS:
            if label in generated:
                return label
        # Fallback: return closest prefix match
        for label in self.LABELS:
            if generated.startswith(label[:4]):
                return label
        logger.warning(f"Could not match generated text to label: {repr(generated)}")
        return generated

    def _compute_label_logprobs(self, prompt_input: Any, tokenizer) -> Dict[str, float]:
        """
        Compute the log-probability of each label token given the prompt.
        Used for return_proba=True in predict().
        """
        from tinker import types

        logprob_dict = {}
        for label in self.LABELS:
            completion = f" {label}"
            completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
            full_input = types.ModelInput.from_ints(
                tokens=prompt_input.to_ints() + completion_tokens
            )
            # compute_logprobs returns ConcurrentFuture[list[float | None]]
            logprobs = self.sampling_client.compute_logprobs(full_input).result()
            # Sum logprobs over completion token positions only (last N positions)
            completion_logprob = float(sum(
                lp for lp in logprobs[-len(completion_tokens):]
                if lp is not None
            ))
            logprob_dict[label] = completion_logprob

        return logprob_dict

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary of the current model state."""
        return {
            'base_model': self.base_model,
            'type': 'TinkerClassifier (LoRA)',
            'labels': self.LABELS,
            'training_client': 'active' if self.training_client else 'none',
            'sampling_client': 'active' if self.sampling_client else 'none',
        }


if __name__ == '__main__':
    print("Models module loaded. Available classes:")
    print("  ClassBalancer")
    print("  TinkerClassifier")
