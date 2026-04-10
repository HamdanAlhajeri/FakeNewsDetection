"""
Model definition for Fake News Detection.
TinkerClassifier — LoRA fine-tuning of Nemotron 30B via the Tinker API.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from config import TINKER_CONFIG, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class TinkerClassifier:
    """
    Fake news classifier using Tinker LoRA fine-tuning on Nemotron 30B.

    Training flow:
        classifier = TinkerClassifier()
        classifier.connect()
        classifier.create_training_client()
        tokenizer = classifier.get_tokenizer()
        # build datums via DataProcessor.prepare_tinker_dataset()
        for epoch in range(epochs):
            loss = classifier.train_step(batch)
        uri = classifier.save_for_inference('name')

    Inference flow:
        classifier = TinkerClassifier()
        classifier.connect()
        classifier.load_sampling_client(uri)
        label = classifier.predict("Some political statement")
    """

    LABELS = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']

    def __init__(self, base_model: str = None, prompt_template: str = None):
        self.base_model      = base_model or TINKER_CONFIG['base_model']
        self.prompt_template = prompt_template or PROMPT_TEMPLATE

        self.service_client  = None
        self.training_client = None
        self.sampling_client = None
        self._tokenizer      = None
        logger.info(f"TinkerClassifier — base_model={self.base_model}")

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        try:
            from tinker import ServiceClient
        except ImportError:
            raise ImportError("tinker SDK not installed. Run: pip install tinker")
        logger.info("Connecting to Tinker...")
        self.service_client = ServiceClient()
        logger.info("Connected")

    # ── Training ──────────────────────────────────────────────────────────────

    def create_training_client(self, lora_rank: int = None):
        if self.service_client is None:
            self.connect()
        rank = lora_rank or TINKER_CONFIG['lora_rank']
        logger.info(f"Creating LoRA training client (rank={rank})...")
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=rank,
        )
        logger.info("Training client ready")
        return self.training_client

    def get_tokenizer(self):
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")
        if self._tokenizer is None:
            self._tokenizer = self.training_client.get_tokenizer()
        return self._tokenizer

    def train_step(self, datums: List[Any], learning_rate: float = None) -> float:
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")
        from tinker import types
        lr = learning_rate or TINKER_CONFIG['learning_rate']

        fwdbwd_future = self.training_client.forward_backward(
            data=datums, loss_fn="cross_entropy"
        )
        optim_future = self.training_client.optim_step(
            adam_params=types.AdamParams(learning_rate=lr)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([
            output['logprobs'].to_numpy()
            for output in fwdbwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            datum.loss_fn_inputs['weights'].to_numpy()
            for datum in datums
        ])
        weight_sum = weights.sum()
        return float(-np.dot(logprobs, weights) / weight_sum) if weight_sum > 0 else 0.0

    def save_for_inference(self, name: str = 'fake-news-classifier') -> str:
        if self.training_client is None:
            raise RuntimeError("Call create_training_client() first")
        logger.info(f"Saving weights as '{name}'...")
        result = self.training_client.save_weights_for_sampler(name).result()
        uri = result.path
        self.sampling_client = self.service_client.create_sampling_client(model_path=uri)
        logger.info(f"Weights saved. URI: {uri}")
        return uri

    # ── Inference ─────────────────────────────────────────────────────────────

    def load_sampling_client(self, tinker_uri: str):
        if self.service_client is None:
            self.connect()
        logger.info(f"Loading sampling client from {tinker_uri}...")
        self.sampling_client = self.service_client.create_sampling_client(
            model_path=tinker_uri,
            base_model=self.base_model,
        )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        logger.info("Sampling client loaded")

    def predict(self, text: str, return_proba: bool = False,
                speaker: str = '', party: str = '', job: str = '',
                subject: str = '', context: str = '',
                barely_true_count: int = 0, false_count: int = 0,
                half_true_count: int = 0, mostly_true_count: int = 0,
                pants_on_fire_count: int = 0):
        """
        Predict label for a statement.

        Pass metadata and historical count fields to match the richer
        input used during training.
        """
        if self.sampling_client is None:
            raise RuntimeError("Call load_sampling_client() or save_for_inference() first")

        from tinker import types
        import pandas as pd
        from data_processor import DataProcessor

        row = pd.Series({
            'statement':           text,
            'speaker':             speaker,
            'party_affiliation':   party,
            'job_title':           job,
            'subject':             subject,
            'context':             context,
            'barely_true_count':   barely_true_count,
            'false_count':         false_count,
            'half_true_count':     half_true_count,
            'mostly_true_count':   mostly_true_count,
            'pants_on_fire_count': pants_on_fire_count,
        })
        combined = DataProcessor.build_input_text(row)

        tokenizer    = self._tokenizer or self.sampling_client.get_tokenizer()
        prompt       = self.prompt_template.format(text=combined)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        model_input   = types.ModelInput.from_ints(tokens=prompt_tokens)

        result = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=TINKER_CONFIG['max_inference_tokens'],
                temperature=TINKER_CONFIG['temperature'],
                stop=["\n", ".", ","],
            ),
        ).result()

        generated = tokenizer.decode(result.sequences[0].tokens).strip().lower()
        predicted  = self._match_label(generated)

        if return_proba:
            logprob_dict = self._compute_label_logprobs(model_input, tokenizer)
            return predicted, logprob_dict

        return predicted

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t) for t in texts]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _match_label(self, generated: str) -> str:
        generated = generated.strip().lower()
        for label in self.LABELS:
            if label in generated:
                return label
        for label in self.LABELS:
            if generated.startswith(label[:4]):
                return label
        logger.warning(f"Could not match generated text to label: {repr(generated)}")
        return generated

    def _compute_label_logprobs(self, prompt_input: Any, tokenizer) -> Dict[str, float]:
        from tinker import types
        logprob_dict = {}
        for label in self.LABELS:
            completion        = f" {label}"
            completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
            full_input        = types.ModelInput.from_ints(
                tokens=prompt_input.to_ints() + completion_tokens
            )
            logprobs = self.sampling_client.compute_logprobs(full_input).result()
            completion_logprob = float(sum(
                lp for lp in logprobs[-len(completion_tokens):]
                if lp is not None
            ))
            logprob_dict[label] = completion_logprob
        return logprob_dict
