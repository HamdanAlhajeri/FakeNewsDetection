"""
Prediction script for fake news detection.

Two modes:
  --mode tinker   Use a Tinker-trained LLM (default, requires trained weights)
  --mode local    Use a locally saved BERT/RoBERTa .pt checkpoint

Usage examples:
  python src/predict.py --text "The unemployment rate is 3 percent"
  python src/predict.py --text "Statement here" --proba
  python src/predict.py --file data/statements.txt
  python src/predict.py --mode local --model artifacts/bert_model.pt --type bert --text "..."
"""

import argparse
import json
from pathlib import Path


# ── Tinker predictor ──────────────────────────────────────────────────────────

class TinkerPredictor:
    """
    Predict fake news labels using a Tinker-trained LLM sampling client.

    The tinker_uri is the weights path printed at the end of notebook 03,
    e.g. 'tinker://uuid:train:0/sampler_weights/checkpoint_name'
    """

    def __init__(self, tinker_uri: str):
        """
        Args:
            tinker_uri: Tinker weights URI returned by save_for_inference()
                        or found in artifacts/tinker_weights_uri.txt
        """
        from models import TinkerClassifier
        self.classifier = TinkerClassifier(base_model=tinker_uri)
        self.classifier.connect()
        self.classifier.load_sampling_client(tinker_uri)
        print(f"Loaded Tinker model from {tinker_uri}")

    def predict(self, text: str, return_proba: bool = False):
        return self.classifier.predict(text, return_proba=return_proba)

    def predict_batch(self, texts: list) -> list:
        return self.classifier.predict_batch(texts)


# ── Local (BERT/RoBERTa) predictor ───────────────────────────────────────────

class LocalPredictor:
    """
    Predict fake news labels using a locally saved BERT or RoBERTa .pt checkpoint.
    """

    def __init__(self, model_path: str, model_type: str = 'bert', device: str = 'cuda'):
        """
        Args:
            model_path: Path to .pt checkpoint file
            model_type: 'bert' or 'roberta'
            device: 'cuda' or 'cpu'
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from config import TRUTHFULNESS_LABELS

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        model_name = 'bert-base-uncased' if model_type == 'bert' else 'roberta-base'

        if isinstance(TRUTHFULNESS_LABELS, dict):
            self.labels = [TRUTHFULNESS_LABELS[i] for i in sorted(TRUTHFULNESS_LABELS.keys())]
        else:
            self.labels = list(TRUTHFULNESS_LABELS)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.labels)
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_type.upper()} model from {model_path}")

    def predict(self, text: str, return_proba: bool = False):
        import torch
        import numpy as np

        inputs = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=128, return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        pred_label = self.labels[int(np.argmax(probs))]

        if return_proba:
            return pred_label, {l: float(p) for l, p in zip(self.labels, probs)}
        return pred_label

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict fake news labels')
    parser.add_argument('--mode', default='tinker', choices=['tinker', 'local'],
                        help='Prediction mode: tinker (default) or local (.pt file)')

    # Tinker args
    parser.add_argument('--uri', default=None,
                        help='[tinker mode] Tinker weights URI. '
                             'Defaults to contents of artifacts/tinker_weights_uri.txt')

    # Local args
    parser.add_argument('--model', default='artifacts/bert_model.pt',
                        help='[local mode] Path to .pt model checkpoint')
    parser.add_argument('--type', default='bert', choices=['bert', 'roberta'],
                        help='[local mode] Model type')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='[local mode] Device to run inference on')

    # Shared args
    parser.add_argument('--text', help='Single statement to classify')
    parser.add_argument('--file', help='File with one statement per line')
    parser.add_argument('--proba', action='store_true',
                        help='Print per-label probabilities / log-probabilities')

    args = parser.parse_args()

    # Build predictor
    if args.mode == 'tinker':
        uri = args.uri
        if uri is None:
            uri_file = Path('artifacts/tinker_weights_uri.txt')
            if uri_file.exists():
                uri = uri_file.read_text().strip()
            else:
                parser.error(
                    "No Tinker URI supplied. Pass --uri or train first "
                    "(notebook 03 saves it to artifacts/tinker_weights_uri.txt)"
                )
        predictor = TinkerPredictor(tinker_uri=uri)
    else:
        predictor = LocalPredictor(args.model, model_type=args.type, device=args.device)

    # Run prediction
    if args.text:
        if args.proba:
            label, proba = predictor.predict(args.text, return_proba=True)
            print(f"\nText:       {args.text}")
            print(f"Prediction: {label}")
            print("\nPer-label scores:")
            for l, p in sorted(proba.items(), key=lambda x: x[1], reverse=True):
                print(f"  {l:<14s} {p:.4f}")
        else:
            label = predictor.predict(args.text, return_proba=False)
            print(f"\nText:       {args.text}")
            print(f"Prediction: {label}")

    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        predictions = predictor.predict_batch(texts)
        print(f"\nPredictions for {len(texts)} statements:")
        for i, (text, pred) in enumerate(zip(texts, predictions), 1):
            print(f"  {i}. {text[:60]}... → {pred}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
