"""
Prediction script for fake news detection using trained models.
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import TRUTHFULNESS_LABELS, RANDOM_SEED


class FakeNewsPredictor:
    """Predict fake news using trained models."""
    
    def __init__(self, model_path: str, model_type: str = 'bert', device: str = 'cuda'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            model_type: 'bert' or 'roberta'
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Load model name
        if model_type == 'bert':
            self.model_name = 'bert-base-uncased'
        elif model_type == 'roberta':
            self.model_name = 'roberta-base'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get label names
        if isinstance(TRUTHFULNESS_LABELS, dict):
            self.labels = [TRUTHFULNESS_LABELS[i] for i in sorted(TRUTHFULNESS_LABELS.keys())]
        else:
            self.labels = TRUTHFULNESS_LABELS
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded {model_type.upper()} model from {model_path}")
    
    def predict(self, text: str, return_proba: bool = False):
        """
        Predict label for a text.
        
        Args:
            text: Input text
            return_proba: Return probabilities
            
        Returns:
            Predicted label (and probabilities if return_proba=True)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = np.argmax(probs)
        pred_label = self.labels[pred_idx]
        
        if return_proba:
            proba_dict = {label: float(prob) for label, prob in zip(self.labels, probs)}
            return pred_label, proba_dict
        else:
            return pred_label
    
    def predict_batch(self, texts: list, return_proba: bool = False):
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of texts
            return_proba: Return probabilities
            
        Returns:
            List of predictions
        """
        predictions = []
        confidences = []
        
        for text in texts:
            if return_proba:
                pred, proba = self.predict(text, return_proba=True)
                max_proba = max(proba.values())
                predictions.append(pred)
                confidences.append(max_proba)
            else:
                pred = self.predict(text, return_proba=False)
                predictions.append(pred)
        
        if return_proba:
            return predictions, confidences
        else:
            return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict fake news labels')
    parser.add_argument('--model', default='artifacts/bert_model.pt', help='Path to model')
    parser.add_argument('--type', default='bert', choices=['bert', 'roberta'], help='Model type')
    parser.add_argument('--text', help='Text to predict')
    parser.add_argument('--file', help='File with texts (one per line)')
    parser.add_argument('--proba', action='store_true', help='Return probabilities')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FakeNewsPredictor(args.model, model_type=args.type, device=args.device)
    
    # Predict
    if args.text:
        label, proba = predictor.predict(args.text, return_proba=args.proba)
        print(f"\nText: {args.text}")
        print(f"Prediction: {label}")
        if args.proba:
            print(f"\nProbabilities:")
            for l, p in sorted(proba.items(), key=lambda x: x[1], reverse=True):
                print(f"  {l}: {p:.4f}")
    
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        predictions, confidences = predictor.predict_batch(texts, return_proba=args.proba)
        
        print(f"\nPredictions for {len(texts)} texts:")
        for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidences)):
            print(f"{i+1}. {text[:50]}... -> {pred} (confidence: {conf:.4f})")


if __name__ == '__main__':
    main()
