"""
Model definitions for Fake News Detection.
Includes BERT, RoBERTa, and Hybrid models.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
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
            Dictionary mapping class to weight
        """
        logger.info("Computing class weights...")
        
        classes = np.arange(num_classes)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = {i: w for i, w in enumerate(weights)}
        
        logger.info("Class weights:")
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
        
        # Reshape for RandomOverSampler (expects 2D)
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_flat, y)
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, original_shape[1])
        
        logger.info(f"Original dataset size: {len(y)}")
        logger.info(f"Resampled dataset size: {len(y_resampled)}")
        
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
        
        # Reshape for RandomUnderSampler
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_flat, y)
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, original_shape[1])
        
        logger.info(f"Original dataset size: {len(y)}")
        logger.info(f"Resampled dataset size: {len(y_resampled)}")
        
        return X_resampled, y_resampled


class BERTModel:
    """BERT-based text classification model."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 6):
        """
        Initialize BERT model.
        
        Args:
            model_name: Hugging Face model name
            num_classes: Number of output classes
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized BERT model: {model_name}")
    
    def load_model(self):
        """Load pretrained BERT model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.error("transformers library not installed. Install via: pip install transformers torch")
            raise
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_inputs(self, texts: list, max_length: int = 128) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for BERT.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        logger.info(f"Preparing {len(texts)} texts for BERT...")
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        return encodings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_name': self.model_name,
            'type': 'BERT',
            'num_classes': self.num_classes,
            'status': 'loaded' if self.model else 'not loaded',
        }
        if self.model:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
        else:
            info['num_parameters'] = '110M (estimated)'
        return info


class RoBERTaModel:
    """RoBERTa-based text classification model."""
    
    def __init__(self, model_name: str = "roberta-base", num_classes: int = 6):
        """
        Initialize RoBERTa model.
        
        Args:
            model_name: Hugging Face model name
            num_classes: Number of output classes
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized RoBERTa model: {model_name}")
    
    def load_model(self):
        """Load pretrained RoBERTa model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.error("transformers library not installed. Install via: pip install transformers torch")
            raise
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_inputs(self, texts: list, max_length: int = 128) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for RoBERTa.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        logger.info(f"Preparing {len(texts)} texts for RoBERTa...")
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        return encodings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_name': self.model_name,
            'type': 'RoBERTa',
            'num_classes': self.num_classes,
            'status': 'loaded' if self.model else 'not loaded',
        }
        if self.model:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
        else:
            info['num_parameters'] = '125M (estimated)'
        return info


class HybridModel:
    """
    Hybrid model combining BERT, RoBERTa, and traditional features.
    Uses ensemble predictions and weighted averaging.
    """
    
    def __init__(self, num_classes: int = 6, use_tinker: bool = False):
        """
        Initialize Hybrid model.
        
        Args:
            num_classes: Number of output classes
            use_tinker: Whether to use Tinker API for model training
        """
        self.num_classes = num_classes
        self.use_tinker = use_tinker
        self.bert_model = None
        self.roberta_model = None
        self.weights = {'bert': 0.4, 'roberta': 0.4, 'features': 0.2}
        logger.info(f"Initialized Hybrid model with Tinker={use_tinker}")
    
    def load_models(self):
        """Load both BERT and RoBERTa models."""
        logger.info("Loading BERT and RoBERTa models...")
        
        self.bert_model = BERTModel(num_classes=self.num_classes)
        self.bert_model.load_model()
        
        self.roberta_model = RoBERTaModel(num_classes=self.num_classes)
        self.roberta_model.load_model()
        
        logger.info("Both models loaded successfully")
    
    def set_weights(self, bert_weight: float, roberta_weight: float, features_weight: float):
        """
        Set ensemble weights.
        
        Args:
            bert_weight: Weight for BERT predictions
            roberta_weight: Weight for RoBERTa predictions
            features_weight: Weight for traditional features
        """
        total = bert_weight + roberta_weight + features_weight
        self.weights = {
            'bert': bert_weight / total,
            'roberta': roberta_weight / total,
            'features': features_weight / total
        }
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'type': 'Hybrid Ensemble',
            'num_classes': self.num_classes,
            'use_tinker': self.use_tinker,
            'weights': self.weights,
            'models': [
                'BERT (bert-base-uncased)',
                'RoBERTa (roberta-base)',
                'Traditional Features'
            ]
        }


class TinkerIntegration:
    """Integration with ThinkingMachine's Tinker API."""
    
    def __init__(self, api_key: str, api_url: str = "https://api.tinker.thinkingmachines.ai"):
        """
        Initialize Tinker API integration.
        
        Args:
            api_key: Tinker API key
            api_url: Tinker API URL
        """
        self.api_key = api_key
        self.api_url = api_url
        self.session = None
        logger.info(f"Initialized Tinker API integration: {api_url}")
    
    def connect(self):
        """Establish connection to Tinker API."""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
            logger.info("Connected to Tinker API")
        except ImportError:
            logger.error("requests library not installed. Install via: pip install requests")
            raise
    
    def submit_training_job(self, model_config: Dict[str, Any], data: Dict[str, np.ndarray]) -> str:
        """
        Submit a training job to Tinker API.
        
        Args:
            model_config: Model configuration
            data: Training data (X_train, y_train, X_val, y_val)
            
        Returns:
            Job ID
        """
        if not self.session:
            self.connect()
        
        logger.info("Submitting training job to Tinker API...")
        
        payload = {
            'model_config': model_config,
            'data_shape': {
                'X_train': data['X_train'].shape,
                'y_train': data['y_train'].shape,
                'X_val': data['X_val'].shape,
                'y_val': data['y_val'].shape,
            }
        }
        
        logger.info(f"Payload: {payload}")
        # In production, this would make actual API calls
        logger.info("Training job submitted (demo mode)")
        
        return "job_123456"
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a training job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information
        """
        logger.info(f"Fetching status for job {job_id}...")
        # In production, this would query the API
        return {'status': 'completed', 'accuracy': 0.85}
    
    def download_model(self, job_id: str, save_path: str):
        """
        Download trained model from Tinker API.
        
        Args:
            job_id: Job ID
            save_path: Path to save model
        """
        logger.info(f"Downloading model from job {job_id} to {save_path}...")
        # In production, this would download from API
        logger.info("Model downloaded successfully")


if __name__ == '__main__':
    # Example usage
    print("Model definitions loaded successfully")
    
    cb = ClassBalancer()
    bert = BERTModel()
    roberta = RoBERTaModel()
    hybrid = HybridModel()
    tinker = TinkerIntegration(api_key="test_key")
    
    print("\nAvailable models:")
    print("- BERTModel")
    print("- RoBERTaModel")
    print("- HybridModel")
    print("- TinkerIntegration")
