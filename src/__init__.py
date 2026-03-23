"""
Fake News Detection - Model Training Package

Submodules:
- data_processor: Data loading, cleaning, and encoding
- models: BERT, RoBERTa, and Hybrid model definitions
- training_utils: Training utilities and helpers
- config: Project configuration
"""

__version__ = "1.0.0"
__author__ = "Hamdan"

from .data_processor import DataProcessor
from .models import (
    ClassBalancer,
    BERTModel,
    RoBERTaModel,
    HybridModel,
    TinkerIntegration
)
from .training_utils import DataSplitter, MetricsCalculator, ModelTrainer

__all__ = [
    'DataProcessor',
    'ClassBalancer',
    'BERTModel',
    'RoBERTaModel',
    'HybridModel',
    'TinkerIntegration',
    'DataSplitter',
    'MetricsCalculator',
    'ModelTrainer',
]
