"""
Data processing utilities for Fake News Detection project.
Handles loading, cleaning, and encoding of LIAR2 dataset.
"""

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataProcessor:
    """Main class for data processing pipeline."""
    
    # Truthfulness labels from LIAR2 dataset
    TRUTHFULNESS_LABELS = [
        'true',
        'mostly-true',
        'half-true',
        'barely-true',
        'false',
        'pants-fire'
    ]
    
    def __init__(self):
        """Initialize data processor with label encoder."""
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.TRUTHFULNESS_LABELS)
        self.vocab = {}
        self.vocab_size = 0
        # Initialize stopwords set
        from nltk.corpus import stopwords
        self.stopwords_set = set(stopwords.words('english'))
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Load LIAR2 dataset from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def clean_text(self, text: str, remove_stops: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            remove_stops: Whether to remove stopwords
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if remove_stops:
            words = text.split()
            words = [w for w in words if w not in self.stopwords_set]
            text = ' '.join(words)
        
        return text
    
    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Encode truthfulness labels to numeric values.
        
        Args:
            labels: Series of label strings
            
        Returns:
            Numpy array of encoded labels
        """
        # Ensure labels are strings, normalize to lowercase and strip whitespace
        labels = labels.astype(str).str.lower().str.strip()
        
        try:
            # Encode labels
            encoded = self.label_encoder.transform(labels)
            logger.info(f"Encoded {len(encoded)} labels")
            return encoded
        except ValueError as e:
            logger.error(f"Error encoding labels: {e}")
            logger.error(f"Available classes: {self.label_encoder.classes_}")
            logger.error(f"Unique labels in data: {labels.unique()}")
            raise
    
    def build_vocabulary(self, texts: pd.Series, min_freq: int = 1) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: Series of text documents
            min_freq: Minimum frequency for a word to be included
            
        Returns:
            Dictionary mapping words to indices
        """
        logger.info("Building vocabulary...")
        word_freq = {}
        
        for text in texts:
            tokens = text.split()
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        vocab = {word: idx for idx, (word, freq) in enumerate(
            sorted(
                ((word, freq) for word, freq in word_freq.items() if freq >= min_freq),
                key=lambda x: x[1],
                reverse=True
            )
        )}
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        return vocab
    
    def texts_to_sequences(self, texts: pd.Series) -> List[List[int]]:
        """
        Convert texts to sequences of word indices.
        
        Args:
            texts: Series of text documents
            
        Returns:
            List of sequences
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        sequences = []
        for text in texts:
            tokens = text.split()
            sequence = [self.vocab[token] for token in tokens if token in self.vocab]
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], maxlen: int = 100) -> np.ndarray:
        """
        Pad sequences to fixed length.
        
        Args:
            sequences: List of sequences
            maxlen: Maximum sequence length
            
        Returns:
            Padded sequence array
        """
        padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
        
        for idx, seq in enumerate(sequences):
            length = min(len(seq), maxlen)
            padded[idx, :length] = seq[:length]
        
        return padded
    
    def process_pipeline(self, df: pd.DataFrame, 
                        text_column: str = 'statement',
                        label_column: str = 'label',
                        remove_stops: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Complete processing pipeline: clean text, remove nulls, encode labels.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            remove_stops: Whether to remove stopwords
            
        Returns:
            Tuple of (processed DataFrame, encoded labels)
        """
        logger.info("Starting data processing pipeline...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Remove rows with missing values in key columns
        df = df.dropna(subset=[text_column, label_column])
        logger.info(f"Removed nulls: {len(df)} records remaining")
        
        # Clean text with optional stopword removal
        df['text_cleaned'] = df[text_column].apply(lambda x: self.clean_text(x, remove_stops=remove_stops))
        
        # Remove empty texts
        df = df[df['text_cleaned'].str.len() > 0]
        logger.info(f"Removed empty texts: {len(df)} records remaining")
        
        # Normalize labels (lowercase and strip whitespace)
        df[label_column] = df[label_column].astype(str).str.lower().str.strip()
        
        # Encode labels
        encoded_labels = self.encode_labels(df[label_column])
        
        logger.info("Data processing pipeline completed successfully")

        return df, encoded_labels

    @staticmethod
    def prepare_tinker_datum(text: str, label: str, tokenizer: Any,
                             prompt_template: str) -> Any:
        """
        Convert a single text/label pair into a Tinker Datum for cross-entropy training.

        The prompt tokens receive weight 0 (model is not trained on them).
        The completion tokens receive weight 1 (model is trained to predict them).
        Tokens are shifted by one position for next-token prediction.

        Args:
            text: Input statement text
            label: Truthfulness label string (e.g. 'false', 'half-true')
            tokenizer: Tinker tokenizer from training_client.get_tokenizer()
            prompt_template: Format string with {text} placeholder

        Returns:
            types.Datum ready for forward_backward()
        """
        from tinker import types, TensorData

        prompt = prompt_template.format(text=text)
        completion = f" {label}"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

        # Weight 0 = don't train on prompt, 1 = train on completion
        weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
        all_tokens = prompt_tokens + completion_tokens

        # Shift for next-token prediction: input[t] → target[t+1]
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]
        weights = weights[1:]  # drop first weight (aligns with shifted targets)

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(
                weights=TensorData(data=weights, dtype='float32'),
                target_tokens=TensorData(data=target_tokens, dtype='int64'),
            )
        )

    @staticmethod
    def prepare_tinker_dataset(texts: List[str], labels: List[str],
                               tokenizer: Any, prompt_template: str) -> List[Any]:
        """
        Convert a list of texts and labels into Tinker Datums.

        Args:
            texts: List of input statements
            labels: List of label strings matching TRUTHFULNESS_LABELS
            tokenizer: Tinker tokenizer
            prompt_template: Format string with {text} placeholder

        Returns:
            List of types.Datum objects
        """
        datums = []
        for text, label in zip(texts, labels):
            datum = DataProcessor.prepare_tinker_datum(
                text, label, tokenizer, prompt_template
            )
            datums.append(datum)
        logger.info(f"Prepared {len(datums)} Tinker datums")
        return datums
