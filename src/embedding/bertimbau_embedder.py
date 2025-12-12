"""
Generate embeddings using BERTimbau model for embedding inversion experiments.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional
from tqdm import tqdm
import pickle


class BERTimbauEmbedder:
    """Generate embeddings using BERTimbau model."""
    
    def __init__(
        self, 
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize BERTimbau embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 16,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings using [CLS] token.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract [CLS] token embeddings (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(cls_embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape[1]}")
        
        return embeddings
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: np.ndarray,
        output_path: str,
        metadata: Optional[dict] = None
    ):
        """
        Save embeddings and associated data.
        
        Args:
            embeddings: Embedding vectors
            texts: Original texts
            labels: Labels for classification
            output_path: Path to save the data
            metadata: Additional metadata to save
        """
        data = {
            'embeddings': embeddings,
            'texts': texts,
            'labels': labels,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'metadata': metadata or {}
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved embeddings to {output_path}")
    
    @staticmethod
    def load_embeddings(input_path: str) -> dict:
        """
        Load saved embeddings.
        
        Args:
            input_path: Path to load from
            
        Returns:
            Dictionary with embeddings and metadata
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {len(data['embeddings'])} embeddings from {input_path}")
        return data
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'combined_text',
        label_column: str = 'label',
        batch_size: int = 16
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Process a dataframe to generate embeddings.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings, texts, labels)
        """
        texts = df[text_column].tolist()
        labels = df[label_column].values
        
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        
        return embeddings, texts, labels


def generate_embeddings_from_data(
    data_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 16
):
    """
    Complete pipeline to generate embeddings from dataset.
    
    Args:
        data_path: Path to input parquet file
        output_dir: Directory to save embeddings
        max_samples: Maximum samples to process
        batch_size: Batch size for processing
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from preprocessing.data_loader import LaborDataLoader
    
    # Load data
    loader = LaborDataLoader(data_path)
    train_df, test_df = loader.prepare_for_embedding(max_samples=max_samples)
    
    # Initialize embedder
    embedder = BERTimbauEmbedder()
    
    # Process train set
    print("\n=== Processing Training Set ===")
    train_embeddings, train_texts, train_labels = embedder.process_dataframe(
        train_df, batch_size=batch_size
    )
    embedder.save_embeddings(
        train_embeddings,
        train_texts,
        train_labels,
        f"{output_dir}/train_embeddings.pkl",
        metadata={'split': 'train', 'size': len(train_df)}
    )
    
    # Process test set
    print("\n=== Processing Test Set ===")
    test_embeddings, test_texts, test_labels = embedder.process_dataframe(
        test_df, batch_size=batch_size
    )
    embedder.save_embeddings(
        test_embeddings,
        test_texts,
        test_labels,
        f"{output_dir}/test_embeddings.pkl",
        metadata={'split': 'test', 'size': len(test_df)}
    )
    
    print("\n=== Embedding Generation Complete ===")
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")


if __name__ == "__main__":
    # Generate embeddings
    generate_embeddings_from_data(
        data_path="updated_dataset_preprocessed.parquet_new.gzip",
        output_dir="data/embeddings",
        max_samples=5000,  # Start with 5k samples for faster experimentation
        batch_size=32
    )