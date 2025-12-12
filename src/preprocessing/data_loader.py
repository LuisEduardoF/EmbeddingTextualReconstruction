"""
Data loading and preprocessing for embedding inversion attack experiment.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


class LaborDataLoader:
    """Load and preprocess labor court data for embedding inversion experiments."""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the parquet file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the parquet dataset."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def create_text_field(self) -> pd.DataFrame:
        """
        Create a combined text field from relevant columns for embedding generation.
        This combines sensitive information that we'll try to reconstruct.
        """
        print("Creating combined text field...")
        
        text_components = []
        
        # Add relevant textual fields
        if 'ASSUNTOS' in self.df.columns:
            text_components.append(self.df['ASSUNTOS'].fillna(''))
        
        if 'CLASSE PROCESSUAL' in self.df.columns:
            text_components.append(self.df['CLASSE PROCESSUAL'].fillna(''))
            
        if 'RAMO DE ATIVIDADE' in self.df.columns:
            text_components.append(self.df['RAMO DE ATIVIDADE'].fillna(''))
        
        if 'MAGISTRADO' in self.df.columns:
            text_components.append(self.df['MAGISTRADO'].fillna(''))
            
        # Combine all components
        self.df['combined_text'] = text_components[0]
        for component in text_components[1:]:
            self.df['combined_text'] = self.df['combined_text'] + ' | ' + component
        
        # Clean up
        self.df['combined_text'] = self.df['combined_text'].str.strip()
        
        print(f"Created combined text field. Sample length: {len(self.df['combined_text'].iloc[0])}")
        return self.df
    
    def prepare_for_embedding(
        self, 
        max_samples: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for embedding generation and train/test split.
        
        Args:
            max_samples: Maximum number of samples to use (for faster experimentation)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        if 'combined_text' not in self.df.columns:
            self.create_text_field()
        
        # Filter out empty texts
        df_filtered = self.df[self.df['combined_text'].str.len() > 10].copy()
        print(f"Filtered to {len(df_filtered)} records with meaningful text")
        
        # Sample if requested
        if max_samples and len(df_filtered) > max_samples:
            df_filtered = df_filtered.sample(n=max_samples, random_state=random_state)
            print(f"Sampled {max_samples} records for faster experimentation")
        
        # Train/test split
        train_df, test_df = train_test_split(
            df_filtered,
            test_size=test_size,
            random_state=random_state,
            stratify=df_filtered['label']
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        print(f"Label distribution in train: {train_df['label'].value_counts().to_dict()}")
        
        return train_df, test_df
    
    def get_sensitive_fields(self) -> List[str]:
        """Return list of sensitive fields that should be protected."""
        return [
            'MAGISTRADO',
            'ASSUNTOS',
            'DOCUMENTOS DOS RECLAMANTES',
            'DOCUMENTOS DAS RECLAMADAS',
            'OAB',
            'RAMO DE ATIVIDADE'
        ]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important keywords from text for reconstruction evaluation.
        
        Args:
            text: Input text
            top_n: Number of top keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on frequency
        words = text.lower().split()
        # Remove very short words and common stopwords
        stopwords = {'de', 'da', 'do', 'das', 'dos', 'a', 'o', 'e', 'em', 'para', 'com', 'por'}
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        return [word for word, _ in word_counts.most_common(top_n)]


if __name__ == "__main__":
    # Test the data loader
    loader = LaborDataLoader("updated_dataset_preprocessed.parquet_new.gzip")
    train_df, test_df = loader.prepare_for_embedding(max_samples=1000)
    
    print("\nSample combined text:")
    print(train_df['combined_text'].iloc[0][:200])
    
    print("\nSensitive fields:")
    print(loader.get_sensitive_fields())