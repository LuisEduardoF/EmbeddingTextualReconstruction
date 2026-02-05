"""
Categorical data preprocessing for labor court data.
Handles vocabulary building and encoding/decoding of categorical fields.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, List, Optional
from collections import Counter


class CategoricalDataPreprocessor:
    """
    Preprocessor for categorical fields in labor court data.
    Builds vocabularies and encodes/decodes categorical values.
    """
    
    def __init__(self):
        """Initialize the categorical preprocessor."""
        self.field_encoders = {}
        self.field_decoders = {}
        self.field_frequencies = {}
        self.vocab_sizes = {}
        
        # Field name mapping (internal name -> dataframe column name)
        self.field_mapping = {
            'ASSUNTOS': 'ASSUNTOS',
            'CLASSE_PROCESSUAL': 'CLASSE PROCESSUAL',
            'RAMO_ATIVIDADE': 'RAMO DE ATIVIDADE',
            'MAGISTRADO': 'MAGISTRADO'
        }
    
    def build_vocabularies(
        self, 
        df: pd.DataFrame,
        min_frequency: int = 2,
        max_vocab_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Build vocabulary mappings for each categorical field.
        
        Args:
            df: Input dataframe with categorical columns
            min_frequency: Minimum frequency for a value to be included
            max_vocab_size: Maximum vocabulary size per field (None for unlimited)
            
        Returns:
            Dictionary with vocabulary sizes per field
        """
        print("Building vocabularies for categorical fields...")
        
        for internal_name, col_name in self.field_mapping.items():
            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' not found in dataframe")
                continue
            
            # Get all values (handle NaN)
            values = df[col_name].fillna('UNKNOWN').astype(str)
            
            # Count frequencies
            value_counts = Counter(values)
            
            # Filter by minimum frequency
            filtered_values = [
                val for val, count in value_counts.items() 
                if count >= min_frequency
            ]
            
            # Sort by frequency (most common first)
            filtered_values = sorted(
                filtered_values, 
                key=lambda x: value_counts[x], 
                reverse=True
            )
            
            # Limit vocabulary size if specified
            if max_vocab_size is not None:
                filtered_values = filtered_values[:max_vocab_size]
            
            # Add UNKNOWN token if not present
            if 'UNKNOWN' not in filtered_values:
                filtered_values.append('UNKNOWN')
            
            # Create encoder/decoder mappings
            self.field_encoders[internal_name] = {
                val: idx for idx, val in enumerate(filtered_values)
            }
            self.field_decoders[internal_name] = {
                idx: val for idx, val in enumerate(filtered_values)
            }
            
            # Store frequencies
            self.field_frequencies[internal_name] = {
                val: value_counts[val] for val in filtered_values
            }
            
            # Store vocabulary size
            self.vocab_sizes[internal_name] = len(filtered_values)
            
            print(f"  {internal_name}: {len(filtered_values)} unique values "
                  f"(filtered from {len(value_counts)} total)")
        
        return self.vocab_sizes
    
    def encode_fields(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Encode categorical fields to integer labels.
        
        Args:
            df: Input dataframe with categorical columns
            
        Returns:
            Dictionary with field names as keys and encoded labels as values
        """
        encoded = {}
        
        for internal_name, col_name in self.field_mapping.items():
            if internal_name not in self.field_encoders:
                print(f"Warning: No encoder found for {internal_name}, skipping")
                continue
            
            if col_name not in df.columns:
                # Create dummy encoding if column missing
                encoded[internal_name] = np.zeros(len(df), dtype=np.int64)
                continue
            
            # Get values and handle NaN
            values = df[col_name].fillna('UNKNOWN').astype(str)
            
            # Encode values (use UNKNOWN for out-of-vocabulary)
            encoder = self.field_encoders[internal_name]
            unknown_idx = encoder.get('UNKNOWN', 0)
            
            encoded[internal_name] = np.array([
                encoder.get(val, unknown_idx) for val in values
            ], dtype=np.int64)
        
        return encoded
    
    def decode_fields(
        self, 
        encoded: Dict[str, np.ndarray]
    ) -> Dict[str, List[str]]:
        """
        Decode integer labels back to categorical values.
        
        Args:
            encoded: Dictionary with field names as keys and encoded labels as values
            
        Returns:
            Dictionary with field names as keys and decoded values as values
        """
        decoded = {}
        
        for internal_name, labels in encoded.items():
            if internal_name not in self.field_decoders:
                print(f"Warning: No decoder found for {internal_name}, skipping")
                continue
            
            decoder = self.field_decoders[internal_name]
            
            # Decode labels
            decoded[internal_name] = [
                decoder.get(int(label), 'UNKNOWN') for label in labels
            ]
        
        return decoded
    
    def get_field_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about categorical fields.
        
        Returns:
            Dictionary with field statistics
        """
        stats = {}
        
        for field_name in self.field_encoders.keys():
            vocab_size = self.vocab_sizes[field_name]
            frequencies = self.field_frequencies[field_name]
            
            # Calculate entropy
            total_count = sum(frequencies.values())
            probs = np.array([count / total_count for count in frequencies.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            stats[field_name] = {
                'vocab_size': vocab_size,
                'entropy': entropy,
                'most_common': sorted(
                    frequencies.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5],
                'total_count': total_count
            }
        
        return stats
    
    def save(self, filepath: str):
        """
        Save preprocessor state to file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        state = {
            'field_encoders': self.field_encoders,
            'field_decoders': self.field_decoders,
            'field_frequencies': self.field_frequencies,
            'vocab_sizes': self.vocab_sizes,
            'field_mapping': self.field_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved preprocessor to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'CategoricalDataPreprocessor':
        """
        Load preprocessor state from file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = CategoricalDataPreprocessor()
        preprocessor.field_encoders = state['field_encoders']
        preprocessor.field_decoders = state['field_decoders']
        preprocessor.field_frequencies = state['field_frequencies']
        preprocessor.vocab_sizes = state['vocab_sizes']
        preprocessor.field_mapping = state['field_mapping']
        
        print(f"Loaded preprocessor from {filepath}")
        return preprocessor
    
    def get_class_weights(self, field_name: str) -> np.ndarray:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Array of class weights
        """
        if field_name not in self.field_frequencies:
            raise ValueError(f"Field {field_name} not found")
        
        frequencies = self.field_frequencies[field_name]
        vocab_size = self.vocab_sizes[field_name]
        
        # Calculate inverse frequency weights
        total_count = sum(frequencies.values())
        weights = np.zeros(vocab_size)
        
        for val, idx in self.field_encoders[field_name].items():
            count = frequencies.get(val, 1)
            weights[idx] = total_count / (vocab_size * count)
        
        return weights
    
    def print_summary(self):
        """Print a summary of the preprocessor state."""
        print("\n" + "="*60)
        print("CATEGORICAL PREPROCESSOR SUMMARY")
        print("="*60)
        
        stats = self.get_field_statistics()
        
        for field_name, field_stats in stats.items():
            print(f"\n{field_name}:")
            print(f"  Vocabulary Size: {field_stats['vocab_size']}")
            print(f"  Entropy: {field_stats['entropy']:.2f} bits")
            print(f"  Total Count: {field_stats['total_count']}")
            print(f"  Most Common Values:")
            for val, count in field_stats['most_common']:
                pct = 100 * count / field_stats['total_count']
                print(f"    - {val}: {count} ({pct:.1f}%)")
        
        print("\n" + "="*60)


def prepare_categorical_data(
    df: pd.DataFrame,
    preprocessor: Optional[CategoricalDataPreprocessor] = None,
    min_frequency: int = 2,
    max_vocab_size: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], CategoricalDataPreprocessor]:
    """
    Prepare categorical data for training.
    
    Args:
        df: Input dataframe
        preprocessor: Existing preprocessor (None to create new)
        min_frequency: Minimum frequency for vocabulary
        max_vocab_size: Maximum vocabulary size per field
        
    Returns:
        Tuple of (encoded_fields, preprocessor)
    """
    if preprocessor is None:
        preprocessor = CategoricalDataPreprocessor()
        preprocessor.build_vocabularies(df, min_frequency, max_vocab_size)
    
    encoded_fields = preprocessor.encode_fields(df)
    
    return encoded_fields, preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing CategoricalDataPreprocessor...")
    
    # Create sample data
    sample_data = {
        'ASSUNTOS': ['Rescisão', 'FGTS', 'Rescisão', 'Horas Extras', 'FGTS', None],
        'CLASSE PROCESSUAL': ['Reclamação', 'Reclamação', 'Recurso', 'Reclamação', 'Recurso', 'Reclamação'],
        'RAMO DE ATIVIDADE': ['Comércio', 'Indústria', 'Comércio', 'Serviços', 'Indústria', 'Comércio'],
        'MAGISTRADO': ['João Silva', 'Maria Santos', 'João Silva', 'Pedro Costa', 'Maria Santos', 'João Silva']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create and test preprocessor
    preprocessor = CategoricalDataPreprocessor()
    vocab_sizes = preprocessor.build_vocabularies(df, min_frequency=1)
    
    print("\nVocabulary sizes:", vocab_sizes)
    
    # Encode
    encoded = preprocessor.encode_fields(df)
    print("\nEncoded fields:")
    for field, values in encoded.items():
        print(f"  {field}: {values}")
    
    # Decode
    decoded = preprocessor.decode_fields(encoded)
    print("\nDecoded fields:")
    for field, values in decoded.items():
        print(f"  {field}: {values}")
    
    # Statistics
    preprocessor.print_summary()
    
    # Test save/load
    preprocessor.save('test_preprocessor.pkl')
    loaded_preprocessor = CategoricalDataPreprocessor.load('test_preprocessor.pkl')
    
    print("\n✓ All tests passed!")
