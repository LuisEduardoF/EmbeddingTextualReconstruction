"""
Generate embeddings with categorical field information for categorical inverter training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle
from bertimbau_embedder import BERTimbauEmbedder
from preprocessing.data_loader import LaborDataLoader
from preprocessing.categorical_preprocessor import CategoricalDataPreprocessor


def generate_embeddings_with_categorical(
    data_path: str,
    output_dir: str,
    max_samples: int = None,
    batch_size: int = 32
):
    """
    Generate embeddings along with categorical field encodings.
    
    Args:
        data_path: Path to input parquet file
        output_dir: Directory to save embeddings
        max_samples: Maximum samples to process
        batch_size: Batch size for processing
    """
    print("="*70)
    print("GENERATING EMBEDDINGS WITH CATEGORICAL FIELDS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    loader = LaborDataLoader(data_path)
    train_df, test_df = loader.prepare_for_embedding(max_samples=max_samples)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize embedder
    print("\nInitializing BERTimbau embedder...")
    embedder = BERTimbauEmbedder()
    
    # Create categorical preprocessor
    print("\nBuilding categorical vocabularies...")
    preprocessor = CategoricalDataPreprocessor()
    
    # Build vocabularies from training data only
    vocab_sizes = preprocessor.build_vocabularies(
        train_df, 
        min_frequency=2,
        max_vocab_size=None
    )
    
    # Print statistics
    preprocessor.print_summary()
    
    # Save preprocessor
    os.makedirs('data/processed', exist_ok=True)
    preprocessor.save('data/processed/categorical_preprocessor.pkl')
    
    # Process training set
    print("\n" + "="*70)
    print("PROCESSING TRAINING SET")
    print("="*70)
    
    # Generate embeddings
    train_embeddings, train_texts, train_labels = embedder.process_dataframe(
        train_df, 
        batch_size=batch_size
    )
    
    # Encode categorical fields
    print("\nEncoding categorical fields...")
    train_categorical = preprocessor.encode_fields(train_df)
    
    # Verify encoding
    print("\nCategorical field shapes:")
    for field, encoded in train_categorical.items():
        print(f"  {field}: {encoded.shape}")
    
    # Save training data
    train_data = {
        'embeddings': train_embeddings,
        'texts': train_texts,
        'labels': train_labels,
        'categorical_fields': train_categorical,
        'model_name': embedder.model_name,
        'embedding_dim': train_embeddings.shape[1],
        'metadata': {
            'split': 'train',
            'size': len(train_df),
            'vocab_sizes': vocab_sizes
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_embeddings.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\n✓ Saved training embeddings to {train_path}")
    
    # Process test set
    print("\n" + "="*70)
    print("PROCESSING TEST SET")
    print("="*70)
    
    # Generate embeddings
    test_embeddings, test_texts, test_labels = embedder.process_dataframe(
        test_df, 
        batch_size=batch_size
    )
    
    # Encode categorical fields (using same preprocessor)
    print("\nEncoding categorical fields...")
    test_categorical = preprocessor.encode_fields(test_df)
    
    # Verify encoding
    print("\nCategorical field shapes:")
    for field, encoded in test_categorical.items():
        print(f"  {field}: {encoded.shape}")
    
    # Save test data
    test_data = {
        'embeddings': test_embeddings,
        'texts': test_texts,
        'labels': test_labels,
        'categorical_fields': test_categorical,
        'model_name': embedder.model_name,
        'embedding_dim': test_embeddings.shape[1],
        'metadata': {
            'split': 'test',
            'size': len(test_df),
            'vocab_sizes': vocab_sizes
        }
    }
    
    test_path = os.path.join(output_dir, 'test_embeddings.pkl')
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"\n✓ Saved test embeddings to {test_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)
    print(f"\nTrain embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")
    print(f"\nCategorical fields:")
    for field, size in vocab_sizes.items():
        print(f"  {field}: {size} unique values")
    print(f"\nFiles saved:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - data/processed/categorical_preprocessor.pkl")
    print("="*70)


def verify_embeddings(embedding_path: str):
    """
    Verify that embeddings file contains categorical fields.
    
    Args:
        embedding_path: Path to embeddings file
    """
    print(f"\nVerifying {embedding_path}...")
    
    with open(embedding_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Keys in file: {list(data.keys())}")
    print(f"Embeddings shape: {data['embeddings'].shape}")
    
    if 'categorical_fields' in data:
        print("✓ Categorical fields found:")
        for field, values in data['categorical_fields'].items():
            print(f"  {field}: {values.shape}, unique values: {len(np.unique(values))}")
    else:
        print("✗ Categorical fields NOT found in embeddings file")
    
    if 'metadata' in data:
        print(f"Metadata: {data['metadata']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate embeddings with categorical field information'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='updated_dataset_preprocessed.parquet_new.gzip',
        help='Path to input parquet file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/embeddings',
        help='Directory to save embeddings'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help='Maximum samples to process (None for all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--verify',
        type=str,
        default=None,
        help='Path to embeddings file to verify'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_embeddings(args.verify)
    else:
        generate_embeddings_with_categorical(
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
