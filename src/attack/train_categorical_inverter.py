"""
Training script for the categorical attention inverter model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attack.categorical_inverter_model import CategoricalAttentionInverter, CategoricalInversionLoss
from preprocessing.categorical_preprocessor import CategoricalDataPreprocessor


class CategoricalInversionDataset(Dataset):
    """Dataset for training categorical field inversion models."""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        categorical_labels: Dict[str, np.ndarray]
    ):
        """
        Initialize dataset.
        
        Args:
            embeddings: Embedding vectors [N, embedding_dim]
            categorical_labels: Dictionary {field_name: labels [N]}
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = {
            field: torch.LongTensor(labels) 
            for field, labels in categorical_labels.items()
        }
        
        # Verify all fields have same length
        lengths = [len(labels) for labels in self.labels.values()]
        assert all(l == len(self.embeddings) for l in lengths), \
            "All fields must have same number of samples as embeddings"
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'labels': {field: labels[idx] for field, labels in self.labels.items()}
        }


class CategoricalInverterTrainer:
    """Trainer for categorical attention inverter models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        field_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1
    ):
        """
        Initialize trainer.
        
        Args:
            model: Categorical inverter model
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            field_weights: Field importance weights for loss
            label_smoothing: Label smoothing factor
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function
        self.criterion = CategoricalInversionLoss(
            field_weights=field_weights,
            label_smoothing=label_smoothing
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_field_losses': [],
            'val_field_losses': [],
            'train_field_accuracies': [],
            'val_field_accuracies': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, field_losses, field_accuracies)
        """
        self.model.train()
        total_loss = 0
        accumulated_field_losses = {field: 0.0 for field in self.model.field_names}
        accumulated_field_accs = {field: 0.0 for field in self.model.field_names}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            embeddings = batch['embedding'].to(self.device)
            target_labels = {
                field: labels.to(self.device) 
                for field, labels in batch['labels'].items()
            }
            
            # Forward pass
            predictions, _ = self.model(embeddings)
            
            # Calculate loss
            loss, field_losses, field_accs = self.criterion(predictions, target_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            for field in self.model.field_names:
                accumulated_field_losses[field] += field_losses[field]
                accumulated_field_accs[field] += field_accs[field]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_acc': np.mean(list(field_accs.values()))
            })
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Calculate averages
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_field_losses = {
            field: accumulated_field_losses[field] / num_batches 
            for field in self.model.field_names
        }
        avg_field_accs = {
            field: accumulated_field_accs[field] / num_batches 
            for field in self.model.field_names
        }
        
        return avg_loss, avg_field_losses, avg_field_accs
    
    def evaluate(
        self, 
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, field_losses, field_accuracies)
        """
        self.model.eval()
        total_loss = 0
        accumulated_field_losses = {field: 0.0 for field in self.model.field_names}
        accumulated_field_accs = {field: 0.0 for field in self.model.field_names}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                embeddings = batch['embedding'].to(self.device)
                target_labels = {
                    field: labels.to(self.device) 
                    for field, labels in batch['labels'].items()
                }
                
                # Forward pass
                predictions, _ = self.model(embeddings)
                
                # Calculate loss
                loss, field_losses, field_accs = self.criterion(predictions, target_labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                for field in self.model.field_names:
                    accumulated_field_losses[field] += field_losses[field]
                    accumulated_field_accs[field] += field_accs[field]
        
        # Calculate averages
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_field_losses = {
            field: accumulated_field_losses[field] / num_batches 
            for field in self.model.field_names
        }
        avg_field_accs = {
            field: accumulated_field_accs[field] / num_batches 
            for field in self.model.field_names
        }
        
        return avg_loss, avg_field_losses, avg_field_accs
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_dir: str = 'models/attacker/categorical',
        early_stopping_patience: int = 5
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_field_losses, train_field_accs = self.train_epoch(
                train_loader, epoch
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_field_losses'].append(train_field_losses)
            self.history['train_field_accuracies'].append(train_field_accs)
            
            # Validate
            val_loss, val_field_losses, val_field_accs = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_field_losses'].append(val_field_losses)
            self.history['val_field_accuracies'].append(val_field_accs)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("\nField-wise Training Accuracies:")
            for field, acc in train_field_accs.items():
                print(f"  {field}: {acc:.4f}")
            print("\nField-wise Validation Accuracies:")
            for field, acc in val_field_accs.items():
                print(f"  {field}: {acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_path = os.path.join(save_dir, 'best_categorical_inverter.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_field_accuracies': val_field_accs,
                    'history': self.history,
                    'vocab_sizes': self.model.vocab_sizes
                }, model_path)
                print(f"\n✓ Saved best model to {model_path}")
            else:
                patience_counter += 1
                print(f"\nNo improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*70)


def main():
    """Main training function for categorical inverter."""
    
    # Configuration
    EMBEDDING_PATH_TRAIN = "data/embeddings/train_embeddings.pkl"
    EMBEDDING_PATH_TEST = "data/embeddings/test_embeddings.pkl"
    PREPROCESSOR_PATH = "data/processed/categorical_preprocessor.pkl"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    
    print("="*70)
    print("CATEGORICAL ATTENTION INVERTER - TRAINING")
    print("="*70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    with open(EMBEDDING_PATH_TRAIN, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(EMBEDDING_PATH_TEST, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Train samples: {len(train_data['embeddings'])}")
    print(f"Test samples: {len(test_data['embeddings'])}")
    
    embedding_dim = train_data['embeddings'].shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Load or create preprocessor
    if os.path.exists(PREPROCESSOR_PATH):
        print(f"\nLoading preprocessor from {PREPROCESSOR_PATH}...")
        preprocessor = CategoricalDataPreprocessor.load(PREPROCESSOR_PATH)
    else:
        print("\nCreating new preprocessor...")
        from preprocessing.data_loader import LaborDataLoader
        
        # Load original data to build vocabularies
        loader = LaborDataLoader("updated_dataset_preprocessed.parquet_new.gzip")
        train_df, _ = loader.prepare_for_embedding()
        
        preprocessor = CategoricalDataPreprocessor()
        preprocessor.build_vocabularies(train_df, min_frequency=2)
        
        os.makedirs('data/processed', exist_ok=True)
        preprocessor.save(PREPROCESSOR_PATH)
    
    # Print preprocessor summary
    preprocessor.print_summary()
    
    # Encode categorical fields
    print("\nEncoding categorical fields...")
    
    # For train data
    train_df_temp = {
        'ASSUNTOS': train_data.get('assuntos', ['']*len(train_data['embeddings'])),
        'CLASSE PROCESSUAL': train_data.get('classe_processual', ['']*len(train_data['embeddings'])),
        'RAMO DE ATIVIDADE': train_data.get('ramo_atividade', ['']*len(train_data['embeddings'])),
        'MAGISTRADO': train_data.get('magistrado', ['']*len(train_data['embeddings']))
    }
    
    # Check if categorical data is stored in the embeddings file
    if 'categorical_fields' in train_data:
        train_categorical = train_data['categorical_fields']
    else:
        print("Warning: Categorical fields not found in embeddings file.")
        print("You may need to regenerate embeddings with categorical field information.")
        # Create dummy data for testing
        train_categorical = {
            'ASSUNTOS': np.zeros(len(train_data['embeddings']), dtype=np.int64),
            'CLASSE_PROCESSUAL': np.zeros(len(train_data['embeddings']), dtype=np.int64),
            'RAMO_ATIVIDADE': np.zeros(len(train_data['embeddings']), dtype=np.int64),
            'MAGISTRADO': np.zeros(len(train_data['embeddings']), dtype=np.int64)
        }
    
    if 'categorical_fields' in test_data:
        test_categorical = test_data['categorical_fields']
    else:
        test_categorical = {
            'ASSUNTOS': np.zeros(len(test_data['embeddings']), dtype=np.int64),
            'CLASSE_PROCESSUAL': np.zeros(len(test_data['embeddings']), dtype=np.int64),
            'RAMO_ATIVIDADE': np.zeros(len(test_data['embeddings']), dtype=np.int64),
            'MAGISTRADO': np.zeros(len(test_data['embeddings']), dtype=np.int64)
        }
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CategoricalInversionDataset(
        train_data['embeddings'],
        train_categorical
    )
    
    test_dataset = CategoricalInversionDataset(
        test_data['embeddings'],
        test_categorical
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating categorical attention inverter model...")
    model = CategoricalAttentionInverter(
        embedding_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        assuntos_vocab_size=preprocessor.vocab_sizes['ASSUNTOS'],
        classe_vocab_size=preprocessor.vocab_sizes['CLASSE_PROCESSUAL'],
        ramo_vocab_size=preprocessor.vocab_sizes['RAMO_ATIVIDADE'],
        magistrado_vocab_size=preprocessor.vocab_sizes['MAGISTRADO']
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = CategoricalInverterTrainer(
        model=model,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        save_dir='models/attacker/categorical'
    )
    
    # Save final results
    os.makedirs('results', exist_ok=True)
    with open('results/categorical_training_results.pkl', 'wb') as f:
        pickle.dump({
            'history': trainer.history,
            'vocab_sizes': preprocessor.vocab_sizes,
            'model_config': {
                'embedding_dim': embedding_dim,
                'hidden_dim': HIDDEN_DIM,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS
            }
        }, f)
    
    print("\n✓ Training results saved to results/categorical_training_results.pkl")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
