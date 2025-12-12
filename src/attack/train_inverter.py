"""
Training script for the embedding inverter model.
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
from transformers import AutoTokenizer

from .inverter_model import EmbeddingInverter, LSTMInverter, AttentionInverter


class EmbeddingInversionDataset(Dataset):
    """Dataset for training embedding inversion models."""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize dataset.
        
        Args:
            embeddings: Embedding vectors
            texts: Original texts
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
        print("Tokenizing texts...")
        self.tokenized = []
        for text in tqdm(texts):
            encoded = tokenizer.encode(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.tokenized.append(encoded.squeeze(0))
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'token_ids': self.tokenized[idx],
            'text': self.texts[idx]
        }


class InverterTrainer:
    """Trainer for embedding inverter models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: Inverter model
            tokenizer: Tokenizer
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.tokenizer = tokenizer
        
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            embeddings = batch['embedding'].to(self.device)
            target_ids = batch['token_ids'].to(self.device)
            
            # Forward pass
            logits = self.model(embeddings)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)
            
            # Calculate loss
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = target_ids != self.tokenizer.pad_token_id
            correct = (predictions == target_ids) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct_tokens / total_tokens if total_tokens > 0 else 0
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                embeddings = batch['embedding'].to(self.device)
                target_ids = batch['token_ids'].to(self.device)
                
                # Forward pass
                logits = self.model(embeddings)
                
                # Reshape for loss calculation
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = target_ids.view(-1)
                
                # Calculate loss
                loss = self.criterion(logits_flat, targets_flat)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = target_ids != self.tokenizer.pad_token_id
                correct = (predictions == target_ids) & mask
                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        save_dir: str = 'models/attacker',
        early_stopping_patience: int = 3
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
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_path = os.path.join(save_dir, 'best_inverter.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'history': self.history
                }, model_path)
                print(f"✓ Saved best model to {model_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*50)


def main():
    """Main training function."""
    
    # Configuration
    EMBEDDING_PATH_TRAIN = "data/embeddings/train_embeddings.pkl"
    EMBEDDING_PATH_TEST = "data/embeddings/test_embeddings.pkl"
    MODEL_TYPE = "mlp"  # Options: 'mlp', 'lstm', 'attention'
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 128
    
    print("Loading embeddings...")
    with open(EMBEDDING_PATH_TRAIN, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(EMBEDDING_PATH_TEST, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Train samples: {len(train_data['embeddings'])}")
    print(f"Test samples: {len(test_data['embeddings'])}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    vocab_size = tokenizer.vocab_size
    embedding_dim = train_data['embeddings'].shape[1]
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = EmbeddingInversionDataset(
        train_data['embeddings'],
        train_data['texts'],
        tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = EmbeddingInversionDataset(
        test_data['embeddings'],
        test_data['texts'],
        tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print(f"Creating {MODEL_TYPE} model...")
    if MODEL_TYPE == 'mlp':
        model = EmbeddingInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=MAX_LENGTH
        )
    elif MODEL_TYPE == 'lstm':
        model = LSTMInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=MAX_LENGTH
        )
    elif MODEL_TYPE == 'attention':
        model = AttentionInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=MAX_LENGTH
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = InverterTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        save_dir=f'models/attacker/{MODEL_TYPE}'
    )


if __name__ == "__main__":
    main()