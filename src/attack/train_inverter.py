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

from .inverter_model import EmbeddingInverter, LSTMInverter, AttentionInverter, CategoricalLSTMInverter


class EmbeddingInversionDataset(Dataset):
    """Dataset for training embedding inversion models."""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        tokenizer,
        max_length: int = 128,
        use_categories: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            embeddings: Embedding vectors
            texts: Original texts
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            use_categories: If True, split texts into category segments
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_categories = use_categories
        
        if use_categories:
            # Expand dataset: 1 sample → 3 samples (one per category)
            self._create_categorical_dataset(embeddings, texts)
        else:
            # Standard dataset
            self.embeddings = torch.FloatTensor(embeddings)
            self.texts = texts
            self.category_ids = None
            
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
    
    def _create_categorical_dataset(self, embeddings: np.ndarray, texts: List[str]):
        """Create expanded dataset with category segments."""
        print("Creating categorical dataset (expanding samples)...")
        
        expanded_embeddings = []
        expanded_texts = []
        expanded_categories = []
        
        for i, (emb, text) in enumerate(tqdm(zip(embeddings, texts), total=len(embeddings))):
            # Split text by ' | ' delimiter
            segments = text.split(' | ')
            
            # We expect 4 segments, but use first 3 for categories
            # (ASSUNTOS, CLASSE PROCESSUAL, RAMO DE ATIVIDADE)
            num_segments = min(3, len(segments))
            
            for cat_id in range(num_segments):
                if cat_id < len(segments):
                    segment_text = segments[cat_id].strip()
                    if segment_text:  # Only add non-empty segments
                        expanded_embeddings.append(emb)
                        expanded_texts.append(segment_text)
                        expanded_categories.append(cat_id)
        
        self.embeddings = torch.FloatTensor(np.array(expanded_embeddings))
        self.texts = expanded_texts
        self.category_ids = torch.LongTensor(expanded_categories)
        
        print(f"Expanded from {len(embeddings)} to {len(self.embeddings)} samples")
        print(f"Category distribution: {torch.bincount(self.category_ids).tolist()}")
        
        # Pre-tokenize all texts
        print("Tokenizing segment texts...")
        self.tokenized = []
        for text in tqdm(self.texts):
            encoded = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.tokenized.append(encoded.squeeze(0))
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        item = {
            'embedding': self.embeddings[idx],
            'token_ids': self.tokenized[idx],
            'text': self.texts[idx]
        }
        
        if self.category_ids is not None:
            item['category_id'] = self.category_ids[idx]
        
        return item


class InverterTrainer:
    """Trainer for embedding inverter models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        is_categorical: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Inverter model
            tokenizer: Tokenizer
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            is_categorical: Whether the model requires category_ids
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_categorical = is_categorical
        
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
        
        for batch_idx, batch in enumerate(pbar):
            embeddings = batch['embedding'].to(self.device)
            target_ids = batch['token_ids'].to(self.device)
            
            # Forward pass
            if self.is_categorical:
                # Use actual category IDs from dataset
                if 'category_id' in batch:
                    category_ids = batch['category_id'].to(self.device)
                else:
                    # Fallback to random if not available
                    batch_size = embeddings.size(0)
                    category_ids = torch.randint(0, 3, (batch_size,)).to(self.device)
                logits = self.model(embeddings, category_ids)
            else:
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
            
            # Calculate accuracy (detach to save memory)
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = target_ids != self.tokenizer.pad_token_id
                correct = (predictions == target_ids) & mask
                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            # Clear cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct_tokens / total_tokens if total_tokens > 0 else 0
            })
            
            # Delete tensors to free memory
            del embeddings, target_ids, logits, loss, predictions, mask, correct
        
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
                if self.is_categorical:
                    # Use actual category IDs from dataset
                    if 'category_id' in batch:
                        category_ids = batch['category_id'].to(self.device)
                    else:
                        # Fallback to random if not available
                        batch_size = embeddings.size(0)
                        category_ids = torch.randint(0, 3, (batch_size,)).to(self.device)
                    logits = self.model(embeddings, category_ids)
                else:
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


def train_single_model(
    model_type: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    tokenizer,
    embedding_dim: int,
    vocab_size: int,
    max_length: int,
    num_epochs: int,
    learning_rate: float,
    num_categories: int = None
) -> Dict:
    """
    Train a single model and return results.
    
    Args:
        model_type: Type of model ('mlp', 'lstm', 'attention', 'categorical_lstm')
        train_loader: Training data loader
        test_loader: Test data loader
        tokenizer: Tokenizer
        embedding_dim: Embedding dimension
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        num_epochs: Number of epochs
        learning_rate: Learning rate
        num_categories: Number of categories (required for categorical_lstm)
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*80}")
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'mlp':
        model = EmbeddingInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=max_length
        )
    elif model_type == 'lstm':
        model = LSTMInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=max_length
        )
    elif model_type == 'attention':
        model = AttentionInverter(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=max_length
        )
    elif model_type == 'categorical_lstm':
        if num_categories is None:
            raise ValueError("num_categories must be provided for categorical_lstm model")
        model = CategoricalLSTMInverter(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            max_seq_length=max_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = InverterTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        is_categorical=(model_type == 'categorical_lstm')
    )
    
    # Train
    print(f"\nStarting training for {model_type}...")
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        save_dir=f'models/attacker/{model_type}'
    )
    
    # Get final results
    results = {
        'model_type': model_type,
        'num_params': num_params,
        'best_val_loss': min(trainer.history['val_loss']) if trainer.history['val_loss'] else float('inf'),
        'best_val_accuracy': max(trainer.history['val_accuracy']) if trainer.history['val_accuracy'] else 0,
        'history': trainer.history
    }
    
    print(f"\n✓ {model_type.upper()} training complete!")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    print(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    # Clear memory
    import gc
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    """Main training function - trains all models."""
    
    # Configuration
    EMBEDDING_PATH_TRAIN = "data/embeddings/train_embeddings.pkl"
    EMBEDDING_PATH_TEST = "data/embeddings/test_embeddings.pkl"
    MODEL_TYPES = ["mlp", "lstm", "attention", "categorical_lstm"]  # Train all models
    BATCH_SIZE = 16   # Reduced for memory efficiency
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 64
    NUM_CATEGORIES = 3  # ["ASSUNTOS", "CLASSE PROCESSUAL", "RAMO DE ATIVIDADE"]
    
    print("="*80)
    print("EMBEDDING INVERSION ATTACK - TRAINING ALL MODELS")
    print("="*80)
    
    print("\nLoading embeddings...")
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
    
    # Note: We'll create datasets per model type to handle categorical separately
    # This is a placeholder - datasets will be created in the training loop
    datasets_created = False
    
    # Clear memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Train all models
    all_results = {}
    
    for model_type in MODEL_TYPES:
        try:
            print(f"\n{'='*80}")
            print(f"Preparing datasets for {model_type.upper()} model")
            print(f"{'='*80}")
            
            # Create appropriate dataset for model type
            use_categorical = (model_type == 'categorical_lstm')
            
            print(f"Creating {'categorical' if use_categorical else 'standard'} dataset...")
            train_dataset = EmbeddingInversionDataset(
                train_data['embeddings'],
                train_data['texts'],
                tokenizer,
                max_length=MAX_LENGTH,
                use_categories=use_categorical
            )
            
            test_dataset = EmbeddingInversionDataset(
                test_data['embeddings'],
                test_data['texts'],
                tokenizer,
                max_length=MAX_LENGTH,
                use_categories=use_categorical
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
            
            # Train model
            results = train_single_model(
                model_type=model_type,
                train_loader=train_loader,
                test_loader=test_loader,
                tokenizer=tokenizer,
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                max_length=MAX_LENGTH,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                num_categories=NUM_CATEGORIES if model_type == 'categorical_lstm' else None
            )
            all_results[model_type] = results
            
            # Clean up
            del train_dataset, test_dataset, train_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n✗ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    print("\n" + "="*80)
    print("TRAINING SUMMARY - ALL MODELS")
    print("="*80)
    
    summary_lines = []
    summary_lines.append("\n| Model | Parameters | Best Val Loss | Best Val Accuracy |")
    summary_lines.append("|-------|------------|---------------|-------------------|")
    
    for model_type in MODEL_TYPES:
        if model_type in all_results:
            results = all_results[model_type]
            summary_lines.append(
                f"| {model_type.upper():9} | "
                f"{results['num_params']:>10,} | "
                f"{results['best_val_loss']:>13.4f} | "
                f"{results['best_val_accuracy']:>17.4f} |"
            )
    
    summary = "\n".join(summary_lines)
    print(summary)
    
    # Save summary to file
    os.makedirs('results', exist_ok=True)
    with open('results/training_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("EMBEDDING INVERSION ATTACK - TRAINING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(summary)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Trained Models:\n")
        for model_type in MODEL_TYPES:
            if model_type in all_results:
                f.write(f"  ✓ {model_type.upper()}: models/attacker/{model_type}/best_inverter.pt\n")
        f.write("="*80 + "\n")
    
    print("\n✓ Training summary saved to results/training_summary.txt")
    
    # Save detailed results
    with open('results/training_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("✓ Detailed results saved to results/training_results.pkl")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()