"""
Categorical Attention-based Inverter Model for Labor Court Data.
Optimized for categorical fields: ASSUNTOS, CLASSE PROCESSUAL, RAMO DE ATIVIDADE, MAGISTRADO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class CategoricalAttentionInverter(nn.Module):
    """
    Attention-based inverter for categorical labor court data.
    Predicts 4 categorical fields from BERTimbau embeddings using hierarchical attention.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        # Field-specific vocabulary sizes (determined from data)
        assuntos_vocab_size: int = 500,
        classe_vocab_size: int = 50,
        ramo_vocab_size: int = 100,
        magistrado_vocab_size: int = 200
    ):
        """
        Initialize categorical attention inverter.
        
        Args:
            embedding_dim: Dimension of input embeddings (768 for BERT-base)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer decoder layers
            dropout: Dropout rate
            assuntos_vocab_size: Vocabulary size for ASSUNTOS field
            classe_vocab_size: Vocabulary size for CLASSE PROCESSUAL field
            ramo_vocab_size: Vocabulary size for RAMO DE ATIVIDADE field
            magistrado_vocab_size: Vocabulary size for MAGISTRADO field
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Field names in order
        self.field_names = ['ASSUNTOS', 'CLASSE_PROCESSUAL', 'RAMO_ATIVIDADE', 'MAGISTRADO']
        self.num_fields = len(self.field_names)
        
        # Vocabulary sizes per field
        self.vocab_sizes = {
            'ASSUNTOS': assuntos_vocab_size,
            'CLASSE_PROCESSUAL': classe_vocab_size,
            'RAMO_ATIVIDADE': ramo_vocab_size,
            'MAGISTRADO': magistrado_vocab_size
        }
        
        # 1. Embedding projection layer
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Learnable field queries (one per categorical field)
        # These queries will attend to different aspects of the embedding
        self.field_queries = nn.Parameter(
            torch.randn(self.num_fields, hidden_dim)
        )
        nn.init.xavier_uniform_(self.field_queries)
        
        # 3. Multi-head cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Transformer decoder for refinement
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 5. Field-specific classification heads
        self.field_classifiers = nn.ModuleDict({
            'ASSUNTOS': self._make_classifier(hidden_dim, assuntos_vocab_size, dropout),
            'CLASSE_PROCESSUAL': self._make_classifier(hidden_dim, classe_vocab_size, dropout),
            'RAMO_ATIVIDADE': self._make_classifier(hidden_dim, ramo_vocab_size, dropout),
            'MAGISTRADO': self._make_classifier(hidden_dim, magistrado_vocab_size, dropout)
        })
        
        # Layer normalization for field representations
        self.field_norm = nn.LayerNorm(hidden_dim)
        
    def _make_classifier(self, hidden_dim: int, vocab_size: int, dropout: float) -> nn.Module:
        """
        Create a classification head for a specific categorical field.
        
        Args:
            hidden_dim: Input hidden dimension
            vocab_size: Output vocabulary size for this field
            dropout: Dropout rate
            
        Returns:
            Sequential classifier module
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, vocab_size)
        )
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the categorical attention inverter.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Dictionary with field names as keys and logits as values
                        Each value has shape [batch_size, vocab_size]
            attention_weights: Optional attention weights [batch_size, num_fields, 1]
        """
        batch_size = embeddings.size(0)
        
        # Step 1: Project embedding to hidden space
        memory = self.embedding_projection(embeddings).unsqueeze(1)  # [B, 1, H]
        
        # Step 2: Expand field queries for batch
        queries = self.field_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_fields, H]
        
        # Step 3: Cross-attention - field queries attend to embedding
        attended, attention_weights = self.cross_attention(
            queries, memory, memory
        )  # attended: [B, num_fields, H], attention_weights: [B, num_fields, 1]
        
        # Step 4: Refine with transformer decoder
        refined = self.transformer_decoder(attended, memory)  # [B, num_fields, H]
        
        # Step 5: Normalize field representations
        refined = self.field_norm(refined)
        
        # Step 6: Field-specific predictions
        predictions = {}
        for idx, field_name in enumerate(self.field_names):
            field_repr = refined[:, idx, :]  # [B, H]
            predictions[field_name] = self.field_classifiers[field_name](field_repr)  # [B, vocab_size]
        
        if return_attention:
            return predictions, attention_weights
        return predictions, None
    
    def predict_fields(
        self, 
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Predict categorical field values from embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            temperature: Sampling temperature for softmax
            top_k: Number of top predictions to return per field
            
        Returns:
            Dictionary with field names as keys and predicted class indices as values
            Each value has shape [batch_size, top_k]
        """
        with torch.no_grad():
            predictions, _ = self.forward(embeddings)
            
            predicted_fields = {}
            for field_name, logits in predictions.items():
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get top-k predictions
                if top_k == 1:
                    predicted_fields[field_name] = torch.argmax(logits, dim=-1)
                else:
                    _, top_indices = torch.topk(logits, k=top_k, dim=-1)
                    predicted_fields[field_name] = top_indices
            
        return predicted_fields
    
    def get_field_probabilities(
        self, 
        embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get probability distributions for each field.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Dictionary with field names as keys and probability distributions as values
            Each value has shape [batch_size, vocab_size]
        """
        with torch.no_grad():
            predictions, _ = self.forward(embeddings)
            
            probabilities = {}
            for field_name, logits in predictions.items():
                probabilities[field_name] = F.softmax(logits, dim=-1)
        
        return probabilities


class CategoricalInversionLoss(nn.Module):
    """
    Multi-task loss function for categorical field prediction.
    Supports field-specific weighting and label smoothing.
    """
    
    def __init__(
        self,
        field_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1
    ):
        """
        Initialize categorical inversion loss.
        
        Args:
            field_weights: Dictionary of field importance weights
            label_smoothing: Label smoothing factor for regularization
        """
        super().__init__()
        
        # Default field weights (can be adjusted based on importance)
        self.field_weights = field_weights or {
            'ASSUNTOS': 1.5,           # Most important - case subjects
            'CLASSE_PROCESSUAL': 1.2,  # Important - process type
            'RAMO_ATIVIDADE': 1.0,     # Moderate - business sector
            'MAGISTRADO': 1.3          # Important - judge identity
        }
        
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        Calculate multi-task loss across all categorical fields.
        
        Args:
            predictions: Dictionary {field_name: logits [batch_size, vocab_size]}
            targets: Dictionary {field_name: labels [batch_size]}
            
        Returns:
            total_loss: Weighted sum of field losses
            field_losses: Dictionary of individual field losses
            field_accuracies: Dictionary of field-wise accuracies
        """
        total_loss = 0
        field_losses = {}
        field_accuracies = {}
        
        for field_name in predictions.keys():
            # Cross-entropy loss with label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = criterion(predictions[field_name], targets[field_name])
            
            # Apply field-specific weight
            weight = self.field_weights.get(field_name, 1.0)
            weighted_loss = loss * weight
            
            # Calculate accuracy
            preds = torch.argmax(predictions[field_name], dim=-1)
            acc = (preds == targets[field_name]).float().mean()
            
            # Store metrics
            field_losses[field_name] = loss.item()
            field_accuracies[field_name] = acc.item()
            total_loss += weighted_loss
        
        return total_loss, field_losses, field_accuracies


if __name__ == "__main__":
    # Test the categorical attention inverter
    print("Testing CategoricalAttentionInverter...")
    
    batch_size = 8
    embedding_dim = 768
    
    # Create model
    model = CategoricalAttentionInverter(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        assuntos_vocab_size=500,
        classe_vocab_size=50,
        ramo_vocab_size=100,
        magistrado_vocab_size=200
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create dummy embeddings
    embeddings = torch.randn(batch_size, embedding_dim)
    
    # Forward pass
    predictions, attention_weights = model(embeddings, return_attention=True)
    
    print("\nPredictions:")
    for field_name, logits in predictions.items():
        print(f"  {field_name}: {logits.shape}")
    
    print(f"\nAttention weights: {attention_weights.shape}")
    
    # Test prediction
    predicted_fields = model.predict_fields(embeddings, top_k=3)
    print("\nTop-3 predictions:")
    for field_name, indices in predicted_fields.items():
        print(f"  {field_name}: {indices.shape}")
    
    # Test loss
    print("\nTesting loss function...")
    targets = {
        'ASSUNTOS': torch.randint(0, 500, (batch_size,)),
        'CLASSE_PROCESSUAL': torch.randint(0, 50, (batch_size,)),
        'RAMO_ATIVIDADE': torch.randint(0, 100, (batch_size,)),
        'MAGISTRADO': torch.randint(0, 200, (batch_size,))
    }
    
    loss_fn = CategoricalInversionLoss()
    total_loss, field_losses, field_accs = loss_fn(predictions, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Field losses:", {k: f"{v:.4f}" for k, v in field_losses.items()})
    print("Field accuracies:", {k: f"{v:.4f}" for k, v in field_accs.items()})
    
    print("\n✓ All tests passed!")
