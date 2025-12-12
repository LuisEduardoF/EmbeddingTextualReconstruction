"""
Adversarial inverter model for embedding inversion attack.
This model attempts to reconstruct text from BERTimbau embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class EmbeddingInverter(nn.Module):
    """
    Neural network that attempts to invert embeddings back to text.
    Uses a decoder architecture to map from embedding space to vocabulary space.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        vocab_size: int = 30000,
        hidden_dims: list = [512, 512],
        max_seq_length: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the inverter model.
        
        Args:
            embedding_dim: Dimension of input embeddings (768 for BERT-base)
            vocab_size: Size of vocabulary
            hidden_dims: List of hidden layer dimensions
            max_seq_length: Maximum sequence length to reconstruct
            dropout: Dropout rate
        """
        super(EmbeddingInverter, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Build decoder layers
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output projection to vocabulary logits for each position
        self.output_projection = nn.Linear(prev_dim, vocab_size * max_seq_length)
        
        # Alternative: sequence-to-sequence decoder with attention
        self.use_seq2seq = False  # Can be toggled for different architectures
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embedding -> reconstructed token logits.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Token logits [batch_size, max_seq_length, vocab_size]
        """
        batch_size = embeddings.size(0)
        
        # Decode embeddings
        hidden = self.decoder(embeddings)
        
        # Project to vocabulary space
        logits = self.output_projection(hidden)
        
        # Reshape to [batch_size, max_seq_length, vocab_size]
        logits = logits.view(batch_size, self.max_seq_length, self.vocab_size)
        
        return logits
    
    def predict_tokens(
        self, 
        embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Predict token IDs from embeddings.
        
        Args:
            embeddings: Input embeddings
            temperature: Sampling temperature
            
        Returns:
            Predicted token IDs [batch_size, max_seq_length]
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get most likely tokens
            predicted_ids = torch.argmax(logits, dim=-1)
            
        return predicted_ids


class LSTMInverter(nn.Module):
    """
    LSTM-based inverter that generates text sequentially.
    More sophisticated than the MLP inverter.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        vocab_size: int = 30000,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_seq_length: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM inverter.
        
        Args:
            embedding_dim: Dimension of input embeddings
            vocab_size: Size of vocabulary
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(LSTMInverter, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Project embedding to initial hidden state
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim * num_layers * 2)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=vocab_size,  # One-hot or embedding input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Embedding layer for teacher forcing
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        
    def init_hidden(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from embedding.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Tuple of (h0, c0) for LSTM
        """
        batch_size = embeddings.size(0)
        
        # Project embedding to hidden state
        projected = self.embedding_projection(embeddings)
        projected = projected.view(batch_size, self.num_layers, 2, self.hidden_dim)
        
        h0 = projected[:, :, 0, :].transpose(0, 1).contiguous()
        c0 = projected[:, :, 1, :].transpose(0, 1).contiguous()
        
        return h0, c0
    
    def forward(
        self,
        embeddings: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            target_tokens: Target token IDs for teacher forcing [batch_size, seq_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Token logits [batch_size, max_seq_length, vocab_size]
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Initialize hidden state
        h, c = self.init_hidden(embeddings)
        
        # Start token (assume 0 is padding/start)
        input_token = torch.zeros(batch_size, 1, self.vocab_size).to(device)
        
        outputs = []
        
        for t in range(self.max_seq_length):
            # LSTM step
            lstm_out, (h, c) = self.lstm(input_token, (h, c))
            
            # Project to vocabulary
            logits = self.output_projection(lstm_out.squeeze(1))
            outputs.append(logits)
            
            # Prepare next input
            if target_tokens is not None and np.random.random() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                if t < target_tokens.size(1) - 1:
                    next_token_id = target_tokens[:, t + 1]
                    input_token = F.one_hot(next_token_id, self.vocab_size).float().unsqueeze(1)
                else:
                    input_token = torch.zeros(batch_size, 1, self.vocab_size).to(device)
            else:
                # Use model's own prediction
                predicted_id = torch.argmax(logits, dim=-1)
                input_token = F.one_hot(predicted_id, self.vocab_size).float().unsqueeze(1)
        
        # Stack outputs
        logits = torch.stack(outputs, dim=1)
        
        return logits


class AttentionInverter(nn.Module):
    """
    Transformer-based inverter with attention mechanism.
    Most sophisticated architecture for inversion.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        vocab_size: int = 30000,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 512,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize attention-based inverter.
        
        Args:
            embedding_dim: Dimension of input embeddings
            vocab_size: Size of vocabulary
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(AttentionInverter, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        
        # Project embedding to hidden dimension
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_dim)
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention-based inverter.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Token logits [batch_size, max_seq_length, vocab_size]
        """
        batch_size = embeddings.size(0)
        
        # Project embedding
        memory = self.embedding_projection(embeddings).unsqueeze(1)  # [B, 1, H]
        
        # Create target sequence with positional encoding
        tgt = self.pos_encoding.expand(batch_size, -1, -1)  # [B, L, H]
        
        # Decode
        decoded = self.transformer_decoder(tgt, memory)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    embedding_dim = 768
    vocab_size = 30000
    max_seq_length = 128
    
    # Create dummy embeddings
    embeddings = torch.randn(batch_size, embedding_dim)
    
    print("Testing EmbeddingInverter...")
    model1 = EmbeddingInverter(embedding_dim, vocab_size, max_seq_length=max_seq_length)
    output1 = model1(embeddings)
    print(f"Output shape: {output1.shape}")
    
    print("\nTesting LSTMInverter...")
    model2 = LSTMInverter(embedding_dim, vocab_size, max_seq_length=max_seq_length)
    output2 = model2(embeddings)
    print(f"Output shape: {output2.shape}")
    
    print("\nTesting AttentionInverter...")
    model3 = AttentionInverter(embedding_dim, vocab_size, max_seq_length=max_seq_length)
    output3 = model3(embeddings)
    print(f"Output shape: {output3.shape}")
    
    print("\nAll models working correctly!")