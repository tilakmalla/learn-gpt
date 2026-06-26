"""
Phase 3: GPT Model Architecture
================================
This module implements a GPT (Generative Pre-trained Transformer) from scratch.

ARCHITECTURE OVERVIEW:
    Input Tokens
         ↓
    Token Embedding + Positional Embedding
         ↓
    ┌─────────────────────────────────┐
    │     Transformer Block (×N)      │
    │  ┌───────────────────────────┐  │
    │  │   Multi-Head Attention    │  │
    │  │   + Residual + LayerNorm  │  │
    │  ├───────────────────────────┤  │
    │  │   Feed-Forward Network    │  │
    │  │   + Residual + LayerNorm  │  │
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
         ↓
    Final LayerNorm
         ↓
    Linear (vocab_size outputs)
         ↓
    Logits (probability distribution over next token)

RUN THIS FILE:
    python gpt_model.py

EXPECTED OUTPUT:
    - Model configuration and parameter count (~10.8M parameters)
    - Forward pass test showing input/output shapes
    - Initial loss ~4.17 (random guessing)

NEXT STEP:
    python train.py    # Train the model (Phase 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# HYPERPARAMETERS (for reference - will be passed to model)
# =============================================================================
"""
These control the model's capacity and behavior:

- vocab_size: Number of unique tokens (65 for character-level Shakespeare)
- n_embd: Embedding dimension (384) - size of token vectors
- n_head: Number of attention heads (6) - parallel attention patterns
- n_layer: Number of transformer blocks (6) - depth of the network
- block_size: Maximum context length (256) - how far back model can "see"
- dropout: Regularization (0.2) - randomly zero out 20% of values during training
"""


# =============================================================================
# PART 1: EMBEDDINGS
# =============================================================================
"""
EMBEDDINGS: Converting discrete tokens to continuous vectors

Why embeddings?
- Tokens are discrete (integers 0, 1, 2, ...)
- Neural networks need continuous, differentiable inputs
- We learn a vector representation for each token

TOKEN EMBEDDING:
    Each token ID maps to a learnable vector of size n_embd
    
    Mathematically: E_token ∈ ℝ^(vocab_size × n_embd)
    
    token_id = 5  →  E_token[5] = [0.2, -0.5, 0.8, ...]  (n_embd values)

POSITIONAL EMBEDDING:
    Transformers have no built-in notion of sequence order!
    (Unlike RNNs which process sequentially)
    
    We add position information: "this token is at position 3"
    
    Mathematically: E_pos ∈ ℝ^(block_size × n_embd)
    
    position = 3  →  E_pos[3] = [0.1, 0.3, -0.2, ...]
    
FINAL EMBEDDING:
    x = E_token[token_id] + E_pos[position]
    
    Shape: (batch_size, sequence_length, n_embd)
"""


# =============================================================================
# PART 2: SELF-ATTENTION (The Heart of Transformers)
# =============================================================================
"""
SELF-ATTENTION: How tokens "communicate" with each other

THE INTUITION:
    When processing the word "it" in "The cat sat on the mat. It was tired."
    The model needs to know "it" refers to "cat".
    
    Self-attention lets each token look at ALL previous tokens
    and decide which ones are relevant.

THE MECHANISM: Query, Key, Value (Q, K, V)

    Think of it like a search engine:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What information do I provide?"
    
    Each token generates Q, K, V by linear projection:
        Q = X @ W_q    (X is input, W_q is learnable weights)
        K = X @ W_k
        V = X @ W_v

THE MATH:
    
    1. Compute attention scores (how relevant is each position?):
       
       scores = Q @ K^T / √(d_k)
       
       - Q @ K^T: dot product measures similarity
       - √(d_k): scaling prevents scores from getting too large
                 (which would make softmax too peaky)
    
    2. Apply causal mask (can't look at future tokens!):
       
       scores = scores.masked_fill(mask == 0, -inf)
       
       Example with sequence length 4:
       mask = [[1, 0, 0, 0],    Token 0 can only see itself
               [1, 1, 0, 0],    Token 1 can see 0 and itself
               [1, 1, 1, 0],    Token 2 can see 0, 1, and itself
               [1, 1, 1, 1]]    Token 3 can see everything before it
    
    3. Convert to probabilities:
       
       attention_weights = softmax(scores)
       
       Each row sums to 1 (probability distribution)
    
    4. Weighted sum of values:
       
       output = attention_weights @ V
       
       Each position is now a weighted combination of all
       previous positions' values, weighted by relevance.

DIMENSIONS (for one head):
    X: (batch, seq_len, n_embd)
    Q, K, V: (batch, seq_len, head_size)  where head_size = n_embd // n_head
    scores: (batch, seq_len, seq_len)
    output: (batch, seq_len, head_size)
"""

class Head(nn.Module):
    """
    Single head of self-attention.
    
    Each head learns to attend to different types of relationships:
    - One head might learn syntactic relationships
    - Another might learn semantic relationships
    - etc.
    """
    
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        """
        Args:
            n_embd: Input embedding dimension
            head_size: Output dimension for this head (usually n_embd // n_head)
            block_size: Maximum sequence length (for causal mask)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Linear projections for Q, K, V
        # No bias is common in attention (slight efficiency gain, minimal impact)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask: lower triangular matrix
        # register_buffer: not a parameter, but should be saved with model
        # tril = "triangular lower"
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store for scaling
        self.head_size = head_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of single attention head.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch, seq_len, head_size)
        """
        B, T, C = x.shape  # batch, sequence length (time), channels (embedding dim)
        
        # Compute Q, K, V projections
        # Shape: (B, T, head_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Apply causal mask: tokens can only attend to previous positions
        # We use [:T, :T] because actual sequence might be shorter than block_size
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Convert to probabilities
        # softmax along last dimension (which positions to attend to)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout (randomly zero out some attention weights)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        output = attention_weights @ v
        
        return output


# =============================================================================
# PART 3: MULTI-HEAD ATTENTION
# =============================================================================
"""
MULTI-HEAD ATTENTION: Run multiple attention heads in parallel

WHY MULTIPLE HEADS?
    - Single head might focus on one type of relationship
    - Multiple heads can capture different relationship types simultaneously
    - Then we concatenate and project back
    
    "Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions."
    - Attention Is All You Need (Vaswani et al., 2017)

THE MATH:
    head_i = Attention(Q_i, K_i, V_i)
    
    MultiHead = Concat(head_1, ..., head_h) @ W_o
    
    Where W_o ∈ ℝ^(h·head_size × n_embd) projects back to original dimension

DIMENSIONS:
    Each head: (batch, seq_len, head_size)
    Concatenated: (batch, seq_len, n_head * head_size) = (batch, seq_len, n_embd)
    After projection: (batch, seq_len, n_embd)
"""

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        """
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Each head has dimension n_embd // n_head
        head_size = n_embd // n_head
        
        # Create multiple heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        
        # Output projection: combine all heads back to n_embd
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Run all heads in parallel and concatenate along last dimension
        # Each head outputs (B, T, head_size)
        # Concatenated: (B, T, n_head * head_size) = (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project back and apply dropout
        out = self.dropout(self.proj(out))
        
        return out


# =============================================================================
# PART 4: FEED-FORWARD NETWORK
# =============================================================================
"""
FEED-FORWARD NETWORK (FFN): Per-position processing

After attention aggregates information across positions,
FFN processes each position independently.

ARCHITECTURE:
    FFN(x) = Linear_2(GELU(Linear_1(x)))
    
    - Linear_1: n_embd → 4 * n_embd  (expand)
    - GELU: non-linearity (smooth ReLU variant)
    - Linear_2: 4 * n_embd → n_embd  (contract)

WHY 4x EXPANSION?
    The inner dimension (4 * n_embd) gives the network more capacity
    to learn complex transformations. This is a hyperparameter choice
    from the original Transformer paper.

WHY GELU (Gaussian Error Linear Unit)?
    GELU(x) = x * Φ(x)  where Φ is the standard Gaussian CDF
    
    - Smoother than ReLU: helps gradient flow
    - Used in GPT-2, BERT, GPT-3
    - Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
"""

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Applied independently to each position in the sequence.
    """
    
    def __init__(self, n_embd: int, dropout: float):
        """
        Args:
            n_embd: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Expand: n_embd → 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),
            # Non-linearity
            nn.GELU(),
            # Contract: 4 * n_embd → n_embd
            nn.Linear(4 * n_embd, n_embd),
            # Regularization
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        return self.net(x)


# =============================================================================
# PART 5: TRANSFORMER BLOCK
# =============================================================================
"""
TRANSFORMER BLOCK: One complete layer of the transformer

ARCHITECTURE:
    x → LayerNorm → MultiHeadAttention → + (residual) → 
      → LayerNorm → FeedForward → + (residual) → output

KEY CONCEPTS:

1. RESIDUAL CONNECTIONS (the + operations):
    output = x + sublayer(x)
    
    Why? Helps gradients flow through deep networks.
    The gradient can "skip" the sublayer via the residual path.
    
    Without residuals: gradient must flow through every layer
    With residuals: gradient has a "highway" straight to earlier layers

2. LAYER NORMALIZATION:
    Normalizes activations to have mean=0, std=1, then applies
    learnable scale (γ) and shift (β).
    
    LayerNorm(x) = γ * (x - mean(x)) / (std(x) + ε) + β
    
    Why? Stabilizes training by keeping activations in a reasonable range.
    
    Note: GPT uses "Pre-Norm" (normalize before sublayer)
    Original Transformer used "Post-Norm" (normalize after)
    Pre-Norm is more stable for deep networks.

DIMENSIONS:
    Input: (batch, seq_len, n_embd)
    Output: (batch, seq_len, n_embd)
    
    The shape stays the same! Transformer blocks are stackable.
"""

class TransformerBlock(nn.Module):
    """
    Single transformer block: attention + feed-forward with residuals.
    """
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        """
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Layer normalizations (Pre-Norm architecture)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Self-attention
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(n_embd, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Attention with residual connection (Pre-Norm)
        # x + attention(LayerNorm(x))
        x = x + self.attn(self.ln1(x))
        
        # Feed-forward with residual connection (Pre-Norm)
        # x + ffn(LayerNorm(x))
        x = x + self.ffn(self.ln2(x))
        
        return x


# =============================================================================
# PART 6: FULL GPT MODEL
# =============================================================================
"""
GPT MODEL: Putting it all together

ARCHITECTURE:
    1. Token embedding + Positional embedding
    2. Stack of N transformer blocks
    3. Final layer normalization
    4. Linear projection to vocabulary size (logits)

THE OUTPUT (Logits):
    For each position, we get vocab_size numbers.
    These are "logits" - unnormalized log-probabilities.
    
    softmax(logits) → probability distribution over next token
    
    Example:
        logits[position=5] = [1.2, -0.5, 3.1, ...]  (vocab_size values)
        probs[position=5] = softmax(logits[position=5])
        predicted_token = argmax(probs[position=5])

LOSS FUNCTION:
    Cross-entropy loss measures how well our probability distribution
    matches the true next token.
    
    loss = -log(prob[correct_token])
    
    If we're confident about the right answer: loss ≈ 0
    If we're confident about the wrong answer: loss → ∞
"""

class GPT(nn.Module):
    """
    Full GPT language model.
    
    Usage:
        model = GPT(vocab_size=65, n_embd=384, n_head=6, n_layer=6, 
                    block_size=256, dropout=0.2)
        
        # Forward pass
        logits, loss = model(x, targets)
        
        # Generation
        new_tokens = model.generate(x, max_new_tokens=100)
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float
    ):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            n_embd: Embedding dimension
            n_head: Number of attention heads per block
            n_layer: Number of transformer blocks
            block_size: Maximum sequence length (context window)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.block_size = block_size
        
        # Token embedding: vocab_size → n_embd
        # Each token ID maps to a learnable vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional embedding: block_size → n_embd
        # Each position maps to a learnable vector
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output projection: n_embd → vocab_size
        # Produces logits for next token prediction
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights (important for training stability)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights with small random values.
        
        Standard initialization for transformers:
        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - Biases: Zero
        - LayerNorm: Default (ones for scale, zeros for shift)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GPT.
        
        Args:
            idx: Input token indices, shape (batch, seq_len)
            targets: Target token indices, shape (batch, seq_len), optional
            
        Returns:
            logits: Predictions, shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided), else None
        """
        B, T = idx.shape  # batch size, sequence length
        
        # Get token embeddings
        # (B, T) → (B, T, n_embd)
        tok_emb = self.token_embedding(idx)
        
        # Get positional embeddings
        # positions: [0, 1, 2, ..., T-1]
        # (T,) → (T, n_embd) → broadcasts to (B, T, n_embd)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)
        
        # Combine: token embedding + positional embedding
        # (B, T, n_embd)
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        # (B, T, n_embd) → (B, T, n_embd)
        x = self.blocks(x)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Project to vocabulary size (logits)
        # (B, T, n_embd) → (B, T, vocab_size)
        logits = self.lm_head(x)
        
        # Compute loss if targets are provided
        if targets is not None:
            # Reshape for cross_entropy:
            # logits: (B*T, vocab_size)
            # targets: (B*T,)
            B, T, V = logits.shape
            logits_flat = logits.view(B*T, V)
            targets_flat = targets.view(B*T)
            
            # Cross-entropy loss
            # Measures how well our predictions match the targets
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None
        
        return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting context, shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness (1.0 = normal, <1 = more deterministic)
            top_k: Only sample from top k most likely tokens (None = all)
            
        Returns:
            Extended sequence, shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (model's maximum context length)
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last position (what comes next?)
            # (B, T, vocab_size) → (B, vocab_size)
            logits = logits[:, -1, :]
            
            # Apply temperature (higher = more random)
            logits = logits / temperature
            
            # Optional: top-k filtering
            if top_k is not None:
                # Keep only top k logits, set others to -inf
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            # (B, vocab_size) → (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            # (B, T) + (B, 1) → (B, T+1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# DEMO: Run this file to see the model architecture
# =============================================================================

if __name__ == "__main__":
    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Mac M1 GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    print("\n" + "="*60)
    print(" GPT Model Architecture Demo")
    print("="*60)
    
    # Model hyperparameters (same as in GPT_LEARNING_PROMPT.md)
    config = {
        'vocab_size': 65,       # Character-level vocabulary
        'n_embd': 384,          # Embedding dimension
        'n_head': 6,            # Attention heads
        'n_layer': 6,           # Transformer blocks
        'block_size': 256,      # Context window
        'dropout': 0.2,         # Regularization
    }
    
    print("\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = GPT(**config).to(device)
    
    print(f"\nTotal Parameters: {model.count_parameters():,}")
    
    # Parameter breakdown
    print("\nParameter Breakdown:")
    
    # Token embedding
    tok_params = config['vocab_size'] * config['n_embd']
    print(f"  Token Embedding:    {tok_params:>10,}  ({config['vocab_size']} × {config['n_embd']})")
    
    # Position embedding
    pos_params = config['block_size'] * config['n_embd']
    print(f"  Position Embedding: {pos_params:>10,}  ({config['block_size']} × {config['n_embd']})")
    
    # Per transformer block
    head_size = config['n_embd'] // config['n_head']
    qkv_params = 3 * config['n_embd'] * head_size * config['n_head']  # Q, K, V
    proj_params = config['n_embd'] * config['n_embd']  # output projection
    ffn_params = 2 * config['n_embd'] * 4 * config['n_embd']  # two linear layers
    ln_params = 2 * 2 * config['n_embd']  # two layer norms (scale + shift each)
    block_params = qkv_params + proj_params + ffn_params + ln_params
    print(f"  Per Transformer Block: {block_params:>8,}")
    print(f"    - Attention (Q,K,V): {qkv_params:>8,}")
    print(f"    - Attention (proj):  {proj_params:>8,}")
    print(f"    - FFN:               {ffn_params:>8,}")
    print(f"    - LayerNorms:        {ln_params:>8,}")
    print(f"  × {config['n_layer']} blocks:        {block_params * config['n_layer']:>10,}")
    
    # Final layer norm + output projection
    final_ln = 2 * config['n_embd']
    lm_head = config['n_embd'] * config['vocab_size']
    print(f"  Final LayerNorm:    {final_ln:>10,}")
    print(f"  Output Projection:  {lm_head:>10,}")
    
    # Test forward pass
    print("\n" + "-"*60)
    print(" Forward Pass Test")
    print("-"*60)
    
    # Create dummy input
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    print(f"\nInput shape:  {tuple(x.shape)}  (batch={batch_size}, seq_len={seq_len})")
    print(f"Target shape: {tuple(targets.shape)}")
    
    # Forward pass
    logits, loss = model(x, targets)
    
    print(f"Logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected initial loss: ~{math.log(config['vocab_size']):.4f} (random = -log(1/vocab_size))")
    
    # Test generation
    print("\n" + "-"*60)
    print(" Generation Test")
    print("-"*60)
    
    # Start with a single token
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate 20 tokens
    model.eval()  # Disable dropout for generation
    with torch.no_grad():
        generated = model.generate(start, max_new_tokens=20, temperature=1.0)
    
    print(f"\nGenerated token indices: {generated[0].tolist()}")
    print("(These are random until we train the model!)")
    
    print("\n" + "="*60)
    print(" Phase 3 Complete!")
    print("="*60)
    print("\nNext: Phase 4 - Training Infrastructure")
    print("  - Loss function & optimizer")
    print("  - Learning rate scheduler")
    print("  - Training loop")
