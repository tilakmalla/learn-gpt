"""
Phase 2: Data Pipeline for GPT
==============================
This module handles converting raw text into training data for our GPT model.

KEY CONCEPTS:
1. Tokenization: Text → Numbers (we use character-level for simplicity)
2. Vocabulary: The set of all unique tokens (characters)
3. Context Window: How much text the model "sees" at once
4. Batching: Processing multiple sequences in parallel for efficiency

Run this file directly to see the pipeline in action:
    python data_pipeline.py
"""

import torch
import os
from pathlib import Path

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Set seeds for reproducible results across runs
SEED = 1337
torch.manual_seed(SEED)

# =============================================================================
# PART 1: LOADING THE DATA
# =============================================================================

def load_text(filepath: str) -> str:
    """
    Load raw text from file.
    
    Shakespeare dataset is ~1MB, ~1 million characters.
    Perfect size for learning - small enough to train quickly,
    large enough to learn interesting patterns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            f"Download it with:\n"
            f"  curl -o {filepath} https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# =============================================================================
# PART 2: CHARACTER-LEVEL TOKENIZATION
# =============================================================================
"""
TOKENIZATION: Converting text to numbers

Why do we need this?
- Neural networks operate on numbers, not text
- We need a reversible mapping: text ↔ numbers

Two main approaches:
1. Character-level: Each character = one token
   - Vocabulary size ≈ 65 (a-z, A-Z, 0-9, punctuation)
   - Pros: Simple, handles any word, small vocabulary
   - Cons: Longer sequences, harder to capture word meaning

2. Subword-level (BPE, WordPiece): Common character sequences = one token
   - Vocabulary size ≈ 50,000
   - Pros: Shorter sequences, captures word parts
   - Cons: Complex to implement
   
We use CHARACTER-LEVEL because:
- Easier to understand
- No external tokenizer needed
- Good for learning the fundamentals
"""

class CharacterTokenizer:
    """
    Maps characters ↔ integers.
    
    Example:
        'hello' → [46, 43, 50, 50, 53]
        [46, 43, 50, 50, 53] → 'hello'
    """
    
    def __init__(self, text: str):
        """
        Build vocabulary from text.
        
        Steps:
        1. Find all unique characters
        2. Sort them (for reproducibility)
        3. Create bidirectional mappings
        """
        # Get all unique characters (the vocabulary)
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        # stoi: "string to integer" - for encoding
        # itos: "integer to string" - for decoding
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")
    
    def encode(self, text: str) -> list[int]:
        """
        Convert text to list of integers.
        
        'Hi!' → [20, 47, 1]
        
        MATH: This is essentially a lookup table.
        f(c) = stoi[c] for each character c
        """
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        """
        Convert list of integers back to text.
        
        [20, 47, 1] → 'Hi!'
        
        This is the inverse of encode:
        f⁻¹(i) = itos[i] for each integer i
        """
        return ''.join([self.itos[i] for i in tokens])


# =============================================================================
# PART 3: TRAIN/VALIDATION SPLIT
# =============================================================================
"""
WHY SPLIT THE DATA?

Training data: Used to update model weights (what the model learns from)
Validation data: Used to check if model generalizes (never trained on)

If val_loss >> train_loss: OVERFITTING (memorizing, not learning)
If both high: UNDERFITTING (model too small or needs more training)

Standard split: 90% train, 10% validation
"""

def train_val_split(data: torch.Tensor, train_ratio: float = 0.9):
    """
    Split data into training and validation sets.
    
    Args:
        data: Full dataset as tensor of token IDs
        train_ratio: Fraction for training (default 90%)
    
    Returns:
        train_data, val_data: Two tensors
    """
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training tokens:   {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    return train_data, val_data


# =============================================================================
# PART 4: BATCHING WITH CONTEXT WINDOWS
# =============================================================================
"""
CONTEXT WINDOW (block_size)

GPT processes text in fixed-size chunks called "context windows".

Example with block_size=8:
    Text: "Hello World"
    Input:  "Hello Wo"  (8 chars)
    Target: "ello Wor"  (shifted by 1)

WHY THE SHIFT?
For each position, we predict the NEXT character:
    Position 0: Given "H", predict "e"
    Position 1: Given "He", predict "l"
    Position 2: Given "Hel", predict "l"
    ...

This is AUTOREGRESSIVE language modeling:
    P(next_char | all_previous_chars)


BATCHING

Instead of one sequence at a time, we process multiple in parallel:
    - Batch of 4 sequences, each length 8
    - Shape: (4, 8) = (batch_size, block_size)
    
Why batch?
    - GPU parallelism: Matrix ops on (4,8) barely slower than (1,8)
    - Faster training: 4x more data per forward pass
    - Better gradients: Average over multiple examples
"""

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random batch of training examples.
    
    Args:
        data: Full dataset (1D tensor of token IDs)
        batch_size: Number of sequences per batch (e.g., 64)
        block_size: Length of each sequence (e.g., 256)
        device: 'cpu', 'mps' (Mac), or 'cuda' (NVIDIA)
    
    Returns:
        x: Input sequences, shape (batch_size, block_size)
        y: Target sequences, shape (batch_size, block_size)
        
    EXAMPLE (batch_size=2, block_size=4):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        Random starting positions: [2, 5]
        
        x = [[2, 3, 4, 5],    # Starting at position 2
             [5, 6, 7, 8]]    # Starting at position 5
             
        y = [[3, 4, 5, 6],    # Each is x shifted by 1
             [6, 7, 8, 9]]
    """
    # Random starting positions for each sequence in the batch
    # We need room for block_size + 1 tokens (input + 1 target)
    # max valid start = len(data) - block_size - 1 (so last target at len-1)
    max_start = len(data) - block_size
    ix = torch.randint(0, max_start, (batch_size,))
    
    # Stack sequences into batch tensors
    # x[i] = data[ix[i] : ix[i] + block_size]
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # y is x shifted by 1 position (the prediction target)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to target device (CPU/GPU)
    x, y = x.to(device), y.to(device)
    
    return x, y


# =============================================================================
# PART 5: PUTTING IT ALL TOGETHER
# =============================================================================

class GPTDataset:
    """
    Complete data pipeline for GPT training.
    
    Usage:
        dataset = GPTDataset('data/input.txt', block_size=256)
        x, y = dataset.get_batch('train', batch_size=64)
    """
    
    def __init__(
        self,
        filepath: str,
        block_size: int = 256,
        train_ratio: float = 0.9,
        device: str = 'cpu'
    ):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path to text file
            block_size: Context window size
            train_ratio: Fraction for training
            device: Compute device
        """
        self.block_size = block_size
        self.device = device
        
        # Load and tokenize
        print("Loading data...")
        text = load_text(filepath)
        print(f"Total characters: {len(text):,}")
        
        # Build tokenizer
        print("\nBuilding tokenizer...")
        self.tokenizer = CharacterTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size
        
        # Convert to tensor
        print("\nEncoding text...")
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # Split
        print("\nSplitting data...")
        self.train_data, self.val_data = train_val_split(data, train_ratio)
    
    def get_batch(self, split: str, batch_size: int):
        """
        Get a batch from train or validation set.
        
        Args:
            split: 'train' or 'val'
            batch_size: Number of sequences
            
        Returns:
            x, y: Input and target tensors
        """
        data = self.train_data if split == 'train' else self.val_data
        return get_batch(data, batch_size, self.block_size, self.device)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tensor to text."""
        return self.tokenizer.decode(tokens.tolist())


# =============================================================================
# DEMO: Run this file to see the pipeline in action
# =============================================================================

if __name__ == "__main__":
    # Detect device (MPS for Mac M1, CUDA for NVIDIA, else CPU)
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
    print(" GPT Data Pipeline Demo")
    print("="*60)
    
    # Initialize dataset
    dataset = GPTDataset(
        filepath='data/input.txt',
        block_size=8,  # Small for demo visibility
        device=device
    )
    
    # Show tokenizer in action
    print("\n" + "-"*60)
    print(" TOKENIZER DEMO")
    print("-"*60)
    
    sample_text = "Hello, World!"
    tokens = dataset.tokenizer.encode(sample_text)
    decoded = dataset.tokenizer.decode(tokens)
    
    print(f"Original:  '{sample_text}'")
    print(f"Encoded:   {tokens}")
    print(f"Decoded:   '{decoded}'")
    print(f"Roundtrip: {sample_text == decoded}")
    
    # Show batching
    print("\n" + "-"*60)
    print(" BATCHING DEMO")
    print("-"*60)
    
    x, y = dataset.get_batch('train', batch_size=4)
    
    print(f"Input shape:  {x.shape}  (batch_size=4, block_size=8)")
    print(f"Target shape: {y.shape}")
    print(f"Device: {x.device}")
    
    print("\nFirst sequence in batch:")
    print(f"  Input (x):  {dataset.decode(x[0])!r}")
    print(f"  Target (y): {dataset.decode(y[0])!r}")
    print("  Notice: y is x shifted by 1 character!")
    
    # Show the prediction task
    print("\n" + "-"*60)
    print(" THE PREDICTION TASK")
    print("-"*60)
    
    seq = x[0]
    target = y[0]
    print("At each position, predict the next character:\n")
    for i in range(len(seq)):
        context = dataset.decode(seq[:i+1])
        next_char = dataset.decode(target[i:i+1])
        print(f"  Given: {context!r:12} → Predict: {next_char!r}")
    
    print("\n" + "="*60)
    print(" Phase 2 Complete!")
    print("="*60)
    print("\nNext: Phase 3 - Building the GPT model architecture")
