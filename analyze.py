"""
Phase 6: Analysis & Experiments
================================
This module provides tools for understanding and experimenting with the GPT model.

TOPICS:
1. Attention Visualization: See what the model "looks at"
2. Hyperparameter Experiments: How do changes affect performance?
3. Loss Analysis: Training curves and overfitting detection
4. Token Probability Analysis: What does the model predict?

BEFORE RUNNING:
    You must train the model first:
    python train.py

RUN THIS FILE:
    python analyze.py --experiments                    # Show experiment configs
    python analyze.py --attention "Newton" --layer 0  # Visualize attention (physics)
    python analyze.py --attention "def " --layer 0    # Visualize attention (python)
    python analyze.py --predictions "Energy is"       # Analyze predictions

OUTPUT:
    - Attention heatmaps saved to analysis/ folder
    - Shows which tokens the model attends to
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

from data_pipeline import GPTDataset
from gpt_model import GPT


# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================
"""
ATTENTION VISUALIZATION: See what tokens attend to what

The attention weights show which positions each token "looks at"
when computing its representation.

Shape: (n_head, seq_len, seq_len)
- Row i = attention pattern for position i
- Column j = how much position i attends to position j
- Each row sums to 1 (probability distribution)

PATTERNS TO LOOK FOR:
- Diagonal: Tokens attending to themselves
- Previous token: Strong attention to immediately preceding token
- Punctuation: Special attention to sentence boundaries
- Names/entities: Attention to character names in dialogue
"""

class AttentionVisualizer:
    """
    Visualize attention patterns in a trained GPT model.
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """Load model for visualization."""
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Load tokenizer
        dataset = GPTDataset('data/input.txt', block_size=config['block_size'], device=device)
        self.tokenizer = dataset.tokenizer
        
        # Load model
        self.model = GPT(**config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.config = config
        self.n_head = config['n_head']
        self.n_layer = config['n_layer']
    
    def get_attention_weights(self, text: str) -> list:
        """
        Extract attention weights for all layers and heads.
        
        Returns:
            List of (n_layer) tensors, each of shape (n_head, seq_len, seq_len)
        """
        # Encode text
        tokens = self.tokenizer.encode(text)
        idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # We need to modify forward pass to capture attention weights
        # For now, we'll compute them manually
        
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.model.token_embedding(idx)
        pos_emb = self.model.position_embedding(torch.arange(T, device=self.device))
        x = self.model.dropout(tok_emb + pos_emb)
        
        attention_weights = []
        
        # Go through each block
        for block in self.model.blocks:
            # Layer norm
            x_norm = block.ln1(x)
            
            # Collect attention weights from each head
            head_weights = []
            head_outputs = []
            
            for head in block.attn.heads:
                q = head.query(x_norm)
                k = head.key(x_norm)
                v = head.value(x_norm)
                
                # Compute attention scores
                scores = q @ k.transpose(-2, -1) * (head.head_size ** -0.5)
                scores = scores.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
                weights = F.softmax(scores, dim=-1)
                
                head_weights.append(weights.squeeze(0))  # (seq_len, seq_len)
                head_outputs.append(weights @ v)
            
            # Stack heads: (n_head, seq_len, seq_len)
            attention_weights.append(torch.stack(head_weights))
            
            # Continue forward pass
            attn_out = torch.cat(head_outputs, dim=-1)
            attn_out = block.attn.dropout(block.attn.proj(attn_out))
            x = x + attn_out
            x = x + block.ffn(block.ln2(x))
        
        return attention_weights, tokens
    
    def plot_attention(
        self,
        text: str,
        layer: int = 0,
        head: int = 0,
        save_path: str = None
    ):
        """
        Plot attention weights for a specific layer and head.
        
        Args:
            text: Input text to analyze
            layer: Which transformer layer (0 to n_layer-1)
            head: Which attention head (0 to n_head-1)
            save_path: Path to save figure (None = display)
        """
        attention_weights, tokens = self.get_attention_weights(text)
        
        # Get specific layer/head
        weights = attention_weights[layer][head].cpu().detach().numpy()
        
        # Create labels (characters)
        labels = [self.tokenizer.itos[t] for t in tokens]
        # Replace special characters for display
        labels = [repr(c)[1:-1] if c in '\n\t ' else c for c in labels]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(weights, cmap='Blues')
        
        # Labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        
        ax.set_xlabel('Key Position (attending to)')
        ax.set_ylabel('Query Position (attending from)')
        ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
        
        plt.colorbar(im)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_heads(self, text: str, layer: int = 0, save_path: str = None):
        """
        Plot attention patterns for all heads in a layer.
        """
        attention_weights, tokens = self.get_attention_weights(text)
        
        labels = [self.tokenizer.itos[t] for t in tokens]
        labels = [repr(c)[1:-1] if c in '\n\t ' else c for c in labels]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for head in range(self.n_head):
            weights = attention_weights[layer][head].cpu().detach().numpy()
            
            ax = axes[head]
            im = ax.imshow(weights, cmap='Blues')
            ax.set_title(f'Head {head}')
            
            if len(tokens) <= 20:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=8)
                ax.set_yticklabels(labels, fontsize=8)
        
        plt.suptitle(f'Attention Patterns - Layer {layer}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# TRAINING ANALYSIS
# =============================================================================
"""
TRAINING ANALYSIS: Understanding the learning process

Key metrics:
- Training loss: How well model fits training data
- Validation loss: How well model generalizes
- Gap between them: Indicates overfitting

HEALTHY TRAINING:
- Both losses decrease
- Val loss follows train loss closely
- Eventually val loss plateaus (or increases = overfitting)
"""

def plot_training_curves(log_file: str = 'training_log.txt', save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Note: Requires training log with format:
    iteration,train_loss,val_loss,lr
    """
    # Parse log file (you'd need to modify train.py to save this)
    # For now, show example with dummy data
    
    print("To plot training curves, modify train.py to log losses to a file.")
    print("Example format: iteration,train_loss,val_loss,lr")


# =============================================================================
# HYPERPARAMETER EXPERIMENTS
# =============================================================================
"""
HYPERPARAMETER EXPERIMENTS

Key hyperparameters and their effects:

1. n_embd (embedding dimension):
   - Larger = more capacity, slower, more memory
   - Typical: 128, 256, 384, 512, 768
   
2. n_head (attention heads):
   - More heads = more diverse attention patterns
   - Must divide n_embd evenly
   - Typical: 4, 6, 8, 12
   
3. n_layer (transformer blocks):
   - More layers = deeper network, more capacity
   - Diminishing returns after certain point
   - Typical: 4, 6, 8, 12
   
4. block_size (context window):
   - Larger = can see more context
   - Memory scales quadratically with block_size
   - Typical: 128, 256, 512, 1024
   
5. dropout:
   - Higher = more regularization
   - Too high = underfitting
   - Typical: 0.1, 0.2, 0.3
   
6. learning_rate:
   - Too high = unstable training
   - Too low = slow convergence
   - Typical: 1e-4, 3e-4, 1e-3
"""

# Experiment configurations to try
EXPERIMENTS = {
    'baseline': {
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'block_size': 256,
        'dropout': 0.2,
    },
    'small_model': {
        'n_embd': 128,
        'n_head': 4,
        'n_layer': 4,
        'block_size': 128,
        'dropout': 0.1,
    },
    'wide_model': {
        'n_embd': 512,
        'n_head': 8,
        'n_layer': 4,
        'block_size': 256,
        'dropout': 0.2,
    },
    'deep_model': {
        'n_embd': 256,
        'n_head': 4,
        'n_layer': 8,
        'block_size': 256,
        'dropout': 0.2,
    },
    'large_context': {
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'block_size': 512,
        'dropout': 0.2,
    },
}

def count_parameters(config: dict) -> int:
    """Estimate parameter count for a configuration."""
    vocab_size = 65  # Shakespeare
    n_embd = config['n_embd']
    n_head = config['n_head']
    n_layer = config['n_layer']
    block_size = config['block_size']
    
    # Token + position embeddings
    params = vocab_size * n_embd + block_size * n_embd
    
    # Per transformer block
    # Attention: Q, K, V projections + output projection
    attn_params = 4 * n_embd * n_embd
    # FFN: two linear layers (expand 4x then contract)
    ffn_params = 2 * n_embd * (4 * n_embd)
    # Layer norms: 2 per block
    ln_params = 4 * n_embd
    
    block_params = attn_params + ffn_params + ln_params
    params += n_layer * block_params
    
    # Final layer norm + output projection
    params += 2 * n_embd + n_embd * vocab_size
    
    return params

def show_experiments():
    """Display experiment configurations."""
    print("\n" + "="*70)
    print(" Hyperparameter Experiment Configurations")
    print("="*70)
    
    for name, config in EXPERIMENTS.items():
        params = count_parameters(config)
        print(f"\n{name.upper()}")
        print(f"  n_embd={config['n_embd']}, n_head={config['n_head']}, "
              f"n_layer={config['n_layer']}, block_size={config['block_size']}")
        print(f"  Parameters: {params:,}")
    
    print("\n" + "-"*70)
    print("To run an experiment, modify the hyperparameters in train.py")
    print("and run: python train.py")


# =============================================================================
# TOKEN PROBABILITY ANALYSIS
# =============================================================================

def analyze_predictions(checkpoint_path: str, text: str, device: str = None):
    """
    Analyze what the model predicts at each position.
    
    Shows top-k predictions for each position in the input.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    dataset = GPTDataset('data/input.txt', block_size=config['block_size'], device=device)
    tokenizer = dataset.tokenizer
    
    model = GPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Encode text
    tokens = tokenizer.encode(text)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        logits, _ = model(idx)
    
    # Analyze each position
    print("\n" + "="*70)
    print(" Token Prediction Analysis")
    print("="*70)
    print(f"\nInput: '{text}'")
    print("\nPosition | Context → Predictions (probability)")
    print("-"*70)
    
    for i in range(len(tokens)):
        context = tokenizer.decode(tokens[:i+1])
        
        # Get probabilities for next token
        probs = F.softmax(logits[0, i], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        
        # Format predictions
        preds = []
        for prob, idx in zip(top_probs, top_indices):
            char = tokenizer.itos[idx.item()]
            char_repr = repr(char)[1:-1]  # Clean representation
            preds.append(f"'{char_repr}':{prob.item():.2f}")
        
        # Truncate context for display
        ctx_display = context[-20:] if len(context) > 20 else context
        ctx_display = repr(ctx_display)[1:-1]
        
        print(f"{i:3d}      | {ctx_display:22s} → {', '.join(preds)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze GPT model')
    parser.add_argument(
        '--checkpoint', '-c',
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--attention', '-a',
        type=str,
        help='Text to visualize attention for'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=0,
        help='Layer to visualize (for attention)'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=-1,
        help='Head to visualize (-1 for all heads)'
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Text to analyze predictions for'
    )
    parser.add_argument(
        '--experiments', '-e',
        action='store_true',
        help='Show experiment configurations'
    )
    parser.add_argument(
        '--save-dir',
        default='analysis',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    # Create save directory
    Path(args.save_dir).mkdir(exist_ok=True)
    
    if args.experiments:
        show_experiments()
        return
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Train the model first: python train.py")
        return
    
    if args.attention:
        print("Loading model for attention visualization...")
        viz = AttentionVisualizer(args.checkpoint)
        
        if args.head == -1:
            # All heads
            save_path = Path(args.save_dir) / f'attention_layer{args.layer}_all_heads.png'
            viz.plot_all_heads(args.attention, layer=args.layer, save_path=str(save_path))
        else:
            # Single head
            save_path = Path(args.save_dir) / f'attention_layer{args.layer}_head{args.head}.png'
            viz.plot_attention(args.attention, layer=args.layer, head=args.head, 
                             save_path=str(save_path))
    
    if args.predictions:
        analyze_predictions(args.checkpoint, args.predictions)
    
    if not args.attention and not args.predictions and not args.experiments:
        print("Usage examples:")
        print("  python analyze.py --experiments")
        print("  python analyze.py --attention 'ROMEO:' --layer 0")
        print("  python analyze.py --predictions 'To be or not'")


if __name__ == "__main__":
    main()
