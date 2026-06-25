"""
Phase 4: Training Infrastructure
=================================
This module handles training the GPT model.

KEY CONCEPTS:
1. Loss Function: Cross-entropy measures prediction quality
2. Optimizer: AdamW updates weights to minimize loss
3. Learning Rate Schedule: Warmup + cosine decay
4. Training Loop: Forward pass → Loss → Backward pass → Update
5. Validation: Check if model generalizes (not just memorizing)
6. Checkpointing: Save progress to resume later

Run this file to train the model:
    python train.py

Expected training time: ~20-30 minutes on M1 Mac
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import time
import os
from pathlib import Path

# Import our modules
from data_pipeline import GPTDataset
from gpt_model import GPT

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Model architecture (must match gpt_model.py)
VOCAB_SIZE = 65          # Will be set from dataset
N_EMBD = 384             # Embedding dimension
N_HEAD = 6               # Number of attention heads
N_LAYER = 6              # Number of transformer blocks
BLOCK_SIZE = 256         # Context window (max sequence length)
DROPOUT = 0.2            # Dropout probability

# Training
BATCH_SIZE = 64          # Sequences per batch
MAX_ITERS = 5000         # Total training iterations
EVAL_INTERVAL = 500      # Evaluate every N iterations
EVAL_ITERS = 200         # Number of batches for evaluation
LEARNING_RATE = 3e-4     # Peak learning rate
WARMUP_ITERS = 100       # Linear warmup steps
MIN_LR = 1e-5            # Minimum learning rate (for cosine decay)

# Checkpointing
CHECKPOINT_DIR = 'checkpoints'
SAVE_INTERVAL = 1000     # Save checkpoint every N iterations

# Reproducibility
SEED = 1337


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
"""
LEARNING RATE SCHEDULE: How fast to update weights

WHY NOT CONSTANT LR?
- Too high at start: Unstable, loss explodes
- Too high at end: Overshoots optimal weights
- Too low: Training is very slow

THE SCHEDULE:
1. WARMUP (linear increase):
   - Start from 0, linearly increase to LEARNING_RATE
   - Allows model to "settle" before aggressive updates
   
2. COSINE DECAY:
   - Smoothly decrease from LEARNING_RATE to MIN_LR
   - Follows cosine curve: LR = MIN_LR + 0.5*(MAX_LR - MIN_LR)*(1 + cos(π*progress))
   
Visual:
    LR
    │    ╭─────╮
    │   ╱       ╲
    │  ╱         ╲
    │ ╱           ╲____
    └─────────────────── iteration
      warmup  decay
"""

def get_lr(iteration: int) -> float:
    """
    Calculate learning rate for given iteration.
    
    Args:
        iteration: Current training iteration
        
    Returns:
        Learning rate for this iteration
    """
    # Warmup: linear increase from 0 to LEARNING_RATE
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration + 1) / WARMUP_ITERS
    
    # After warmup: cosine decay from LEARNING_RATE to MIN_LR
    # Progress goes from 0 (just after warmup) to 1 (at max_iters)
    progress = (iteration - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    
    # Clamp progress to [0, 1]
    progress = min(1.0, max(0.0, progress))
    
    # Cosine decay formula
    # cos(0) = 1, cos(π) = -1
    # So (1 + cos(π*progress))/2 goes from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# =============================================================================
# LOSS ESTIMATION
# =============================================================================
"""
WHY ESTIMATE LOSS OVER MULTIPLE BATCHES?

Single batch loss is noisy - varies a lot between batches.
Averaging over EVAL_ITERS batches gives a stable estimate.

We evaluate on BOTH train and val sets:
- train_loss: How well model fits training data
- val_loss: How well model generalizes to unseen data

If val_loss >> train_loss: OVERFITTING
If both decreasing together: HEALTHY TRAINING
"""

@torch.no_grad()  # Disable gradient computation (saves memory)
def estimate_loss(
    model: nn.Module,
    dataset: GPTDataset,
    eval_iters: int = EVAL_ITERS
) -> dict:
    """
    Estimate loss on train and validation sets.
    
    Args:
        model: The GPT model
        dataset: GPTDataset with train/val data
        eval_iters: Number of batches to average over
        
    Returns:
        Dictionary with 'train' and 'val' losses
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    
    losses = {}
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, BATCH_SIZE)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    
    model.train()  # Set back to training mode
    return losses


# =============================================================================
# CHECKPOINTING
# =============================================================================
"""
CHECKPOINTING: Save model state to resume later

What we save:
- model_state_dict: All model weights
- optimizer_state_dict: Optimizer momentum/variance (for AdamW)
- iteration: Where we left off
- config: Model hyperparameters
- best_val_loss: For model selection

Why save optimizer state?
AdamW maintains running averages for each parameter.
Without these, training would "restart" even with trained weights.
"""

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    val_loss: float,
    config: dict,
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'val_loss': val_loss,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> tuple:
    """
    Load training checkpoint.
    
    Returns:
        iteration, val_loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration'], checkpoint['val_loss']


# =============================================================================
# TRAINING LOOP
# =============================================================================
"""
THE TRAINING LOOP: Where learning happens

Each iteration:
1. Get a batch of data (x, y)
2. Forward pass: logits, loss = model(x, y)
3. Backward pass: loss.backward() computes gradients
4. Update weights: optimizer.step()
5. Zero gradients: optimizer.zero_grad() (prepare for next iteration)

GRADIENT DESCENT INTUITION:
- Loss is a function of weights: L(w)
- Gradient ∇L tells us which direction increases loss
- We move in the OPPOSITE direction: w = w - lr * ∇L
- Over time, loss decreases and model improves
"""

def train():
    """Main training function."""
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Mac M1 GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = 'cpu'
        print("Using CPU (this will be slow!)")
    
    print("\n" + "="*60)
    print(" GPT Training")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("\nLoading dataset...")
    dataset = GPTDataset(
        filepath='data/input.txt',
        block_size=BLOCK_SIZE,
        device=device
    )
    vocab_size = dataset.vocab_size
    
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print("\nCreating model...")
    config = {
        'vocab_size': vocab_size,
        'n_embd': N_EMBD,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'dropout': DROPOUT,
    }
    
    model = GPT(**config).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    
    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    """
    ADAMW OPTIMIZER
    
    Adam = Adaptive Moment Estimation
    - Maintains per-parameter learning rates
    - Uses momentum (first moment) and variance (second moment)
    
    AdamW = Adam with decoupled Weight Decay
    - Weight decay: Slightly shrink weights each step (regularization)
    - Prevents weights from growing too large
    - "Decoupled" means weight decay is applied separately from gradient update
    
    Why AdamW for transformers?
    - Works well across many architectures
    - Less sensitive to learning rate choice than SGD
    - Standard choice for GPT, BERT, etc.
    """
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    print("\nTraining configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max iterations: {MAX_ITERS}")
    print(f"  Learning rate: {LEARNING_RATE} (with warmup + cosine decay)")
    print(f"  Device: {device}")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Track best model
    best_val_loss = float('inf')
    
    print("\n" + "-"*60)
    print(" Starting Training")
    print("-"*60)
    
    start_time = time.time()
    
    for iteration in range(MAX_ITERS):
        # ---------------------------------------------------------------------
        # Learning rate schedule
        # ---------------------------------------------------------------------
        lr = get_lr(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ---------------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------------
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)
            elapsed = time.time() - start_time
            
            print(f"\nIteration {iteration:5d} | "
                  f"Train Loss: {losses['train']:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(
                    model, optimizer, iteration, losses['val'], config,
                    os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                )
        
        # ---------------------------------------------------------------------
        # Regular checkpoint
        # ---------------------------------------------------------------------
        if iteration > 0 and iteration % SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, iteration, losses.get('val', float('inf')), config,
                os.path.join(CHECKPOINT_DIR, f'checkpoint_{iteration}.pt')
            )
        
        # ---------------------------------------------------------------------
        # Training step
        # ---------------------------------------------------------------------
        
        # Get batch
        x, y = dataset.get_batch('train', BATCH_SIZE)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass (compute gradients)
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients
        loss.backward()                         # Compute new gradients
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
    
    # -------------------------------------------------------------------------
    # Done!
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {CHECKPOINT_DIR}/best_model.pt")
    
    # -------------------------------------------------------------------------
    # Generate sample text
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print(" Sample Generation")
    print("-"*60)
    
    model.eval()
    
    # Start with newline character
    start_tokens = dataset.encode("\n").unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            max_new_tokens=500,
            temperature=0.8,
            top_k=40
        )
    
    print("\nGenerated text:")
    print("-"*40)
    print(dataset.decode(generated[0]))
    print("-"*40)
    
    return model, dataset


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    model, dataset = train()
