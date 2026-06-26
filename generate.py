"""
Phase 5: Text Generation
=========================
This module provides interactive text generation with a trained GPT model.

SAMPLING STRATEGIES:
1. Greedy: Always pick the most likely token (deterministic)
2. Temperature: Scale logits to control randomness
3. Top-k: Sample from only the k most likely tokens
4. Top-p (Nucleus): Sample from smallest set with cumulative prob >= p

BEFORE RUNNING:
    You must train the model first:
    python train.py

RUN THIS FILE:
    python generate.py -i                    # Interactive mode
    python generate.py -p "Newton" -n 200   # For physics dataset
    python generate.py -p "def " -n 200     # For python dataset
    python generate.py --demo               # Compare sampling strategies

EXAMPLE PROMPTS BY DATASET:
    Physics:     "Newton", "Energy", "The force", "Momentum", "Gravity"
    Python:      "def ", "class ", "for i in", "if __name__"
    Shakespeare: "ROMEO:", "To be", "What light"

HOW TO EVALUATE:
    - Physics: Does the output make scientific sense?
    - Python: Is the syntax valid? Does indentation work?
    - Shakespeare: Does it sound like old English dialogue?
"""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

from data_pipeline import GPTDataset, CharacterTokenizer
from gpt_model import GPT


# =============================================================================
# SAMPLING STRATEGIES EXPLAINED
# =============================================================================
"""
THE GENERATION PROCESS:

At each step:
1. Feed context to model → get logits for next token
2. Convert logits to probabilities (with sampling strategy)
3. Sample from probability distribution
4. Append new token to context
5. Repeat

LOGITS → PROBABILITIES:
    logits = model output (unnormalized log-probabilities)
    probs = softmax(logits) = exp(logits) / sum(exp(logits))

SAMPLING STRATEGIES:

1. GREEDY (temperature=0 or argmax)
   - Always pick highest probability token
   - Deterministic: same input → same output
   - Problem: Can get stuck in repetitive loops
   
   Example: probs = [0.4, 0.3, 0.2, 0.1]
            → Always pick token 0

2. TEMPERATURE
   - Scale logits before softmax: probs = softmax(logits / T)
   - T < 1: Sharper distribution (more confident, less random)
   - T = 1: Original distribution
   - T > 1: Flatter distribution (less confident, more random)
   
   Example with logits = [2.0, 1.0, 0.5, 0.0]:
   
   T=0.5 (cold): probs ≈ [0.73, 0.18, 0.07, 0.02]  ← Very confident
   T=1.0:        probs ≈ [0.47, 0.26, 0.17, 0.10]  ← Balanced
   T=2.0 (hot):  probs ≈ [0.35, 0.28, 0.22, 0.15]  ← More uniform

3. TOP-K SAMPLING
   - Only consider the k most likely tokens
   - Set probability of others to 0, renormalize
   - Prevents sampling very unlikely tokens
   
   Example with k=2:
   probs = [0.4, 0.3, 0.2, 0.1]
        → [0.4, 0.3, 0.0, 0.0]  (mask out bottom 2)
        → [0.57, 0.43, 0.0, 0.0]  (renormalize)

4. TOP-P (NUCLEUS) SAMPLING
   - Sort tokens by probability
   - Keep smallest set where cumulative prob >= p
   - More adaptive than top-k
   
   Example with p=0.8:
   probs = [0.4, 0.3, 0.2, 0.1]
   cumsum = [0.4, 0.7, 0.9, 1.0]
   Keep tokens until cumsum >= 0.8 → keep first 3
   probs → [0.44, 0.33, 0.22, 0.0]  (renormalize)

COMBINING STRATEGIES:
   Common practice: temperature + top_p
   Example: temperature=0.8, top_p=0.9
   
   This gives controlled randomness with protection
   against very unlikely tokens.
"""


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    """
    Text generator using a trained GPT model.
    
    Usage:
        gen = Generator('checkpoints/best_model.pt')
        
        # Simple generation (physics dataset)
        text = gen.generate("Newton", max_tokens=100)
        
        # With parameters
        text = gen.generate(
            "Energy is conserved",
            max_tokens=200,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto)
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint['config']
        self.block_size = config['block_size']
        self.vocab_size = config['vocab_size']
        
        # Build tokenizer from dataset
        # We need the same vocabulary as training
        print("Building tokenizer from dataset...")
        dataset = GPTDataset('data/input.txt', block_size=self.block_size, device=device)
        self.tokenizer = dataset.tokenizer
        
        # Create and load model
        print("Loading model weights...")
        self.model = GPT(**config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded! Parameters: {self.model.count_parameters():,}")
        print(f"Validation loss at checkpoint: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tensor to text."""
        return self.tokenizer.decode(tokens.squeeze(0).tolist())
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        stop_on_newline: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Starting text (can be empty)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy, 1 = normal, >1 = more random)
            top_k: Only sample from top k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            stop_on_newline: Stop generation at first newline
            
        Returns:
            Generated text (including prompt)
        """
        # Handle empty prompt
        if prompt:
            idx = self.encode(prompt)
        else:
            # Start with newline (common starting point)
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self.model(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            if temperature == 0:
                # Greedy: pick argmax
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Scale by temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    logits = self._top_k_filtering(logits, top_k)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    logits = self._top_p_filtering(logits, top_p)
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
            
            # Check for stop condition
            if stop_on_newline:
                if self.tokenizer.decode([idx_next.item()]) == '\n':
                    break
        
        return self.decode(idx)
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Filter logits to keep only top-k values.
        
        All other values are set to -inf (zero probability after softmax).
        """
        # Get the k-th largest value
        values, _ = torch.topk(logits, k)
        min_value = values[:, -1].unsqueeze(-1)
        
        # Set everything below min_value to -inf
        return torch.where(
            logits < min_value,
            torch.full_like(logits, float('-inf')),
            logits
        )
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.
        
        Keep smallest set of tokens with cumulative probability >= p.
        """
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff: first position where cumulative prob exceeds p
        # Shift by 1 to keep at least one token
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Set filtered logits to -inf
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def interactive(self):
        """
        Interactive generation mode.
        
        Type a prompt and get generated text.
        Commands:
            /quit - Exit
            /temp <value> - Set temperature
            /topk <value> - Set top-k (0 to disable)
            /topp <value> - Set top-p (0 to disable)
            /tokens <value> - Set max tokens
        """
        print("\n" + "="*60)
        print(" Interactive Text Generation")
        print("="*60)
        print("\nCommands:")
        print("  /quit          - Exit")
        print("  /temp <value>  - Set temperature (default: 0.8)")
        print("  /topk <value>  - Set top-k (default: 40, 0 to disable)")
        print("  /topp <value>  - Set top-p (default: 0.9, 0 to disable)")
        print("  /tokens <value>- Set max tokens (default: 200)")
        print("\nEnter a prompt to generate text:\n")
        
        # Default parameters
        temperature = 0.8
        top_k = 40
        top_p = 0.9
        max_tokens = 200
        
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd == '/quit':
                    print("Goodbye!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                elif cmd == '/topk' and len(parts) > 1:
                    top_k = int(parts[1])
                    top_k = top_k if top_k > 0 else None
                    print(f"Top-k set to {top_k}")
                elif cmd == '/topp' and len(parts) > 1:
                    top_p = float(parts[1])
                    top_p = top_p if top_p > 0 else None
                    print(f"Top-p set to {top_p}")
                elif cmd == '/tokens' and len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                else:
                    print("Unknown command. Use /quit, /temp, /topk, /topp, /tokens")
                continue
            
            # Generate text
            print("\n" + "-"*40)
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            print(result)
            print("-"*40 + "\n")


# =============================================================================
# COMPARISON DEMO
# =============================================================================

def demo_sampling_strategies(gen: Generator):
    """
    Demonstrate different sampling strategies.
    """
    prompt = "Newton"  # Use physics prompt (change for other datasets)
    
    print("\n" + "="*60)
    print(" Sampling Strategies Comparison")
    print("="*60)
    print(f"\nPrompt: '{prompt}'")
    
    strategies = [
        ("Greedy (temp=0)", {"temperature": 0}),
        ("Low temp (0.5)", {"temperature": 0.5}),
        ("Normal (temp=1.0)", {"temperature": 1.0}),
        ("High temp (1.5)", {"temperature": 1.5}),
        ("Top-k (k=10)", {"temperature": 1.0, "top_k": 10}),
        ("Top-k (k=40)", {"temperature": 1.0, "top_k": 40}),
        ("Top-p (p=0.5)", {"temperature": 1.0, "top_p": 0.5}),
        ("Top-p (p=0.9)", {"temperature": 1.0, "top_p": 0.9}),
        ("Combined (temp=0.8, k=40, p=0.9)", {"temperature": 0.8, "top_k": 40, "top_p": 0.9}),
    ]
    
    for name, params in strategies:
        print(f"\n--- {name} ---")
        result = gen.generate(prompt, max_tokens=100, **params)
        # Show just the generated part
        print(result)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate text with trained GPT')
    parser.add_argument(
        '--checkpoint', '-c',
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt', '-p',
        default=None,
        help='Text prompt to start generation'
    )
    parser.add_argument(
        '--tokens', '-n',
        type=int,
        default=200,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=40,
        help='Top-k sampling (0 to disable)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p/nucleus sampling (0 to disable)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo different sampling strategies'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("\nYou need to train the model first:")
        print("  python train.py")
        return
    
    # Load generator
    gen = Generator(args.checkpoint)
    
    # Run mode
    if args.demo:
        demo_sampling_strategies(gen)
    elif args.interactive:
        gen.interactive()
    else:
        # Single generation
        prompt = args.prompt or ""
        top_k = args.top_k if args.top_k > 0 else None
        top_p = args.top_p if args.top_p > 0 else None
        
        print("\n" + "="*60)
        print(" Text Generation")
        print("="*60)
        print(f"\nPrompt: '{prompt}'")
        print(f"Parameters: temp={args.temperature}, top_k={top_k}, top_p={top_p}")
        print("\n" + "-"*40)
        
        result = gen.generate(
            prompt=prompt,
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p
        )
        print(result)
        print("-"*40)


if __name__ == "__main__":
    main()
