# GPT Model Learning Journey - Master Prompt

## Context
- **Platform**: MacBook Pro M1 (Apple Silicon), 32GB RAM, 1TB SSD
- **Background**: Advanced mathematics degree, strong linear algebra/matrices foundation
- **Programming**: Python proficiency
- **Goal**: Hands-on understanding of GPT architecture through self-implementation

---

## The Prompt

```
You are guiding a learner with an advanced mathematics background (linear algebra, matrices, 
calculus) and Python proficiency through building a GPT model from scratch on a MacBook Pro M1.

OBJECTIVE: Build a minimal but complete GPT implementation that illuminates these core concepts:
1. Tokenization - How text becomes numbers
2. Embeddings - How tokens become vectors
3. Positional Encoding - How sequence order is preserved
4. Self-Attention - The "Query, Key, Value" mechanism (the heart of transformers)
5. Multi-Head Attention - Parallel attention patterns
6. Feed-Forward Networks - The MLP layers between attention
7. Layer Normalization - Stabilizing training
8. The Decoder Stack - Putting it all together
9. Training Loop - Cross-entropy loss, backpropagation, optimization
10. Text Generation - Sampling strategies (greedy, temperature, top-k)

CONSTRAINTS:
- Use PyTorch with MPS (Metal Performance Shaders) for M1 GPU acceleration
- Start with character-level tokenization (simplest to understand)
- Train on a small, digestible dataset (Shakespeare ~1MB is ideal)
- Model should be trainable in <30 minutes on M1
- Each component must be implemented manually (no HuggingFace transformers initially)
- Code must be heavily commented explaining the math

DELIVERABLES:
1. Single-file implementation (~300-400 lines) with inline mathematical explanations
2. Working model that generates coherent text after training
3. Visualization of attention patterns
4. Experiments: vary hyperparameters and observe effects

SUCCESS CRITERIA:
- Learner can explain each component mathematically
- Learner can modify architecture and predict effects
- Generated text shows learned patterns (not random garbage)
```

---

## Learning Path & Time Estimates

### Phase 1: Environment Setup (1 hour)
- [ ] Install Python 3.11+ via Homebrew (Apple Silicon native)
- [ ] Create virtual environment
- [ ] Install PyTorch with MPS support
- [ ] Verify GPU acceleration works
- [ ] Download Shakespeare dataset

### Phase 2: Data Pipeline (1.5 hours)
- [ ] Character-level tokenizer implementation
- [ ] Vocabulary building (encode/decode functions)
- [ ] Train/validation split
- [ ] Batching with context windows
- [ ] DataLoader setup

### Phase 3: Core Components (4 hours)
- [ ] **Embeddings** (30 min): Token + positional embeddings
- [ ] **Self-Attention** (1.5 hr): Q, K, V matrices, scaled dot-product, masking
- [ ] **Multi-Head Attention** (45 min): Parallel heads, concatenation, projection
- [ ] **Feed-Forward Network** (30 min): Two linear layers with GELU
- [ ] **Transformer Block** (30 min): Attention → Add&Norm → FFN → Add&Norm
- [ ] **Full GPT Model** (30 min): Stack blocks + final linear head

### Phase 4: Training Infrastructure (2 hours)
- [ ] Loss function (cross-entropy)
- [ ] Optimizer (AdamW with weight decay)
- [ ] Learning rate scheduler (cosine with warmup)
- [ ] Training loop with gradient accumulation
- [ ] Validation loop and loss tracking
- [ ] Checkpointing

### Phase 5: Text Generation (1.5 hours)
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-k sampling
- [ ] Top-p (nucleus) sampling
- [ ] Interactive generation loop

### Phase 6: Analysis & Experiments (2 hours)
- [ ] Visualize attention patterns
- [ ] Experiment: Change number of heads
- [ ] Experiment: Change embedding dimension
- [ ] Experiment: Change number of layers
- [ ] Observe overfitting vs underfitting

---

## Total Time: ~12 hours (spread over 2-3 days recommended)

| Phase | Hours | Complexity |
|-------|-------|------------|
| Environment Setup | 1.0 | Low |
| Data Pipeline | 1.5 | Medium |
| Core Components | 4.0 | High |
| Training Infrastructure | 2.0 | Medium |
| Text Generation | 1.5 | Medium |
| Analysis & Experiments | 2.0 | Medium |
| **Total** | **12.0** | - |

---

## Mac M1 Specific Considerations

### MPS (Metal Performance Shaders) Setup
```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Fallback: some operations not supported on MPS, use CPU selectively
```

### Recommended Hyperparameters for M1 32GB
```python
batch_size = 64          # Can go higher with 32GB
block_size = 256         # Context window
n_embd = 384             # Embedding dimension
n_head = 6               # Attention heads
n_layer = 6              # Transformer blocks
dropout = 0.2            # Regularization
learning_rate = 3e-4     # Standard for transformers
max_iters = 5000         # ~20-30 min training
```

### Memory Tips
- Use `torch.compile()` for faster training (PyTorch 2.0+)
- Enable automatic mixed precision if needed
- Monitor with `Activity Monitor` or `htop`

---

## Dataset
**Shakespeare Tiny** (~1MB): Perfect for learning
- URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Character vocabulary: ~65 unique characters
- Easy to see if model learns patterns (iambic pentameter, character names, etc.)

---

## No-Deadlock Guarantees

1. **No CUDA dependencies** - MPS is native, no driver issues
2. **Character-level tokenization** - No BPE/SentencePiece complexity
3. **Single file implementation** - No import chain issues
4. **Small dataset** - Fits in RAM, no streaming complexity
5. **Proven hyperparameters** - Based on nanoGPT (Karpathy) known to work
6. **Incremental building** - Test each component before combining

---

## Next Steps

When ready, we will implement in this order:
1. `gpt_from_scratch.py` - Main implementation
2. `train.py` - Training script
3. `generate.py` - Text generation
4. `visualize.py` - Attention visualization

**Reply "Start Phase 1" to begin environment setup.**
