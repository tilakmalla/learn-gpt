# GPT Learning Project - Complete Guide

---

# 🎉 YES, YOU BUILT A GPT MODEL!

You have implemented a **Generative Pre-trained Transformer (GPT)** from scratch. This is the same fundamental architecture behind:
- **GPT-2** (OpenAI, 2019)
- **GPT-3** (OpenAI, 2020)
- **GPT-4** (OpenAI, 2023)
- **ChatGPT**

The only differences between your model and these are:
1. **Scale**: Your model has ~10 million parameters. GPT-3 has 175 billion.
2. **Data**: You trained on 1MB of Shakespeare. GPT-3 trained on hundreds of GB.
3. **Training time**: You trained for 30 minutes. GPT-3 trained for months on thousands of GPUs.

**But the core architecture is identical.** You understand how the most important AI breakthrough of the decade works.

---

# How to Test If Your Model Works

## Step 1: Verify Setup (Phase 1)
```bash
cd ~/path/to/learn-gpt
source .venv/bin/activate
python verify_setup.py
```

**What to look for:**
- All checks should show ✅
- MPS (M1 GPU) should say "GPU acceleration ready!"

## Step 2: Test Data Pipeline (Phase 2)
```bash
python data_pipeline.py
```

**What to look for:**
- Tokenizer encodes and decodes correctly: `'Hello'` → `[numbers]` → `'Hello'`
- Vocabulary size should be 65 (Shakespeare's unique characters)

## Step 3: Test Model Architecture (Phase 3)
```bash
python gpt_model.py
```

**What to look for:**
- Total parameters: ~10,788,929
- Initial loss: ~4.17 (this is `-log(1/65)` = random guessing among 65 characters)
- No errors during forward pass

## Step 4: Train the Model (Phase 4) - THE MAIN TEST
```bash
python train.py
```

**What to look for (this proves learning is happening):**

| Iteration | Train Loss | Val Loss | What It Means |
|-----------|------------|----------|---------------|
| 0 | ~4.17 | ~4.17 | Random guessing (expected) |
| 500 | ~2.0 | ~2.1 | Learning patterns! |
| 2000 | ~1.4 | ~1.5 | Learning words and structure |
| 5000 | ~1.0 | ~1.4 | Good fit, slight overfitting |

**Key insight:** If loss goes DOWN, your model is LEARNING.

## Step 5: Test Generation (Phase 5)
```bash
python generate.py -i
```

**What to look for:**
- Generated text should look like Shakespeare (character names, dialogue format)
- Not perfect, but recognizable patterns

**Example of SUCCESS:**
```
>>> ROMEO:
ROMEO:
What say'st thou? I have seen the day
That I have worn a visor and could tell
A whispering tale in a fair lady's ear...
```

**Example of FAILURE (untrained model):**
```
>>> ROMEO:
xK3$)mNz!qR&*vB2...
```

## Step 6: Visualize What the Model Learned (Phase 6)
```bash
python analyze.py --attention "ROMEO:" --layer 0
```

**What to look for:**
- Attention heatmaps saved to `analysis/` folder
- Patterns in attention (not random noise)

---

# Project Structure

```
learn-gpt/
├── data/
│   └── input.txt          # Shakespeare dataset (~1MB, ~1 million characters)
├── checkpoints/
│   └── best_model.pt      # Your trained model (~40MB)
├── analysis/
│   └── *.png              # Attention visualization images
│
├── data_pipeline.py       # Phase 2: Tokenization & batching
├── gpt_model.py           # Phase 3: GPT architecture (THE CORE)
├── train.py               # Phase 4: Training loop
├── generate.py            # Phase 5: Text generation
├── analyze.py             # Phase 6: Visualization & experiments
│
├── verify_setup.py        # Phase 1: Setup verification
├── setup_mac.sh           # Phase 1: Mac setup script
├── requirements.txt       # Dependencies
├── GPT_LEARNING_PROMPT.md # Original learning roadmap
└── QUICKSTART.md          # This file
```

---

# What You Learned - Detailed Breakdown

---

## Phase 1: Environment Setup

**What you learned:**
- **Virtual environments** isolate project dependencies
- **PyTorch** is the deep learning framework (like TensorFlow, but more Pythonic)
- **MPS (Metal Performance Shaders)** allows M1 Macs to use GPU acceleration
- **The dataset** is ~1MB of Shakespeare text - small but enough to learn patterns

**Key concept: GPU acceleration**
```
CPU: Process one operation at a time (sequential)
GPU: Process thousands of operations in parallel

Matrix multiplication (the core of neural networks):
  CPU: 1000x1000 matrix multiply → ~1 second
  GPU: 1000x1000 matrix multiply → ~0.001 seconds
```

### Commands:
```bash
cd ~/path/to/learn-gpt
chmod +x setup_mac.sh
./setup_mac.sh
# OR manual:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_setup.py
```

---

## Phase 2: Data Pipeline

**What you learned:**

### 1. Tokenization: Converting Text to Numbers
Neural networks can't read text - they only understand numbers.

```
"Hello" → [20, 43, 50, 50, 53]

Character-level tokenization (what we use):
  'H' → 20
  'e' → 43
  'l' → 50
  'o' → 53

Each unique character gets a unique number (0-64 for Shakespeare)
```

**Why character-level?**
- Simple to implement and understand
- Vocabulary size is small (65 characters)
- Real GPT uses "subword" tokenization (BPE) with ~50,000 tokens

### 2. The Prediction Task: Next-Token Prediction
GPT learns by predicting the next character given previous characters.

```
Training example from "Hello!":

Input (x):  H  e  l  l  o
Target (y): e  l  l  o  !

At position 0: See "H", predict "e"
At position 1: See "He", predict "l"
At position 2: See "Hel", predict "l"
At position 3: See "Hell", predict "o"
At position 4: See "Hello", predict "!"
```

**This is called "autoregressive" modeling** - each prediction depends on all previous tokens.

### 3. Context Window (block_size = 256)
The model can only "see" 256 characters at a time.

```
Text: "To be or not to be, that is the question..."

Context window: [To be or not to be, that is the qu]  ← 256 chars
                 ↑                                 ↑
              start                              end

The model predicts the next character based only on this window.
```

### 4. Batching: Processing Multiple Sequences in Parallel
Instead of one sequence at a time, we process 64 sequences simultaneously.

```
Batch (batch_size=64, block_size=256):

Sequence 1: "ROMEO: What light..."    (256 chars)
Sequence 2: "JULIET: O Romeo..."      (256 chars)
Sequence 3: "HAMLET: To be or..."     (256 chars)
...
Sequence 64: "MACBETH: Is this..."    (256 chars)

Shape: (64, 256) = 16,384 tokens processed at once!
```

### Commands:
```bash
python data_pipeline.py
```

---

## Phase 3: GPT Model Architecture (THE CORE)

**What you learned:**

This is the heart of the transformer. You implemented each component from scratch.

### The Architecture (what happens to input tokens):

```
Input: "ROMEO:" → [30, 27, 25, 17, 27, 10]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  1. TOKEN EMBEDDING                                          │
│     Each token ID → 384-dimensional vector                   │
│     [30, 27, 25, 17, 27, 10] → (6, 384) matrix               │
├─────────────────────────────────────────────────────────────┤
│  2. POSITIONAL EMBEDDING                                     │
│     Add position information (transformers have no order!)   │
│     Position [0,1,2,3,4,5] → (6, 384) matrix                 │
│     Combined: token_emb + pos_emb = (6, 384)                 │
├─────────────────────────────────────────────────────────────┤
│  3. TRANSFORMER BLOCKS (×6)                                  │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  Multi-Head Self-Attention                          │ │
│     │  → Each token attends to previous tokens            │ │
│     │  → 6 heads learn different patterns                 │ │
│     │                                                     │ │
│     │  Feed-Forward Network                               │ │
│     │  → Process each position independently              │ │
│     │  → 384 → 1536 → 384 (expand then contract)          │ │
│     │                                                     │ │
│     │  Residual Connections + Layer Normalization         │ │
│     │  → Help gradients flow through deep network         │ │
│     └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  4. OUTPUT PROJECTION                                        │
│     (6, 384) → (6, 65) = probability over 65 characters      │
│     For each position: "what character comes next?"          │
└─────────────────────────────────────────────────────────────┘
                              ↓
Output: Probabilities for next character at each position
```

### Self-Attention: The Key Innovation

**The problem:** How does a token know about other tokens?

**The solution:** Each token creates three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

```
Example: "The cat sat"

Token "sat" creates a Query: "I'm a verb, looking for my subject"
Token "cat" has a Key: "I'm a noun, could be a subject"
Token "The" has a Key: "I'm an article"

Attention scores:
  sat → cat: HIGH (verb looking at its subject)
  sat → The: LOW (verb doesn't care much about articles)

Final representation of "sat" = weighted sum of all Values,
weighted by attention scores.
```

**The math:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Causal masking:** Tokens can only attend to PREVIOUS tokens (not future ones).
```
Token positions: 0  1  2  3  4
Mask:           [1, 0, 0, 0, 0]  ← Position 0 sees only itself
                [1, 1, 0, 0, 0]  ← Position 1 sees 0 and itself
                [1, 1, 1, 0, 0]  ← Position 2 sees 0, 1, and itself
                [1, 1, 1, 1, 0]  ← Position 3 sees 0, 1, 2, and itself
                [1, 1, 1, 1, 1]  ← Position 4 sees everything before it
```

### Multi-Head Attention: Different Perspectives
Instead of one attention pattern, we have 6 heads running in parallel.

```
Head 1: Might learn syntax (subject-verb agreement)
Head 2: Might learn punctuation patterns
Head 3: Might learn character names in dialogue
Head 4: Might learn rhyme patterns
Head 5: Might learn proximity (nearby words)
Head 6: Might learn something else entirely

All heads are concatenated and projected back to 384 dimensions.
```

### Feed-Forward Network: Per-Token Processing
After attention aggregates information, FFN processes each position independently.

```
Input: 384 dimensions
  ↓
Linear: 384 → 1536 (expand by 4x)
  ↓
GELU activation (smooth non-linearity)
  ↓
Linear: 1536 → 384 (contract back)
  ↓
Output: 384 dimensions
```

### Residual Connections: Gradient Highways
```
Without residuals:        With residuals:
Input → Block → Output    Input → Block → Output
                                ↘      ↗
                                  ADD

Gradients must flow        Gradients can "skip"
through every layer        directly through
(can vanish/explode)       (stable training)
```

### Parameter Count
```
Token embedding:      65 × 384 =       24,960
Position embedding:  256 × 384 =       98,304
Per transformer block:              ~1,770,000
  × 6 blocks:                      10,620,000
Final layer norm + output:            ~45,000
────────────────────────────────────────────────
TOTAL:                             ~10,788,929 parameters
```

### Commands:
```bash
python gpt_model.py
```

---

## Phase 4: Training

**What you learned:**

### 1. Loss Function: Cross-Entropy
Measures how "surprised" the model is by the correct answer.

```
Model predicts probabilities for next character:
  P('a') = 0.1, P('b') = 0.7, P('c') = 0.2

If correct answer is 'b':
  Loss = -log(0.7) = 0.36  (low loss, model was confident)

If correct answer is 'c':
  Loss = -log(0.2) = 1.61  (higher loss, model was wrong)

If correct answer is 'a':
  Loss = -log(0.1) = 2.30  (high loss, model was very wrong)
```

**Initial loss:** ~4.17 = -log(1/65) = random guessing among 65 characters
**Final loss:** ~1.0-1.4 = model is much better than random

### 2. AdamW Optimizer
Updates weights to minimize loss.

```
Basic gradient descent:
  weight = weight - learning_rate × gradient

AdamW adds:
  - Momentum: Keep moving in same direction (faster convergence)
  - Adaptive learning rates: Each parameter gets its own rate
  - Weight decay: Slightly shrink weights (prevents overfitting)
```

### 3. Learning Rate Schedule
```
Learning Rate
     ↑
     │    ╭────────╮
     │   ╱          ╲
     │  ╱            ╲
     │ ╱              ╲____
     └────────────────────────→ Iteration
       ↑        ↑
    Warmup   Cosine Decay

Warmup (first 100 steps):
  - Start slow to let model "settle"
  - Prevents early instability

Cosine decay (rest of training):
  - Gradually reduce learning rate
  - Fine-tune as we approach optimum
```

### 4. The Training Loop
```python
for iteration in range(5000):
    # 1. Get random batch of 64 sequences
    x, y = dataset.get_batch('train', batch_size=64)
    
    # 2. Forward pass: compute predictions and loss
    logits, loss = model(x, y)
    
    # 3. Backward pass: compute gradients
    loss.backward()  # PyTorch magic: automatic differentiation
    
    # 4. Update weights
    optimizer.step()
    
    # 5. Clear gradients for next iteration
    optimizer.zero_grad()
```

### 5. Overfitting vs Generalization
```
Iteration  Train Loss  Val Loss   Status
─────────────────────────────────────────────
    0        4.17       4.17      Random (expected)
  500        2.00       2.10      Learning! (good)
 2000        1.40       1.50      Still learning (good)
 4000        1.10       1.44      Gap growing (watch out)
 5000        1.00       1.45      Slight overfitting (OK)
10000        0.80       1.60      Overfitting! (stop)

If val_loss stops improving while train_loss keeps dropping,
the model is memorizing training data instead of learning patterns.
```

### Commands:
```bash
python train.py
# Takes ~20-30 minutes on M1 Mac
```

---

## Phase 5: Text Generation

**What you learned:**

### 1. Autoregressive Generation
Generate one token at a time, feeding output back as input.

```
Start:   "ROMEO:"
Step 1:  "ROMEO:" → model predicts 'W' → "ROMEO:W"
Step 2:  "ROMEO:W" → model predicts 'h' → "ROMEO:Wh"
Step 3:  "ROMEO:Wh" → model predicts 'a' → "ROMEO:Wha"
Step 4:  "ROMEO:Wha" → model predicts 't' → "ROMEO:What"
...continue for max_tokens...
```

### 2. Sampling Strategies

**Greedy (temperature=0):**
```
Always pick the highest probability token.
Deterministic but often repetitive and boring.

Probabilities: [a:0.4, b:0.35, c:0.25]
Greedy picks: 'a' (always)
```

**Temperature Sampling:**
```
Scale probabilities before sampling.

Original:     [a:0.4, b:0.35, c:0.25]
Temp=0.5:     [a:0.6, b:0.3, c:0.1]   ← More confident
Temp=1.0:     [a:0.4, b:0.35, c:0.25] ← Original
Temp=2.0:     [a:0.35, b:0.33, c:0.32] ← More uniform

Low temp = safe, predictable
High temp = creative, risky
```

**Top-k Sampling:**
```
Only consider the k most likely tokens.

Probabilities: [a:0.4, b:0.3, c:0.2, d:0.05, e:0.03, f:0.02]
Top-k (k=3):   [a:0.44, b:0.33, c:0.22, d:0, e:0, f:0]

Prevents sampling very unlikely tokens (reduces nonsense).
```

**Top-p (Nucleus) Sampling:**
```
Keep smallest set of tokens with cumulative probability ≥ p.

Probabilities: [a:0.4, b:0.3, c:0.2, d:0.1]
Cumulative:    [0.4, 0.7, 0.9, 1.0]
Top-p (p=0.9): Keep a, b, c (cumsum reaches 0.9)

More adaptive than top-k - keeps more tokens when uncertain,
fewer when confident.
```

### Commands:
```bash
# Interactive mode
python generate.py -i

# Single generation
python generate.py -p "ROMEO:" -n 200 -t 0.8

# Compare strategies
python generate.py --demo
```

---

## Phase 6: Analysis & Experiments

**What you learned:**

### 1. Attention Visualization
See which tokens each position "looks at."

```
Sentence: "The cat sat"

Attention pattern for "sat":
  The: ▓░░░░ (10%)
  cat: ▓▓▓▓▓▓▓ (70%)   ← "sat" attends strongly to "cat"
  sat: ▓▓░░░ (20%)

This shows the model learned that "sat" relates to "cat" (subject-verb).
```

### 2. Hyperparameter Effects

| Parameter | Increase Effect | Decrease Effect |
|-----------|-----------------|-----------------|
| n_embd | More capacity, slower | Less capacity, faster |
| n_head | More attention patterns | Fewer patterns |
| n_layer | Deeper network, more capacity | Shallower, faster |
| block_size | Longer context, more memory | Shorter context |
| dropout | More regularization | Less regularization |
| learning_rate | Faster learning, unstable | Slower, stable |

### 3. Diagnosing Problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss doesn't decrease | LR too low or bug | Increase LR, check code |
| Loss explodes (NaN) | LR too high | Decrease LR |
| Val loss >> train loss | Overfitting | More dropout, less capacity |
| Both losses high | Underfitting | More capacity, longer training |
| Repetitive output | Greedy/low temp | Increase temperature |
| Nonsense output | High temp/untrained | Lower temp, train more |

### Commands:
```bash
# Show experiment configs
python analyze.py --experiments

# Visualize attention
python analyze.py --attention "ROMEO:" --layer 0

# Analyze predictions
python analyze.py --predictions "To be or not"
```

---

# Quick Command Reference

```bash
# ALWAYS activate virtual environment first!
source .venv/bin/activate

# Phase 1: Verify your setup works
python verify_setup.py

# Phase 2: Test tokenization and batching
python data_pipeline.py

# Phase 3: Test model architecture (no training)
python gpt_model.py

# Phase 4: TRAIN THE MODEL (~20-30 minutes)
python train.py

# Phase 5: Generate text (after training)
python generate.py -i                    # Interactive mode
python generate.py -p "ROMEO:" -n 200    # Single generation
python generate.py --demo                # Compare strategies

# Phase 6: Analyze (after training)
python analyze.py --experiments                    # Show configs
python analyze.py --attention "ROMEO:" --layer 0  # Visualize attention
python analyze.py --predictions "To be"           # Analyze predictions
```

---

# Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "MPS not available" | Mac doesn't support Metal | Training uses CPU (slower but works) |
| "Checkpoint not found" | Haven't trained yet | Run `python train.py` first |
| "No module named torch" | Venv not activated | Run `source .venv/bin/activate` |
| Loss is NaN | Learning rate too high | Reduce `LEARNING_RATE` in train.py |
| Out of memory | Batch too large | Reduce `BATCH_SIZE` in train.py |
| Generated text is garbage | Model not trained enough | Train longer or check checkpoint |

---

# Summary: What You Built

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR GPT IMPLEMENTATION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: "ROMEO:"                                                    │
│           ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DATA PIPELINE (data_pipeline.py)                            │   │
│  │   Tokenize: "ROMEO:" → [30, 27, 25, 17, 27, 10]             │   │
│  │   Batch: Combine 64 sequences for parallel processing       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ GPT MODEL (gpt_model.py)                                    │   │
│  │   Token Embedding: IDs → 384-dim vectors                    │   │
│  │   Position Embedding: Add position information              │   │
│  │   6× Transformer Blocks:                                    │   │
│  │     - Multi-Head Self-Attention (6 heads)                   │   │
│  │     - Feed-Forward Network (384→1536→384)                   │   │
│  │     - Residual connections + LayerNorm                      │   │
│  │   Output: Probabilities over 65 characters                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TRAINING (train.py)                                         │   │
│  │   Loss: Cross-entropy (how wrong were predictions?)         │   │
│  │   Optimizer: AdamW (update weights to reduce loss)          │   │
│  │   Schedule: Warmup + Cosine decay                           │   │
│  │   5000 iterations → Loss: 4.17 → ~1.4                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ GENERATION (generate.py)                                    │   │
│  │   Autoregressive: Generate one token at a time              │   │
│  │   Sampling: Temperature, Top-k, Top-p                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ↓                                                         │
│  OUTPUT: "ROMEO: What say'st thou? I have seen..."              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Numbers

| Metric | Value | Meaning |
|--------|-------|---------|
| Parameters | 10,788,929 | Learnable numbers in the model |
| Vocabulary | 65 | Unique characters |
| Context window | 256 | Max characters model can "see" |
| Embedding dim | 384 | Size of token vectors |
| Attention heads | 6 | Parallel attention patterns |
| Transformer blocks | 6 | Depth of the network |
| Training iterations | 5,000 | Forward+backward passes |
| Batch size | 64 | Sequences per iteration |
| Initial loss | 4.17 | Random guessing |
| Final loss | ~1.0-1.4 | Learned patterns |

## What Each Phase Taught You

| Phase | Concept | Why It Matters |
|-------|---------|----------------|
| 1 | Environment | GPU acceleration makes training 100x faster |
| 2 | Tokenization | Neural nets need numbers, not text |
| 2 | Batching | Parallelism = efficiency |
| 3 | Embeddings | Dense vectors capture meaning |
| 3 | Self-Attention | Tokens communicate with each other |
| 3 | Multi-Head | Multiple perspectives simultaneously |
| 3 | FFN | Non-linear transformations |
| 3 | Residuals | Deep networks can actually train |
| 4 | Cross-Entropy | Quantify prediction quality |
| 4 | AdamW | Smart weight updates |
| 4 | LR Schedule | Stability at start, precision at end |
| 5 | Temperature | Control randomness |
| 5 | Top-k/Top-p | Prevent unlikely tokens |
| 6 | Attention Viz | See what model learned |

---

# Next Steps

Now that you understand GPT from scratch:

1. **Experiment with hyperparameters**
   - Try configs in `python analyze.py --experiments`
   - See how model size affects performance

2. **Try different datasets**
   - Replace `data/input.txt` with your own text
   - Try code, poetry, legal documents, etc.

3. **Scale up**
   - More layers, larger embeddings
   - Longer context windows
   - More training data

4. **Study production implementations**
   - [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
   - [minGPT](https://github.com/karpathy/minGPT) - minimal implementation
   - [HuggingFace Transformers](https://github.com/huggingface/transformers)

5. **Learn about modern improvements**
   - Flash Attention (faster attention)
   - RoPE (better positional encoding)
   - KV-Cache (faster inference)
   - RLHF (how ChatGPT is fine-tuned)

---

**Congratulations! You now understand the architecture behind the most important AI breakthrough of the decade.**
