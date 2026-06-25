# GPT Learning Project - Quick Reference

## Project Structure

```
learn-gpt/
├── data/
│   └── input.txt          # Shakespeare dataset (downloaded)
├── checkpoints/
│   └── best_model.pt      # Trained model (created after training)
├── analysis/
│   └── *.png              # Attention visualizations (created by analyze.py)
├── data_pipeline.py       # Phase 2: Tokenization & batching
├── gpt_model.py           # Phase 3: GPT architecture
├── train.py               # Phase 4: Training loop
├── generate.py            # Phase 5: Text generation
├── analyze.py             # Phase 6: Visualization & experiments
├── verify_setup.py        # Phase 1: Setup verification
├── setup_mac.sh           # Phase 1: Mac setup script
└── requirements.txt       # Dependencies
```

---

## Phase 1: Environment Setup

**Goal:** Install Python, PyTorch, and download the dataset.

### Option A: Run the setup script
```bash
cd ~/path/to/learn-gpt
chmod +x setup_mac.sh
./setup_mac.sh
```

### Option B: Manual setup
```bash
# Navigate to project
cd ~/path/to/learn-gpt

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download Shakespeare dataset
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Verify setup
python verify_setup.py
```

### Expected output:
```
✅ Python Version        Python 3.11.x
✅ Virtual Environment   /path/to/.venv
✅ PyTorch               Version: 2.x.x
✅ MPS (M1 GPU)          GPU acceleration ready!
✅ NumPy                 Version: 1.x.x
✅ Shakespeare Dataset   1,115,394 characters
```

---

## Phase 2: Data Pipeline

**Goal:** Understand tokenization and batching.

### Run the demo:
```bash
source .venv/bin/activate
python data_pipeline.py
```

### Expected output:
```
Loading data...
Total characters: 1,115,394

Building tokenizer...
Vocabulary size: 65
Characters:  !$&'(),-.0123456789:;?ABC...xyz

TOKENIZER DEMO
Original:  'Hello, World!'
Encoded:   [20, 43, 50, 50, 53, 6, 1, 35, 53, 56, 50, 42, 2]
Decoded:   'Hello, World!'

THE PREDICTION TASK
Given: 'H'          → Predict: 'e'
Given: 'He'         → Predict: 'l'
Given: 'Hel'        → Predict: 'l'
...
```

---

## Phase 3: GPT Model

**Goal:** Understand the transformer architecture.

### Run the demo:
```bash
python gpt_model.py
```

### Expected output:
```
Model Configuration:
  vocab_size: 65
  n_embd: 384
  n_head: 6
  n_layer: 6
  block_size: 256
  dropout: 0.2

Total Parameters: 10,788,929

Forward Pass Test
  Input shape:  (4, 32)
  Logits shape: (4, 32, 65)
  Loss: 4.1744 (expected ~4.17 for random init)
```

---

## Phase 4: Training

**Goal:** Train the model on Shakespeare text.

### Run training:
```bash
python train.py
```

### Expected output (takes ~20-30 minutes):
```
Iteration     0 | Train Loss: 4.1744 | Val Loss: 4.1721 | LR: 3.00e-06
Iteration   500 | Train Loss: 2.0145 | Val Loss: 2.1032 | LR: 2.91e-04
Iteration  1000 | Train Loss: 1.6234 | Val Loss: 1.7821 | LR: 2.67e-04
Iteration  2000 | Train Loss: 1.3512 | Val Loss: 1.5234 | LR: 2.12e-04
Iteration  3000 | Train Loss: 1.1823 | Val Loss: 1.4612 | LR: 1.52e-04
Iteration  4000 | Train Loss: 1.0912 | Val Loss: 1.4423 | LR: 8.92e-05
Iteration  5000 | Train Loss: 1.0512 | Val Loss: 1.4523 | LR: 1.00e-05

Training Complete!
Total time: 25.3 minutes
Best validation loss: 1.4423

Generated text:
----------------------------------------
ROMEO:
What say'st thou? Hast thou not a word of comfort?
...
----------------------------------------
```

---

## Phase 5: Text Generation

**Goal:** Generate text with different sampling strategies.

### Interactive mode:
```bash
python generate.py -i
```

### Single generation:
```bash
# Basic
python generate.py -p "ROMEO:" -n 200

# With parameters
python generate.py -p "To be or not" -t 0.8 -k 40 --top-p 0.9 -n 300
```

### Demo sampling strategies:
```bash
python generate.py --demo
```

### Parameters:
| Flag | Description | Default |
|------|-------------|---------|
| `-p, --prompt` | Starting text | "" |
| `-n, --tokens` | Max tokens to generate | 200 |
| `-t, --temperature` | Randomness (0=greedy, 1=normal) | 0.8 |
| `-k, --top-k` | Sample from top k tokens | 40 |
| `--top-p` | Nucleus sampling threshold | 0.9 |
| `-i, --interactive` | Interactive mode | - |

### Interactive commands:
```
>>> ROMEO:              # Enter a prompt
>>> /temp 0.5           # Set temperature
>>> /topk 20            # Set top-k
>>> /topp 0.9           # Set top-p  
>>> /tokens 300         # Set max tokens
>>> /quit               # Exit
```

---

## Phase 6: Analysis & Experiments

**Goal:** Visualize attention patterns and experiment with hyperparameters.

### Show experiment configurations:
```bash
python analyze.py --experiments
```

### Visualize attention:
```bash
# All heads in layer 0
python analyze.py --attention "ROMEO:" --layer 0

# Specific head
python analyze.py --attention "ROMEO:" --layer 0 --head 2
```

### Analyze predictions:
```bash
python analyze.py --predictions "To be or not"
```

### Expected attention output:
Saves PNG files to `analysis/` folder showing heatmaps of attention weights.

---

## Quick Commands Summary

```bash
# Always activate venv first!
source .venv/bin/activate

# Phase 1: Verify setup
python verify_setup.py

# Phase 2: Test data pipeline
python data_pipeline.py

# Phase 3: Test model architecture
python gpt_model.py

# Phase 4: Train the model (~30 min)
python train.py

# Phase 5: Generate text
python generate.py -i

# Phase 6: Analyze model
python analyze.py --experiments
python analyze.py --attention "Hello" --layer 0
```

---

## Troubleshooting

### "MPS not available"
Your Mac may not support Metal. Training will use CPU (slower but works).

### "Checkpoint not found"
Run `python train.py` first to train and save a model.

### "CUDA out of memory"
Reduce `BATCH_SIZE` in train.py (try 32 or 16).

### Import errors
Make sure virtual environment is activated: `source .venv/bin/activate`

---

## Learning Checkpoints

After each phase, you should understand:

| Phase | Key Concepts |
|-------|--------------|
| 1 | Python venv, PyTorch MPS, dataset download |
| 2 | Tokenization, context windows, input/target shift |
| 3 | Embeddings, self-attention (Q,K,V), multi-head, FFN, residuals |
| 4 | Cross-entropy loss, AdamW, learning rate schedule, gradient clipping |
| 5 | Greedy vs sampling, temperature, top-k, top-p (nucleus) |
| 6 | Attention visualization, hyperparameter effects, overfitting |

---

## Next Steps

After completing all phases:

1. **Experiment with hyperparameters** - Try configurations in `analyze.py --experiments`
2. **Try different datasets** - Replace `data/input.txt` with your own text
3. **Scale up** - Increase model size (more layers, larger embeddings)
4. **Add features** - Implement learning rate warmup, gradient accumulation
5. **Study real implementations** - Look at nanoGPT, minGPT, HuggingFace transformers
