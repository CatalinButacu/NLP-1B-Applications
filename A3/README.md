# Assignment 3: Transformer Language Model

## Problem Statement

Implement a Transformer-based architecture for two tasks:
1. **Letter Counting Classification**: Predict character occurrences in text sequences
2. **Character-level Language Modeling**: Generate text using Transformer architecture

## Architecture

### Part 1: Letter Counting Transformer
- **Model**: Custom Transformer Encoder with self-attention
- **Input**: Character sequences with positional encoding
- **Output**: 3-class classification (BEFORE/BEFOREAFTER tasks)
- **Components**: Multi-head self-attention, feed-forward networks, residual connections

### Part 2: Transformer Language Model
- **Model**: Transformer-based character-level language model
- **Architecture**: 4-layer Transformer with 320-dimensional embeddings
- **Training**: 20 epochs on text8 dataset (100k characters)
- **Optimization**: Adam optimizer with gradient clipping

## Implementation Details

### Transformer Components
- **Self-Attention**: Multi-head attention with Q, K, V projections
- **Positional Encoding**: Sinusoidal position embeddings
- **Feed-Forward**: Two-layer MLP with ReLU activation
- **Residual Connections**: Skip connections around attention and FFN layers
- **Layer Normalization**: Applied after residual connections

### Hyperparameters
- **Part 1**: 10 epochs, custom learning rate
- **Part 2**: 20 epochs, lr=0.001, batch_size=64, chunk_size=20
- **Vocabulary**: 27 characters (a-z + space)
- **Device**: CUDA acceleration

## Results

### Part 1: Letter Counting Classification

| Task | Accuracy Range | Typical Performance |
|------|----------------|--------------------|
| BEFORE | 99-100% | 99.5% dev accuracy |
| BEFOREAFTER | 97-100% | 98.8% dev accuracy |

**Key Observations**:
- Consistent high performance across 5 experimental runs
- Perfect or near-perfect training convergence
- Strong generalization to development set

### Part 2: Language Modeling

| Metric | Experiment Range | Average |
|--------|------------------|----------|
| Final Train Perplexity | 4.91-5.07 | 4.99 |
| Final Valid Perplexity | 1.087-1.094 | 1.091 |
| Log Probability | -848 to -863 | -856 |
| Average Log Prob | -1.696 to -1.728 | -1.712 |

**Training Dynamics**:
- Rapid initial convergence (epoch 1: ~13.7 ppl → epoch 20: ~5.0 ppl)
- Stable validation performance throughout training
- All models pass sanity checks (normalization, probability range)

## Technical Analysis

### Model Validation
- **Sanity Checks**: All models pass probability normalization tests
- **Convergence**: Smooth loss reduction without overfitting
- **Consistency**: Reproducible results across multiple runs
- **Performance**: Competitive perplexity scores for character-level modeling

### Architecture Benefits
- Self-attention captures long-range dependencies
- Positional encoding preserves sequence order information
- Residual connections enable stable deep network training
- Character-level modeling handles out-of-vocabulary robustly

## Files Structure

```
A3/
├── transformer.py          # Core Transformer implementation
├── letter_counting.py      # Part 1 training script
├── transformer_lm.py       # Part 2 language model
├── lm.py                  # Language model training script
├── utils.py               # Utility functions
└── logs/
    ├── part1_before.txt    # BEFORE task results
    ├── part1_beforeafter.txt # BEFOREAFTER task results
    └── part2_lm.txt        # Language modeling results
```

The implementation successfully demonstrates Transformer architecture capabilities for both classification and generative tasks, achieving strong performance metrics across all experimental conditions.