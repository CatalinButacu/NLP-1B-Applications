# Assignment 2: Optimization and Deep Averaging Networks

## Overview

**Part 1**: Gradient-based optimization for quadratic functions with SGD and visualization  
**Part 2**: Deep Averaging Network for sentiment classification with typo-robust embeddings

**Dataset**: Rotten Tomatoes movie reviews (6,920 train / 872 dev / 1,821 test)  
**Embeddings**: GloVe 6B 300d pre-trained vectors

## Proposed Solution

## Architecture

1. **Optimization**: SGD with analytical gradients and trajectory visualization
2. **DAN**: Feedforward network with averaged embeddings (300→100→2)
3. **Typo Handling**: Prefix-based fallback embeddings (1-4 char prefixes)

### Optimization
- **Function**: `f(x,y) = (x-1)² + (y-1)²`, gradient `∇f = [2(x-1), 2(y-1)]`
- **SGD**: `x_{t+1} = x_t - η∇f(x_t)`, optimal step size 0.5
- **Visualization**: Matplotlib trajectory plotting

### Deep Averaging Network
- **Architecture**: Linear(300,100) + ReLU + Dropout(0.2) + Linear(100,2)
- **Input**: Mean-pooled word embeddings per sentence
- **Training**: Adam optimizer, cross-entropy loss
- **Typo Handling**: Prefix embeddings (1-4 chars) for OOV words

### Training
- **Hyperparameters**: lr=0.001, epochs=10, batch_size=1, dropout=0.2
- **Process**: Load data → Initialize embeddings → Train DAN → Evaluate

## Results

### Optimization
- Converges from [0,0] → [1,1] in 100 epochs with lr=0.1

### DAN Performance
| Setting | Train Acc | Dev Acc | F1 | Typo Acc | Typo F1 |
|---------|-----------|---------|----|---------|---------|
| Standard | 99.88% | **79.82%** | **0.8053** | - | - |
| Typo | 99.78% | 79.59% | 0.8022 | **73.51%** | **0.7220** |
| Trivial | 52.17% | 50.92% | 0.6748 | - | - |

### Analysis
- **Performance**: 79.82% dev accuracy with simple mean-pooling
- **Typo Robustness**: 6.31% accuracy drop (79.82% → 73.51%) on misspellings
- **Generalization**: Dropout prevents overfitting (99.88% train → 79.82% dev)

## Implementation

### Key Components
- `optimization.py`: Quadratic function, gradient, SGD with visualization
- `models.py`: PrefixEmbeddings, DeepAveragingNetwork, NeuralSentimentClassifier
- `neural_sentiment_classifier.py`: Training script with typo evaluation

### Usage
```bash
# Optimization
python optimization.py --lr 0.1

# DAN training
python neural_sentiment_classifier.py --model DAN

# DAN with typo evaluation
python neural_sentiment_classifier.py --model DAN --use_typo_setting
```

## Technical Notes
- **Gradient Implementation**: Analytical derivatives with numerical verification
- **Embedding Strategy**: GloVe → prefix fallback → random initialization
- **Architecture Choice**: Simple mean-pooling outperforms complex alternatives
- **Typo Handling**: Character-prefix approach maintains 92% relative performance