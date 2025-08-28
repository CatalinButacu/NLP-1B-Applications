# Natural Language Processing: Complete Assignment Portfolio

## Overview

This repository contains a comprehensive implementation suite covering fundamental to advanced Natural Language Processing techniques. The four assignments demonstrate a strategic progression from classical machine learning approaches to state-of-the-art neural architectures, showcasing the evolution of NLP methodologies and their practical applications.

## Assignment Breakdown

### Assignment 1: Classical Sentiment Classification
**Foundation: Feature Engineering & Traditional ML**

- **Problem**: Binary sentiment classification on movie reviews (Rotten Tomatoes dataset)
- **Techniques Implemented**:
  - **Perceptron Classifier** with bag-of-words features
  - **Logistic Regression** with advanced feature engineering
  - **Feature Extraction**: Unigrams, bigrams, better features
- **Key Learning**: Understanding the ML pipeline, feature design impact, and baseline establishment
- **Results**: Perceptron (85.3%), Logistic Regression (76.1%)

### Assignment 2: Neural Networks & Word Embeddings
**Evolution: From Sparse Features to Dense Representations**

- **Problem**: Enhanced sentiment classification using neural approaches
- **Techniques Implemented**:
  - **Deep Averaging Network (DAN)** architecture
  - **GloVe Word Embeddings** (50D and 300D)
  - **SGD Optimization** with manual gradient implementation
  - **PyTorch Integration** for neural network training
- **Key Innovation**: Transition from sparse bag-of-words to dense semantic representations
- **Results**: Deep Averaging Network (79.8%)

### Assignment 3: Transformer Architecture
**Breakthrough: Self-Attention & Language Modeling**

#### Part 1: Custom Transformer Implementation
- **Problem**: Letter counting task (BEFORE/BEFOREAFTER variants)
- **Architecture**: Built Transformer encoder from scratch
  - **Self-attention mechanism** with Q, K, V matrices
  - **Positional encoding** for sequence awareness
  - **Residual connections** and feed-forward networks
- **Results**: Transformer letter counting (99.3%)

#### Part 2: Character-Level Language Modeling
- **Problem**: Next-character prediction on text8 dataset (100M Wikipedia characters)
- **Architecture**: 4-layer Transformer with nn.TransformerEncoder
- **Hyperparameters**: d_model=320, 20 epochs, batch_size=64
- **Results**: Language modeling (5.45 perplexity)

### Assignment 4: LLM Output Analysis & Fact-Checking
**Application: Real-World AI Safety & Verification**

- **Problem**: Fact-checking ChatGPT-generated biographies against Wikipedia
- **Dataset**: FActScore dataset (221 human-annotated fact instances)
- **Methods Implemented**:
  - **Word Overlap Baseline**: Bag-of-words similarity
  - **Neural Entailment**: DeBERTa-v3-base-mnli-fever-anli
- **Results**: Word Overlap (76.9%), Neural Entailment (84.2%)
- **Advanced Analysis**:
  - **Error Categorization**: Semantic confusion, implicit information, temporal mismatch
  - **LLM Comparison**: Systematic evaluation of different model outputs
- **Impact**: Addresses critical AI safety concerns in factual accuracy

## Technical Stack
- **Python 3.8+**, **PyTorch**, **Transformers (Hugging Face)**
- **spaCy**, **NLTK**, **NumPy/SciPy**

## Repository Structure
```
Solutions/
├── A1/    # Classical ML & Feature Engineering
├── A2/    # Neural Networks & Embeddings  
├── A3/    # Transformer Architecture
└── A4/    # LLM Analysis & Fact-Checking
```

*Complete NLP pipeline from theoretical foundations to practical applications.*

