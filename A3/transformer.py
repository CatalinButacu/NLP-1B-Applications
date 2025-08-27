# transformer.py

import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.vocab_index = vocab_index # Store vocab_index
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        # Set batched=False as we are processing one example at a time in this assignment part
        self.positional_encoding = PositionalEncoding(d_model, num_positions, batched=False)

        # Transformer Layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(d_model, d_internal) for _ in range(num_layers)]
        )

        # Final Linear layer to map to output classes
        self.output_linear = nn.Linear(d_model, num_classes)

        # LogSoftmax for output probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # 1. Get embeddings
        # indices: [seq_len]
        # embedded: [seq_len, d_model]
        embedded = self.embedding(indices)

        # 2. Add positional encodings
        # embedded_with_pos: [seq_len, d_model]
        embedded_with_pos = self.positional_encoding(embedded)

        # 3. Pass through Transformer layers
        layer_output = embedded_with_pos
        attention_maps = []
        for i in range(self.num_layers):
            layer_output, attn_map = self.transformer_layers[i](layer_output)
            attention_maps.append(attn_map)

        # 4. Final linear layer and log softmax
        # output: [seq_len, d_model]
        # logits: [seq_len, num_classes]
        logits = self.output_linear(layer_output)

        # log_probs: [seq_len, num_classes]
        log_probs = self.log_softmax(logits)

        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        # Linear layers for Q, K, V
        self.query_linear = nn.Linear(d_model, d_internal)
        self.key_linear = nn.Linear(d_model, d_internal)
        self.value_linear = nn.Linear(d_model, d_internal)

        # Output linear layer for self-attention
        self.output_linear = nn.Linear(d_internal, d_model)

        # Feed-forward network layers
        self.ff_linear1 = nn.Linear(d_model, d_model * 4) # Often d_ff is 4*d_model
        self.relu = nn.ReLU()
        self.ff_linear2 = nn.Linear(d_model * 4, d_model)

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_vecs):
        """
        :param input_vecs: [seq_len, d_model] tensor
        :return: A tuple of (output vectors, attention map). Output vectors is [seq_len, d_model], attention map is [seq_len, seq_len]
        """
        # Self-attention part
        queries = self.query_linear(input_vecs)
        keys = self.key_linear(input_vecs)
        values = self.value_linear(input_vecs)

        # Calculate attention scores: (Q * K^T) / sqrt(d_k)
        # queries: [seq_len, d_internal], keys.T: [d_internal, seq_len]
        # scores: [seq_len, seq_len]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_internal)
        attn_map = self.softmax(scores)

        # Apply attention to values: attn_map * V
        # attn_map: [seq_len, seq_len], values: [seq_len, d_internal]
        # attended_values: [seq_len, d_internal]
        attended_values = torch.matmul(attn_map, values)

        # Pass through output linear layer
        attention_output = self.output_linear(attended_values)

        # First residual connection
        residual1_output = input_vecs + attention_output

        # Feed-forward part
        ff_output = self.ff_linear2(self.relu(self.ff_linear1(residual1_output)))

        # Second residual connection
        final_output = residual1_output + ff_output

        return final_output, attn_map


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train: List[LetterCountingExample], dev: List[LetterCountingExample]):
    # Determine model parameters
    vocab_size = len(train[0].vocab_index)
    num_positions = 20  # Max sequence length
    d_model = 64        # Model dimension
    d_internal = 32     # Internal dimension for self-attention
    num_classes = 3     # 0, 1, or 2 counts
    num_layers = 1      # Number of Transformer layers

    # Instantiate the model
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Set up the loss function
    # NLLLoss expects log probabilities (output of LogSoftmax) and class indices
    loss_fcn = nn.NLLLoss()

    num_epochs = 10 # Or adjust as needed
    print(f"Starting training for {num_epochs} epochs...")

    for t in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(t) # Seed for shuffling consistency
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)

        start_time = time.time()
        for ex_idx in ex_idxs:
            example = train[ex_idx]

            # Zero gradients before the forward pass
            model.zero_grad()

            # Forward pass
            log_probs, _ = model.forward(example.input_tensor)

            # Calculate loss
            # log_probs shape: [seq_len, num_classes] (e.g., 20x3)
            # example.output_tensor shape: [seq_len] (e.g., 20)
            loss = loss_fcn(log_probs, example.output_tensor)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch {t+1}/{num_epochs} finished in {epoch_time:.2f}s, Loss: {loss_this_epoch / len(train):.4f}")
        # Optional: Evaluate on dev set periodically
        # if (t + 1) % 2 == 0:
        #     print(f"Evaluating on dev set after epoch {t+1}...")
        #     model.eval()
        #     with torch.no_grad():
        #         decode(model, dev)
        #     model.train()


    model.eval() # Set the model to evaluation mode after training
    print("Training complete.")
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
