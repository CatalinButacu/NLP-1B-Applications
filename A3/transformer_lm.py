# transformer_lm.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from utils import Indexer

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerLM, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return self.log_softmax(output)

class TextDataset(Dataset):
    def __init__(self, text, vocab_index, chunk_size):
        self.text = text
        self.vocab_index = vocab_index
        self.chunk_size = chunk_size
        self.indices = [self.vocab_index.index_of(c) for c in self.text]

    def __len__(self):
        return (len(self.indices) - 1) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size

        input_seq_indices = [self.vocab_index.index_of(' ')] + self.indices[start_idx : start_idx + self.chunk_size - 1]
        target_seq_indices = self.indices[start_idx : start_idx + self.chunk_size]

        return torch.tensor(input_seq_indices, dtype=torch.long), torch.tensor(target_seq_indices, dtype=torch.long)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, chunk_size=20):
        self.model = model
        self.vocab_index = vocab_index
        self.chunk_size = chunk_size
        self.device = next(model.parameters()).device

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            # Prepend space as start-of-sequence token
            context_with_start = ' ' + context
            # Truncate context if longer than chunk_size
            if len(context_with_start) > self.chunk_size:
                 context_with_start = context_with_start[-self.chunk_size:]

            context_indices = [self.vocab_index.index_of(c) for c in context_with_start]
            context_tensor = torch.tensor([context_indices], dtype=torch.long).to(self.device)

            log_probs_all = self.model(context_tensor)
            # Get log probs for the *next* character after the context
            next_char_log_probs = log_probs_all[0, -1, :]
            return next_char_log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        with torch.no_grad():
            log_prob_sum = 0.0
            current_context = ' ' + context
            for char_to_predict in next_chars:
                # Prepare input tensor for the current context
                if len(current_context) > self.chunk_size:
                    current_context_proc = current_context[-self.chunk_size:]
                else:
                    current_context_proc = current_context

                context_indices = [self.vocab_index.index_of(c) for c in current_context_proc]
                context_tensor = torch.tensor([context_indices], dtype=torch.long).to(self.device)

                # Get log probabilities for the next character
                log_probs_all = self.model(context_tensor)
                next_char_log_probs = log_probs_all[0, -1, :]

                # Add the log probability of the actual next character
                char_idx = self.vocab_index.index_of(char_to_predict)
                log_prob_sum += next_char_log_probs[char_idx].item()

                # Update context for the next step
                current_context += char_to_predict

            return log_prob_sum


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Hyperparameters 
    d_model = 320       # Embedding dimension
    nhid = 720          # Dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4         # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8           # Number of heads in the multiheadattention models
    dropout = 0.1       # Dropout value
    epochs = 20         # Number of training epochs
    batch_size = 64
    lr = 0.001          # Learning rate
    chunk_size = 20     # Sequence length for training chunks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = len(vocab_index)
    model = TransformerLM(vocab_size, d_model, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TextDataset(train_text, vocab_index, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TextDataset(dev_text, vocab_index, chunk_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        num_batches = 0

        for i, (data, targets) in enumerate(train_dataloader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(data)
            
            # Reshape output and targets for loss calculation
            # Output: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # Targets: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate and print average loss and perplexity for the epoch
        avg_epoch_loss = total_loss / num_batches
        print('| end of epoch {:3d} | train loss {:5.3f} | train ppl {:5.3f}'.format(
              epoch, avg_epoch_loss, math.exp(avg_epoch_loss)), end='\t', flush=True)

        # Evaluate on the dev set
        evaluate(model, dev_dataloader, criterion, vocab_size, device)

    print("Training finished.")

    return NeuralLanguageModel(model, vocab_index, chunk_size)

def evaluate(eval_model, data_loader, criterion, vocab_size, device):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = eval_model(data)
            total_loss += criterion(output.view(-1, vocab_size), targets.view(-1)).item() * len(data)

    total_tokens = len(data_loader.dataset) * data_loader.dataset.chunk_size
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    print('| valid loss {:5.3f} | valid ppl {:5.3f}'.format(avg_loss, math.exp(avg_loss)))
    return avg_loss
