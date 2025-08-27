# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from collections import Counter
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class PrefixEmbeddings:
    """
    Generates embeddings based on the first 3 characters of each word.
    """
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings
        self.prefix_to_embedding = self._create_prefix_embeddings()

    def _create_prefix_embeddings(self):
        """
        Creates a mapping from prefixes to their corresponding embeddings.
        The embedding for a prefix is the average of the embeddings of all words that start with that prefix.
        """
        prefix_to_embedding = {}
        prefix_counter = Counter()

        # Iterate through all words in the vocabulary
        for word in self.word_embeddings.word_indexer.objs_to_ints.keys():  
            if len(word) >= 3:
                prefix = word[:3]
                if prefix not in prefix_to_embedding:
                    prefix_to_embedding[prefix] = np.zeros(self.word_embeddings.get_embedding_length())
                prefix_to_embedding[prefix] += self.word_embeddings.get_embedding(word)
                prefix_counter[prefix] += 1

        # Average the embeddings for each prefix
        for prefix in prefix_to_embedding:
            prefix_to_embedding[prefix] /= prefix_counter[prefix]

        return prefix_to_embedding

    def get_embedding(self, word):
        """
        Returns the embedding for a word based on its prefix.
        If the word is shorter than 3 characters, returns the UNK embedding.
        """
        if len(word) >= 3:
            prefix = word[:3]
            if prefix in self.prefix_to_embedding:
                return self.prefix_to_embedding[prefix]
        # Return the UNK embedding if the word is too short or the prefix is not found
        return self.word_embeddings.get_embedding("UNK")

class DeepAveragingNetwork(nn.Module):
    """
    A feedforward neural network that averages word embeddings and passes them through a few layers.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, word_embeddings):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x is a batch of word indices
        embedded = self.embedding_layer(x)  # (batch_size, seq_len, embedding_dim)
        avg_embedded = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        h1 = self.relu(self.fc1(avg_embedded))  # (batch_size, hidden_dim)
        h1 = self.dropout(h1)
        output = self.fc2(h1)  # (batch_size, output_dim)
        log_probs = self.log_softmax(output)  # (batch_size, output_dim)
        return log_probs


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    #def __init__(self):
    #    raise NotImplementedError

    def __init__(self, model, word_embeddings, prefix_embeddings=None):
        self.model = model
        self.word_embeddings = word_embeddings
        self.prefix_embeddings = prefix_embeddings

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Predicts the sentiment for a single sentence.
        :param ex_words: List of words in the sentence
        :param has_typos: Whether the sentence contains typos (not used in this implementation)
        :return: Predicted sentiment (0 or 1)
        """
        #word_indices = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
        #word_indices = [idx if idx != -1 else 1 for idx in word_indices]  # Replace unknown words with UNK
        #word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)
        #log_probs = self.model(word_indices_tensor)
        #prediction = torch.argmax(log_probs, dim=1).item()
        #return prediction
        if has_typos and self.prefix_embeddings:
            # Use prefix embeddings for typo setting
            word_indices = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
        else:
            # Use regular word embeddings for typo-free setting
            word_indices = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
        word_indices = [idx if idx != -1 else 1 for idx in word_indices]  # Replace unknown words with UNK
        word_indices_tensor = torch.tensor([word_indices], dtype=torch.long)
        log_probs = self.model(word_indices_tensor)
        prediction = torch.argmax(log_probs, dim=1).item()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        Predicts the sentiment for a batch of sentences.
        :param all_ex_words: List of sentences (each sentence is a list of words)
        :param has_typos: Whether the sentences contain typos (not used in this implementation)
        :return: List of predicted sentiments (0 or 1)
        """
        predictions = []
        for ex_words in all_ex_words:
            predictions.append(self.predict(ex_words, has_typos))
        return predictions


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    #raise NotImplementedError
     # Hyperparameters
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_dim = 100
    output_dim = 2  # Binary classification
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Initialize the model
    model = DeepAveragingNetwork(embedding_dim, hidden_dim, output_dim, word_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

     # Initialize prefix embeddings if training for typo setting
    prefix_embeddings = None
    if train_model_for_typo_setting:
        prefix_embeddings = PrefixEmbeddings(word_embeddings)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_exs)
        total_loss = 0.0

        for i in range(0, len(train_exs), batch_size):
            batch_exs = train_exs[i:i + batch_size]
            batch_words = [ex.words for ex in batch_exs]
            batch_labels = [ex.label for ex in batch_exs]

            # Convert words to indices
            batch_indices = []
            for words in batch_words:
                #word_indices = [word_embeddings.word_indexer.index_of(word) for word in words]
                #word_indices = [idx if idx != -1 else 1 for idx in word_indices]  # Replace unknown words with UNK
                #batch_indices.append(word_indices)
                if train_model_for_typo_setting:
                    # Use prefix embeddings for typo setting
                    word_indices = [word_embeddings.word_indexer.index_of(word) for word in words]
                else:
                    # Use regular word embeddings for typo-free setting
                    word_indices = [word_embeddings.word_indexer.index_of(word) for word in words]
                word_indices = [idx if idx != -1 else 1 for idx in word_indices]  # Replace unknown words with UNK
                batch_indices.append(word_indices)

            # Pad sequences to the same length
            max_len = max(len(indices) for indices in batch_indices)
            batch_indices = [indices + [0] * (max_len - len(indices)) for indices in batch_indices]  

            # Convert to tensors
            batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)
            batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

            # Forward pass
            log_probs = model(batch_indices_tensor)
            loss = criterion(log_probs, batch_labels_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_exs)}")

    return NeuralSentimentClassifier(model, word_embeddings, prefix_embeddings)

