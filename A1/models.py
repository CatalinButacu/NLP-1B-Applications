# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import random
import math
import spacy
import matplotlib.pyplot as plt

SPACY = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm
SEED = 1234

np.random.seed(SEED)
random.seed(SEED)

############################################################################################################
#   Feature Extractors
############################################################################################################

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

### Acc >= 74% ###
class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNIGRAM_{word}")
            else:
                idx = self.indexer.index_of(f"UNIGRAM_{word}")
            if idx != -1:
                features[idx] += 1
        return features


### Acc >= 77% ###
class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # Strategic improvement: Add basic stopword filtering
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Strategic improvement: Basic sentiment word lists
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'loved', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'incredible', 'awesome', 'beautiful', 'enjoy', 'enjoyed'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated', 'boring', 'stupid', 'ridiculous', 'disappointing', 'poor', 'weak', 'annoying', 'disgusting', 'pathetic'}
        
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        
        # Step 1: Enhanced unigram features with basic filtering
        for word in sentence:
            word_lower = word.lower()
            # Strategic improvement: Filter very short words and basic stopwords
            if len(word) > 2 and word_lower not in self.stopwords:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"UNIGRAM_{word_lower}")
                else:
                    idx = self.indexer.index_of(f"UNIGRAM_{word_lower}")
                if idx != -1:
                    features[idx] += 1.0
        
        # Step 2: Enhanced bigram features with filtering
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i].lower(), sentence[i+1].lower()
            # Strategic improvement: Skip bigrams with stopwords
            if (len(word1) > 2 and len(word2) > 2 and 
                word1 not in self.stopwords and word2 not in self.stopwords):
                bigram = f"{word1}_{word2}"
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"BIGRAM_{bigram}")
                else:
                    idx = self.indexer.index_of(f"BIGRAM_{bigram}")
                if idx != -1:
                    features[idx] += 1.0
        
        # Step 3: Strategic addition - Sentence length feature (normalized)
        if add_to_indexer:
            length_idx = self.indexer.add_and_get_index("SENTENCE_LENGTH")
        else:
            length_idx = self.indexer.index_of("SENTENCE_LENGTH")
        if length_idx != -1:
            # Normalize sentence length to 0-1 range
            features[length_idx] = min(len(sentence) / 30.0, 1.0)
        
        # Step 4: Strategic addition - Basic sentiment polarity features
        pos_count = sum(1 for word in sentence if word.lower() in self.positive_words)
        neg_count = sum(1 for word in sentence if word.lower() in self.negative_words)
        
        if add_to_indexer:
            pos_idx = self.indexer.add_and_get_index("POSITIVE_WORD_COUNT")
            neg_idx = self.indexer.add_and_get_index("NEGATIVE_WORD_COUNT")
        else:
            pos_idx = self.indexer.index_of("POSITIVE_WORD_COUNT")
            neg_idx = self.indexer.index_of("NEGATIVE_WORD_COUNT")
        
        if pos_idx != -1:
            features[pos_idx] = pos_count
        if neg_idx != -1:
            features[neg_idx] = neg_count
                
        return features

### Accuracy: 670 / 872 = 0.768349 ###
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor -> Unigram++
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(SPACY.Defaults.stop_words)
        
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()        
        processed_sent = [word.lower() for word in sentence if word.lower() not in self.stopwords]

        for word in processed_sent:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNIGRAM_{word}")
            else:
                idx = self.indexer.index_of(f"UNIGRAM_{word}")
            if idx != -1:
                features[idx] += 1
                
        sentiment_words = {'great', 'good', 'bad', 'terrible', 'excellent', 'poor', 'amazing', 'awful'}
        for word in processed_sent:
            if word in sentiment_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"SENTIMENT_{word}")
                else:
                    idx = self.indexer.index_of(f"SENTIMENT_{word}")
                if idx != -1:
                    features[idx] = 1
                    
        return features
    

### Accuracy: 671 / 872 = 0.769495 ###
class BetterFeatureExtractor_V2(FeatureExtractor):
    """
    Feature extractor that uses TF-IDF weighting for unigram features.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(SPACY.Defaults.stop_words)
        self.doc_freqs = Counter()  # document frequency for each word
        self.num_docs = 0
        self.idf_values = {}
        self.initialized = False

    def get_indexer(self):
        return self.indexer
    
    def initialize_idf(self, examples: List[SentimentExample]):
        """
        Compute document frequencies and IDF values from a list of examples.
        """
        self.num_docs = len(examples)
        
        # count frequencies
        for ex in examples:
            processed_sent = [word.lower() for word in ex.words if word.lower() not in self.stopwords]
            
            word_set = set(processed_sent)
            for word in word_set:
                self.doc_freqs[word] += 1
        
        # compute IDF 
        for word, doc_freq in self.doc_freqs.items():
            self.idf_values[word] = math.log(self.num_docs / (doc_freq + 1)) + 1
            
        self.initialized = True


    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()

        processed_sent = [word.lower() for word in sentence if word.lower() not in self.stopwords]

        term_freqs = Counter(processed_sent)
        
        for word, tf in term_freqs.items():
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"TFIDF_{word}")
                features[idx] = tf

                if self.initialized:
                    idf = self.idf_values.get(word, math.log(self.num_docs) + 1)  # default IDF for unseen words
                    features[idx] *= idf
                
            else:
                idx = self.indexer.index_of(f"TFIDF_{word}")

                if idx != -1:
                    features[idx] = tf

                    if self.initialized:
                        idf = self.idf_values.get(word, math.log(self.num_docs) + 1)
                        features[idx] *= idf
            
        sentiment_words = {'great', 'good', 'bad', 'terrible', 'excellent', 'poor', 'amazing', 'awful'}
        for word in processed_sent:
            if word in sentiment_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"SENTIMENT_{word}")
                else:
                    idx = self.indexer.index_of(f"SENTIMENT_{word}")
                if idx != -1:
                    features[idx] = 1

        # negation features
        for i, word in enumerate(processed_sent):
            if word in {'not', 'no', 'never', "n't", 'cannot'} and i + 1 < len(processed_sent):
                next_word = processed_sent[i + 1]
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"NEG_{next_word}")
                else:
                    idx = self.indexer.index_of(f"NEG_{next_word}")
                if idx != -1:
                    features[idx] = 1
                    
        return features
                        


############################################################################################################
#   Classifiers
############################################################################################################

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = 0.0
        for feat_idx, feat_val in features.items():
            score += self.weights[feat_idx] * feat_val
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = 0.0
        for feat_idx, feat_val in features.items():
            score += self.weights[feat_idx] * feat_val
        return 1 if self._sigmoid(score) >= 0.5 else 0
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


############################################################################################################
#   Training
############################################################################################################

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    weights = np.zeros(len(feat_extractor.get_indexer()))
    
   # Train for several epochs with enhanced parameters for complex features
    num_epochs = 35  # Increased epochs for better convergence with trigrams
    learning_rate = 0.01  # Slightly lower for stability with complex features
    l2_reg = 0.0008  # Increased regularization for enhanced feature set
    
    progress = list()
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words)
            score = 0.0
            for feat_idx, feat_val in features.items():
                score += weights[feat_idx] * feat_val
            prediction = 1 if score >= 0 else 0
            
            # Update on mistake with L2 regularization
            if prediction != ex.label:
                for feat_idx, feat_val in features.items():
                    weights[feat_idx] += feat_val * (ex.label - prediction)
                # Apply L2 regularization
                weights *= (1 - l2_reg)

        progress.append(np.mean([1 if ex.label == PerceptronClassifier(weights, feat_extractor).predict(ex.words) else 0 for ex in train_exs]))
        print(f"EPOCH:{epoch} Accuracy:", progress[-1])
    
    plt.plot(progress)
    plt.savefig(f"train_perceptron_{len(feat_extractor.get_indexer())}.png")
    return PerceptronClassifier(weights, feat_extractor)

import string

def remove_punctuation(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    table = str.maketrans('', '', string.punctuation)
    for ex in train_exs:
        ex.words = [word.translate(table) for word in ex.words]
    return train_exs

def keep_only_letters(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    for ex in train_exs:
        ex.words = [''.join(filter(str.isalpha, word)) for word in ex.words]
    return train_exs

def remove_prepositions(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    prepositions = {
    'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by',
    'concerning', 'considering',
    'despite', 'down', 'during',
    'except',
    'for', 'from',
    'in', 'inside', 'into',
    'like',
    'near',
    'of', 'off', 'on', 'onto', 'out', 'outside', 'over',
    'past',
    'regarding',
    'since',
    'through', 'throughout', 'to', 'toward', 'towards',
    'under', 'underneath', 'until', 'unto', 'up', 'upon',
    'with', 'within', 'without'
    }
    for ex in train_exs:
        ex.words = [word for word in ex.words if word.lower() not in prepositions]
    return train_exs

def purge_text(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    #train_exs = remove_punctuation(train_exs)    # BEST SOLO Accuracy: 677 / 872 = 0.776376 (UNIGRAM)
    #train_exs = keep_only_letters(train_exs)     # BEST SOLO Accuracy: 678 / 872 = 0.777523 (UNIGRAM)
    train_exs = remove_prepositions(train_exs)   # BEST SOLO Accuracy: 683 / 872 = 0.783257 (UNIGRAM) 
    return train_exs

def purge_text_v2(train_exs: List[SentimentExample]) -> List[SentimentExample]: # MUCH MORE SLOWER..
    texts = [" ".join(ex.words) for ex in train_exs]
    docs = list(SPACY.pipe(texts))  # Use spaCy's pipe for batch processing

    # Preprocess each example
    for ex, doc in zip(train_exs, docs):
        # Remove prepositions (ADP) and non-alphabetic characters
        ex.words = [
            ''.join(filter(str.isalpha, token.text))  # Keep only alphabetic characters
            for token in doc
            if token.pos_ != "ADP"  # Remove prepositions
        ]

        # Remove empty strings (if any)
        ex.words = [word for word in ex.words if word]

    return train_exs


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    # No vocabulary building needed for basic bigram extractor
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(feat_extractor.get_indexer()))

    def activation(x):
        return 1.0 / (1.0 + math.exp(-x)) # sigmoid function
        
    # Optimized parameters for 77%+ target
    num_epochs = 25  # Slightly more epochs for better convergence
    initial_lr = 0.015  # Slightly higher learning rate
    decay_rate = 0.008  # Slower decay for more learning
    l2_reg = 0.0008  # Slightly less regularization
    progress = list()

    for epoch in range(1, num_epochs + 1):
        learning_rate = initial_lr / (1 + decay_rate * epoch)
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words)
            score = sum(weights[feat_idx] * feat_val for feat_idx, feat_val in features.items())
            prediction = activation(score)
            error = ex.label - prediction

            for feat_idx, feat_val in features.items():
                weights[feat_idx] += learning_rate * error * feat_val
            # Apply L2 regularization
            weights *= (1 - learning_rate * l2_reg)

        progress.append(np.mean([1 if ex.label == LogisticRegressionClassifier(weights, feat_extractor).predict(ex.words) else 0 for ex in train_exs]))
        print(f"EPOCH:{epoch} Accuracy:", progress[-1])

    plt.plot(progress)
    plt.savefig(f"train_logistic_regression_{len(feat_extractor.get_indexer())}.png")
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        if True:
            feat_extractor = BetterFeatureExtractor(Indexer())
        else:
            feat_extractor = BetterFeatureExtractor_V2(Indexer())
            feat_extractor.initialize_idf(train_exs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model