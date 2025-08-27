# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import string 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Please download them by running: import nltk; nltk.download('stopwords')")
    STOP_WORDS = set() # Fallback to an empty set if not available


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.label_mapping = ["entailment", "neutral", "contradiction"]

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the probability for entailment (index 0)
            entailment_prob = probs[0, 0].item()
            
            # Get the predicted class (highest probability)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            predicted_class = self.label_mapping[predicted_class_idx]

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        result = {"entailment_prob": entailment_prob, "predicted_class": predicted_class}
        del inputs, outputs, logits, probs
        gc.collect()

        return result


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def _preprocess_text(self, text: str) -> set:
        """
        Converts text to lowercase, removes punctuation, and returns a set of unique words.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        # Remove stop words
        words = {word for word in words if word not in STOP_WORDS}
        return words

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predicts if a fact is supported by any of the passages based on word recall.
        A fact is considered supported if the recall of its words in any passage
        is above a certain threshold.
        """
        fact_words = self._preprocess_text(fact)
        if not fact_words: # Handle empty fact string
            return "NS"

        max_recall = 0.0
        for passage_dict in passages:
            passage_text = passage_dict.get("text", "")
            passage_words = self._preprocess_text(passage_text)
            
            if not passage_words: # Handle empty passage
                continue

            common_words = fact_words.intersection(passage_words)
            recall = len(common_words) / len(fact_words)
            
            if recall > max_recall:
                max_recall = recall
        
        threshold = 0.6
        if max_recall >= threshold:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.nlp = spacy.load('en_core_web_sm')        
        self.word_overlap_threshold = 0.15      # Threshold for word overlap pruning - tuned for optimal performance       
        self.entailment_threshold = 0.7         # Threshold for entailment probability - tuned for optimal performance        
        self.fact_cache = {}                    # Cache for preprocessed facts to avoid redundant computation
        
    def _preprocess_text(self, text: str) -> set:
        """Convert text to lowercase, remove punctuation, and return set of words."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        # Remove stop words
        words = {word for word in words if word not in STOP_WORDS}
        return words
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
        
    def _calculate_word_overlap(self, fact_words: set, passage_words: set) -> float:
        """Calculate word overlap between fact and passage."""
        if not fact_words or not passage_words:
            return 0.0
            
        common_words = fact_words.intersection(passage_words)
        recall = len(common_words) / len(fact_words)
        return recall

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Predicts if a fact is supported by any of the passages based on word overlap and entailment."""  
        
        # Use cached preprocessed fact if available
        if fact in self.fact_cache:
            fact_words = self.fact_cache[fact]
        else:
            fact_words = self._preprocess_text(fact)
            self.fact_cache[fact] = fact_words
            
        if not fact_words:
            return "NS"
            
        max_entailment_prob = 0.0
        is_entailed = False
        
        # Process each passage
        for passage_dict in passages:
            passage_text = passage_dict.get("text", "")
            passage_words = self._preprocess_text(passage_text)
            
            # Skip if passage is empty
            if not passage_words:
                continue
                
            # Apply word overlap pruning
            overlap = self._calculate_word_overlap(fact_words, passage_words)
            if overlap < self.word_overlap_threshold:
                continue
                
            # Split passage into sentences
            sentences = self._split_into_sentences(passage_text)
            
            # Check entailment for each sentence
            for sentence in sentences:
                # Skip very short sentences that are unlikely to contain meaningful information
                if len(sentence.split()) < 3:
                    continue
                    
                # Check entailment
                result = self.ent_model.check_entailment(sentence, fact)
                
                # Update max entailment probability
                if result["entailment_prob"] > max_entailment_prob:
                    max_entailment_prob = result["entailment_prob"]
                
                # Check if entailed based on discrete prediction
                if result["predicted_class"] == "entailment":
                    is_entailed = True
                    
                # Early stopping if we found entailment
                if is_entailed:
                    break
                    
            # Early stopping if we found entailment
            if is_entailed:
                break
                
        # Clean up to prevent OOM issues
        gc.collect()

        # Make final decision based on entailment probability or discrete prediction
        if is_entailed or max_entailment_prob >= self.entailment_threshold:
            return "S"
        else:
            return "NS"            


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

