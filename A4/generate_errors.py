# generate_errors.py

import json
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from factcheck import *
from error_analyzer import generate_errors_file, analyze_errors


def read_passages(path):
    """Read passages from a file and create a dictionary mapping facts to passages."""
    fact_to_passage_dict = {}
    with open(path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict_data = json.loads(nextline)
            name = dict_data["name"]
            for passage in dict_data["passages"]:
                if passage["title"] != name:
                    raise Exception(f"Couldn't find a match for: {name} {passage['title']}")
            fact_to_passage_dict[dict_data["sent"]] = dict_data["passages"]
    return fact_to_passage_dict


def read_fact_examples(labeled_facts_path, fact_to_passage_dict):
    """Read labeled fact examples and construct a dataset.
    
    Returns:
        examples: List of dictionaries with 'fact', 'passages', and 'label' keys
        facts: List of fact strings
        labels: List of label strings ('S' or 'NS')
    """
    examples = []
    facts = []
    labels = []
    
    with open(labeled_facts_path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict_data = json.loads(nextline)
            if dict_data["annotations"] is not None:
                for sent in dict_data["annotations"]:
                    if sent["human-atomic-facts"] is not None:
                        for fact in sent["human-atomic-facts"]:
                            if fact["text"] in fact_to_passage_dict:
                                # Convert 'IR' label to 'NS' for consistency
                                label = 'NS' if fact["label"] == 'IR' else fact["label"]
                                
                                examples.append({
                                    'fact': fact["text"],
                                    'passages': fact_to_passage_dict[fact["text"]],
                                    'label': label
                                })
                                facts.append(fact["text"])
                                labels.append(label)
    
    return examples, facts, labels


def run_model_and_generate_errors(model_type, labeled_facts_path, passages_path, output_path):
    """Run a fact-checking model and generate an errors.json file."""
    print(f"Running {model_type} model and generating errors file...")
    
    # Load passages and examples
    fact_to_passage_dict = read_passages(passages_path)
    examples, facts, labels = read_fact_examples(labeled_facts_path, fact_to_passage_dict)
    
    # Initialize the appropriate fact checker
    fact_checker = None
    if model_type == "random":
        fact_checker = RandomGuessFactChecker()
    elif model_type == "always_entail":
        fact_checker = AlwaysEntailedFactChecker()
    elif model_type == "word_overlap":
        fact_checker = WordRecallThresholdFactChecker()
    elif model_type == "parsing":
        fact_checker = DependencyRecallThresholdFactChecker()
    elif model_type == "entailment":
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer)
        fact_checker = EntailmentFactChecker(ent_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get model predictions
    print("Making predictions...")
    predictions = []
    for example in examples:
        pred = fact_checker.predict(example['fact'], example['passages'])
        predictions.append(pred)
    
    # Generate errors file by comparing predictions with ground truth
    generate_errors_file(predictions, labels, dict(zip(facts, [example['passages'] for example in examples])), output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate and analyze fact-checking model errors')
    parser.add_argument('--model', type=str, default='entailment',
                        choices=['random', 'always_entail', 'word_overlap', 'parsing', 'entailment'],
                        help='Fact-checking model to use')
    parser.add_argument('--labels', type=str, default='data/dev_labeled_ChatGPT.jsonl',
                        help='Path to labeled facts')
    parser.add_argument('--passages', type=str, default='data/passages_bm25_ChatGPT_humfacts.jsonl',
                        help='Path to passages')
    parser.add_argument('--output', type=str, default='errors.json',
                        help='Path to save errors file')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze errors after generating them')
    parser.add_argument('--categorize', action='store_true',
                        help='Interactively categorize errors')
    
    args = parser.parse_args()
    
    # Run model and generate errors
    errors_file = run_model_and_generate_errors(
        args.model, args.labels, args.passages, args.output
    )
    
    # Analyze errors if requested
    if args.analyze:
        if args.categorize:
            analyze_errors(errors_file, 'categorized_' + args.output)
        else:
            analyze_errors(errors_file)


if __name__ == "__main__":
    main()