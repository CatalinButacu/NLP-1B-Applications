import json
import argparse
import os


def load_errors(file_path):
    """Load errors from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_errors_file(model_predictions, ground_truth, passages, output_path):
    """Generate an errors.json file by comparing model predictions with ground truth.
    
    Args:
        model_predictions: List of model predictions ('S' or 'NS')
        ground_truth: List of ground truth labels ('S' or 'NS')
        passages: Dictionary mapping facts to their passages
        output_path: Path to save the errors.json file
    """
    false_positives = []
    false_negatives = []
    
    # Get the list of facts from the passages dictionary keys
    facts = list(passages.keys())
    
    for i, (pred, truth) in enumerate(zip(model_predictions, ground_truth)):
        if i < len(facts):
            fact = facts[i]
            
            # False positive: model predicted Supported when it's Not Supported
            if pred == 'S' and truth == 'NS':
                false_positives.append({
                    'fact': fact,
                    'ground_truth': truth,
                    'prediction': pred,
                    'passages': passages.get(fact, [])
                })
            
            # False negative: model predicted Not Supported when it's Supported
            elif pred == 'NS' and truth == 'S':
                false_negatives.append({
                    'fact': fact,
                    'ground_truth': truth,
                    'prediction': pred,
                    'passages': passages.get(fact, [])
                })
    
    errors = {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2)
    
    print(f"Generated errors file at {output_path}")
    print(f"Found {len(false_positives)} false positives and {len(false_negatives)} false negatives")


def analyze_errors(errors_file, output_categories=None, interactive=False):
    """Analyze errors and categorize them.
    
    Args:
        errors_file: Path to the errors.json file
        output_categories: Optional path to save categorized errors
        interactive: Whether to use interactive categorization
    """
    errors = load_errors(errors_file)
    
    # Print basic statistics
    fp_count = len(errors['false_positives'])
    fn_count = len(errors['false_negatives'])
    total = fp_count + fn_count
    
    print(f"\nError Analysis Summary:")
    print(f"Total errors: {total}")
    
    # Add check to avoid division by zero
    if total > 0:
        print(f"False positives: {fp_count} ({fp_count/total*100:.1f}%)")
        print(f"False negatives: {fn_count} ({fn_count/total*100:.1f}%)")
    else:
        print(f"False positives: {fp_count} (0.0%)")
        print(f"False negatives: {fn_count} (0.0%)")
    
    # Example error categories (you can modify these based on your analysis)
    error_categories = {
        'semantic_confusion': {
            'description': 'Model fails to distinguish between related but distinct concepts',
            'fp_examples': [],
            'fn_examples': []
        },
        'implicit_information': {
            'description': 'Model fails to infer information not explicitly stated in the text',
            'fp_examples': [],
            'fn_examples': []
        },
        'temporal_mismatch': {
            'description': 'Model fails to correctly interpret temporal information',
            'fp_examples': [],
            'fn_examples': []
        },
        'reference_resolution': {
            'description': 'Model fails to correctly resolve references (pronouns, entities, etc.)',
            'fp_examples': [],
            'fn_examples': []
        }
    }
    
    # Categorize false positives and false negatives
    if fp_count > 0:
        categorize_examples(errors['false_positives'], error_categories, 'fp_examples', interactive)
    
    if fn_count > 0:
        categorize_examples(errors['false_negatives'], error_categories, 'fn_examples', interactive)
    
    # Save categorized errors if output path is provided
    if output_categories:
        with open(output_categories, 'w', encoding='utf-8') as f:
            json.dump(error_categories, f, indent=2)
        print(f"\nSaved categorized errors to {output_categories}")
    
    return error_categories


def categorize_examples(examples, error_categories, example_key, interactive=True):
    """Categorize examples into error categories.
    
    Args:
        examples: List of examples to categorize
        error_categories: Dictionary of error categories
        example_key: Key to store examples in error_categories ('fp_examples' or 'fn_examples')
        interactive: Whether to interactively categorize examples or use automatic categorization
    """
    if not examples:
        print(f"No examples to categorize for {example_key}")
        return
    
    # Define category names for easier reference
    categories = list(error_categories.keys())
    
    if interactive:
        print(f"\nCategorizing {len(examples)} examples for {example_key}")
        print("Available categories:")
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat}: {error_categories[cat]['description']}")
        
        for i, example in enumerate(examples):
            print(f"\nExample {i+1}/{len(examples)}:")
            print(f"Fact: {example['fact']}")
            print(f"Ground truth: {example['ground_truth']}")
            print(f"Prediction: {example['prediction']}")
            print("Passages:")
            for j, passage in enumerate(example['passages']):
                print(f"  Passage {j+1}: {passage['text'][:100]}...")
            
            cat_input = input("\nEnter category number (or 'skip' to skip): ")
            if cat_input.lower() == 'skip':
                continue
            
            try:
                cat_idx = int(cat_input) - 1
                if 0 <= cat_idx < len(categories):
                    error_categories[categories[cat_idx]][example_key].append(example)
                    print(f"Added to category: {categories[cat_idx]}")
                else:
                    print("Invalid category number, skipping")
            except ValueError:
                print("Invalid input, skipping")
    else:
        # Automatic categorization based on simple heuristics
        print(f"\nAutomatically categorizing {len(examples)} examples for {example_key}")
        
        for example in examples:
            fact = example['fact'].lower()
            passages_text = ' '.join([p['text'].lower() for p in example['passages']])
            
            # Simple heuristic categorization
            if len(fact.split()) < 5:  # Very short facts often have reference resolution issues
                error_categories['reference_resolution'][example_key].append(example)
            elif any(word in fact for word in ['before', 'after', 'during', 'when', 'while', 'year', 'month', 'day']):
                error_categories['temporal_mismatch'][example_key].append(example)
            elif len(set(fact.split()) & set(passages_text.split())) < len(fact.split()) * 0.5:
                error_categories['implicit_information'][example_key].append(example)
            else:
                error_categories['semantic_confusion'][example_key].append(example)
        
        # Print counts
        for cat in categories:
            count = len(error_categories[cat][example_key])
            print(f"  {cat}: {count} examples")


def generate_error_statistics(categorized_errors):
    """Generate statistics from categorized errors."""
    stats = {}
    
    for category, info in categorized_errors.items():
        fp_count = len(info['fp_examples'])
        fn_count = len(info['fn_examples'])
        total = fp_count + fn_count
        
        stats[category] = {
            'total': total,
            'false_positives': fp_count,
            'false_negatives': fn_count
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze fact-checking model errors')
    parser.add_argument('--errors_file', type=str, default='errors.json',
                        help='Path to the errors.json file')
    parser.add_argument('--categorize', action='store_true',
                        help='Categorize errors and generate statistics')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save categorized errors')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive mode for categorization')
    
    args = parser.parse_args()
    
    # Check if errors file exists
    if not os.path.exists(args.errors_file):
        print(f"Error: File {args.errors_file} not found")
        return
    
    # Analyze errors
    if args.categorize:
        error_categories = analyze_errors(args.errors_file, args.output, args.interactive)
        
        # Generate and print statistics
        stats = generate_error_statistics(error_categories)
        print("\nError Category Statistics:")
        print("-" * 50)
        print(f"{'Category':<25} {'Total':<10} {'FP':<10} {'FN':<10}")
        print("-" * 50)
        for cat, counts in stats.items():
            print(f"{cat:<25} {counts['total']:<10} {counts['false_positives']:<10} {counts['false_negatives']:<10}")
    else:
        analyze_errors(args.errors_file)


if __name__ == "__main__":
    main()