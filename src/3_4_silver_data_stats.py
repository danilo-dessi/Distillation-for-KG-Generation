import os
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ORIGINAL_DATASET_FILE = "./data/balanced_papers_2025.json"

# Target Relation Extraction (RE) output files (excluding Llama)
FILES = {
    "DeepSeek": "./data/deepseek_RE_annotated_balanced_papers_2025.json",
    "Gemini": "./data/gemini_RE_annotated_balanced_papers_2025.json",
    "GPT-4o-mini": "./data/gpt40mini_RE_annotated_balanced_papers_2025.json"
}

def load_original_dataset_stats(filepath):
    """Loads the original dataset to calculate total and unique paper counts."""
    if not os.path.exists(filepath):
        print(f"⚠️ Original dataset file not found: {filepath}")
        return None, None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    total_raw = len(original_data)
    unique_abstracts = len({p.get('abstract', '').strip() for p in original_data.values()})
    
    return total_raw, unique_abstracts

def parse_model_outputs():
    # 1. Inspect original dataset
    orig_total, orig_unique = load_original_dataset_stats(ORIGINAL_DATASET_FILE)
    
    print("=" * 70)
    print("📊 ORIGINAL DATASET SUMMARY")
    print("=" * 70)
    if orig_total is not None:
        print(f"• Total entries in input file:  {orig_total}")
        print(f"• Unique abstracts in input:    {orig_unique}")
        print(f"• Inherent duplicates in input: {orig_total - orig_unique}")
    else:
        print("• Original dataset file could not be loaded.")
    print("=" * 70)

    # 2. Parse model outputs
    results = []
    
    for model_name, file_path in FILES.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found for {model_name}: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        seen_abstracts = set()
        unique_abstracts = 0
        num_entities = 0
        num_relationships = 0
        total_records = len(data)
        
        for pid, paper in data.items():
            abstract_text = paper.get('abstract', '').strip()
            
            # Filter duplicate abstracts
            if abstract_text in seen_abstracts:
                continue
                
            seen_abstracts.add(abstract_text)
            unique_abstracts += 1
            num_entities += len(paper.get('entities', []))
            num_relationships += len(paper.get('relationships', []))
            
        # Calculate coverage against original unique count if available
        coverage_pct = f"{(unique_abstracts / orig_unique * 100):.1f}%" if orig_unique else "N/A"
        
        results.append({
            "Model": model_name,
            "Abstracts": unique_abstracts,
            "Entities": num_entities,
            "Relationships": num_relationships,
            "Duplicates": total_records - unique_abstracts,
            "Coverage": coverage_pct
        })
        
    # ==============================================================================
    # PRINT RESULTS TABLE
    # ==============================================================================
    if not results:
        print("No model output files were found to process.")
        return

    print("\n📈 MODEL EXTRACTION METRICS (Deduplicated)")
    print(f"{'Model':<15} | {'Abstracts':<10} | {'Entities':<10} | {'Relationships':<13} | {'Duplicates':<10} | {'Coverage':<8}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['Model']:<15} | {res['Abstracts']:<10} | {res['Entities']:<10} | {res['Relationships']:<13} | {res['Duplicates']:<10} | {res['Coverage']:<8}")

if __name__ == "__main__":
    parse_model_outputs()
