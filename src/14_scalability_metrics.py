import os
import json
import pandas as pd
import tiktoken
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Change this target file to evaluate different model outputs
TARGET_TRIPLES_FILE = "./eval/gemini_zero_shot.json" 
INPUT_ABSTRACTS_FILE = "./eval/pipeline_eval.json"
ONTOLOGY_FILE = "../resources/relationships.json"

# Set to "v1" (baseline) or "v2" (strict ontology) to use the correct prompt length
PROMPT_VERSION = "v1" 
# ==========================================

def load_ontology_prompt():
    """Loads the JSON ontology to reconstruct the v2 prompt exactly."""
    if not os.path.exists(ONTOLOGY_FILE):
        print(f"Warning: Ontology not found at {ONTOLOGY_FILE}. Using placeholder length.")
        return "[ONTOLOGY PLACEHOLDER]"
        
    with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
        
    relationships = ontology_data.get("relationships", {})
    prompt_lines = ["\nALLOWED PREDICATES AND THEIR DEFINITIONS:"]
    for predicate, details in relationships.items():
        definition = details.get("definition", "")
        prompt_lines.append(f"- {predicate}: {definition}")
        
    return "\n".join(prompt_lines)

def get_base_prompt(version="v2"):
    """Reconstructs the static portion of the prompts."""
    if version == "v2":
        ontology = load_ontology_prompt()
        return f"""You are an expert AI researcher performing Joint Entity and Relation Extraction.
Extract valid knowledge triples from this abstract using standard scientific entities (Task, Model, Algorithm, Dataset, Metric, Hardware).

CRITICAL INSTRUCTION: You must strictly use ONLY the predicates listed below. Do not invent, hallucinate, or modify these predicates in any way.
{ontology}

Abstract:
"""
    else: # v1 baseline
        return """You are an expert AI researcher performing Joint Entity and Relation Extraction.
Extract valid knowledge triples from this abstract using standard scientific entities (Task, Model, Algorithm, Dataset, Metric, Hardware) and predicates (USES, PRODUCES, ANALYZES, INCLUDES, IS_A, COMPARES_TO, INFLUENCES, SHOWS, IDENTIFIES, ADDRESSES, EVALUATED_ON, ACHIEVES).

Abstract:
"""

def main():
    print(f"--- Analyzing Tokens for: {TARGET_TRIPLES_FILE} ---")
    
    # Initialize the tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    
    # 1. Calculate Average Input Tokens
    df_pipeline = pd.read_json(INPUT_ABSTRACTS_FILE)
    abstracts = df_pipeline['abstract_text'].drop_duplicates().tolist()
    
    base_prompt = get_base_prompt(PROMPT_VERSION)
    base_prompt_tokens = len(encoding.encode(base_prompt))
    
    input_token_counts = []
    for abstract in abstracts:
        input_token_counts.append(base_prompt_tokens + len(encoding.encode(abstract)))
        
    avg_input_tokens = np.mean(input_token_counts)
    
    # 2. Calculate Average Output Tokens
    with open(TARGET_TRIPLES_FILE, 'r', encoding='utf-8') as f:
        extracted_triples = json.load(f)
        
    # Group triples by paper_id to see how many output tokens were generated per abstract
    df_triples = pd.DataFrame(extracted_triples)
    
    # Group by paper_id, convert the group back to a JSON string, and count the tokens
    output_token_counts = []
    for paper_id, group in df_triples.groupby('paper_id'):
        # Exclude the paper_id and abstract_text to match the strict schema format
        group_clean = group[['subject', 'subject_type', 'predicate', 'object', 'object_type']].to_dict('records')
        output_json_str = json.dumps({"triples": group_clean})
        output_token_counts.append(len(encoding.encode(output_json_str)))
        
    # Account for papers that generated 0 triples (if they were in the input but not the output)
    missing_papers = len(abstracts) - len(df_triples['paper_id'].unique())
    output_token_counts.extend([0] * missing_papers)
    
    avg_output_tokens = np.mean(output_token_counts)
    
    print("\n[RESULTS]")
    print(f"Average Input Tokens per Abstract:  {avg_input_tokens:.0f}")
    print(f"Average Output Tokens per Abstract: {avg_output_tokens:.0f}")
    print(f"Total Papers Processed:             {len(abstracts)}")
    print("-" * 50)

if __name__ == "__main__":
    main()