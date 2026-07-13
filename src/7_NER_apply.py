# -*- coding: utf-8 -*-
# 7_NER_apply.py

import os
import json
import torch
import math
import re
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./data/papers_2026_01.json",
    "OUTPUT_FILE": "./data/ner_applied_papers_2026_01.json",
    "MODEL_DIR": "./models/ner-gemini-Scideberta-full-finetuned-mlm",
    "GPU_BATCH_SIZE": 4,   
    "DATA_CHUNK_SIZE": 100, 
    "CONFIDENCE_THRESHOLD": 0.70
}

# ==============================================================================
# POST-PROCESSING MODULES
# ==============================================================================

def build_acronym_map(text):
    """
    Acronym Resolution: Scans the abstract for 'Full Name (FN)' patterns.
    Creates a mapping to standardize entities within the same document.
    """
    # Matches patterns like "Large Language Model (LLM)" or "State-of-the-art (SOTA)"
    pattern = r'([A-Z][a-zA-Z0-9\-\s]+?)\s+\(([A-Za-z0-9]{2,})\)'
    matches = re.finditer(pattern, text)
    
    acronym_map = {}
    for match in matches:
        full_form = match.group(1).strip()
        acronym = match.group(2).strip()
        standardized_name = f"{full_form} ({acronym})"
        
        # Map both the abbreviation and the raw full form to the standardized version
        acronym_map[acronym] = standardized_name
        acronym_map[full_form] = standardized_name
        
    return acronym_map

def expand_boundaries(text, start, end):
    """
    Boundary Expansion: Recovers truncated prefixes/suffixes 
    by walking outward to the nearest spaces.
    """
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ['-', '_']):
        start -= 1
        
    while end < len(text) and (text[end].isalnum() or text[end] in ['-', '_']):
        end += 1
        
    return start, end, text[start:end].strip()

def resolve_overlaps_and_deduplicate(entities):
    """
    De-duplication: Resolves overlapping character spans by prioritizing 
    the highest confidence score. Drops exact duplicate names.
    """
    # 1. Sort by confidence score (highest first)
    sorted_entities = sorted(entities, key=lambda x: x['score'], reverse=True)
    resolved_spans = []
    
    for ent in sorted_entities:
        overlap = False
        for keep_ent in resolved_spans:
            # Check for character boundary overlap
            if ent['start'] < keep_ent['end'] and ent['end'] > keep_ent['start']:
                overlap = True
                break
                
        if not overlap:
            resolved_spans.append(ent)
            
    # 2. Extract final unique entities by standardized name
    final_entities = []
    seen_names = set()
    
    for ent in resolved_spans:
        if ent['name'] not in seen_names:
            final_entities.append({
                "name": ent['name'],
                "type": ent['type']
            })
            seen_names.add(ent['name'])
            
    return final_entities

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print(f"--> Loading NER model from {CONFIG['MODEL_DIR']}")
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_DIR"])
    model = AutoModelForTokenClassification.from_pretrained(CONFIG["MODEL_DIR"])
    
    ner_pipeline = pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        device=device,
        aggregation_strategy="simple"
    )

    with open(CONFIG["INPUT_FILE"], 'r', encoding='utf-8') as f:
        papers = json.load(f)

    output_data = {}
    if os.path.exists(CONFIG["OUTPUT_FILE"]):
        try:
            with open(CONFIG["OUTPUT_FILE"], 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except json.JSONDecodeError:
            output_data = {}
            
    papers_to_process = {pid: p for pid, p in papers.items() if pid not in output_data}
    paper_ids = list(papers_to_process.keys())
    
    if not paper_ids:
        print("✅ No new papers to process. Exiting.")
        return

    total_papers = len(paper_ids)
    total_chunks = math.ceil(total_papers / CONFIG["DATA_CHUNK_SIZE"])
    
    print(f"--> Processing {total_papers} papers in {total_chunks} chunks...")

    for chunk_idx in tqdm(range(total_chunks), desc="Processing Chunks"):
        start_idx = chunk_idx * CONFIG["DATA_CHUNK_SIZE"]
        end_idx = start_idx + CONFIG["DATA_CHUNK_SIZE"]
        chunk_ids = paper_ids[start_idx:end_idx]
        
        abstracts = []
        
        for pid in chunk_ids:
            raw_text = str(papers_to_process[pid].get('abstract', ''))
            # Force encode/decode to ensure string indices perfectly match the model's token limits
            tokens = tokenizer.encode(raw_text, truncation=True, max_length=510, add_special_tokens=False)
            safe_text = tokenizer.decode(tokens)
            abstracts.append(safe_text)
            
        results = ner_pipeline(abstracts, batch_size=CONFIG["GPU_BATCH_SIZE"])
        
        for i, out in enumerate(results):
            pid = chunk_ids[i]
            original_text = abstracts[i]
            
            # Phase 1: Build local acronym dictionary for this specific abstract
            acronym_map = build_acronym_map(original_text)
            
            candidate_entities = []
            
            for ent in out:
                if ent['score'] >= CONFIG["CONFIDENCE_THRESHOLD"]:
                    
                    # Phase 2: Boundary Expansion
                    new_start, new_end, expanded_name = expand_boundaries(
                        original_text, ent['start'], ent['end']
                    )
                    
                    if len(expanded_name) == 0:
                        continue
                        
                    # Phase 3: Acronym Resolution (Map to standard form if it exists)
                    standardized_name = acronym_map.get(expanded_name, expanded_name)
                    
                    candidate_entities.append({
                        "start": new_start,
                        "end": new_end,
                        "name": standardized_name,
                        "score": ent['score'],
                        "type": ent['entity_group']
                    })
            
            # Phase 4: Overlap De-duplication
            final_clean_entities = resolve_overlaps_and_deduplicate(candidate_entities)
            
            paper_data = papers_to_process[pid]
            paper_data["entities"] = final_clean_entities
            output_data[pid] = paper_data

        # Save checkpoint after every chunk
        with open(CONFIG["OUTPUT_FILE"], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ NER Extraction complete. Saved to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    main()