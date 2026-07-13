# -*- coding: utf-8 -*-
# 8_RE_apply.py

import os
import json
import torch
import math
from itertools import permutations
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./data/ner_applied_papers_2026_01.json",
    "OUTPUT_FILE": "./data/re_applied_papers_2026_01.json",
    # Point this to your best trained RE model directory
    "MODEL_DIR": "./models/re-gemini-Scideberta-full-finetuned-mlm",
    "GPU_BATCH_SIZE": 16,     # HF pipeline micro-batch size (Keep lower for RE pair sequences)
    "DATA_CHUNK_SIZE": 100,   # How many abstracts we process before saving a checkpoint
    "CONFIDENCE_THRESHOLD": 0.70
}

def main():
    print(f"--> Loading RE model from {CONFIG['MODEL_DIR']}")
    
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_DIR"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["MODEL_DIR"])
    
    # Modern HuggingFace pipeline for sequence classification
    # Enforcing truncation directly in the pipeline prevents Token Length OOM errors.
    re_pipeline = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device,
        truncation=True,
        max_length=512
    )
    
    sep_token = tokenizer.sep_token if tokenizer.sep_token else " "

    with open(CONFIG["INPUT_FILE"], 'r', encoding='utf-8') as f:
        papers = json.load(f)

    output_data = {}
    if os.path.exists(CONFIG["OUTPUT_FILE"]):
        try:
            with open(CONFIG["OUTPUT_FILE"], 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: Output file exists but is corrupted/empty. Starting fresh.")
            output_data = {}
            
    papers_to_process = {pid: p for pid, p in papers.items() if pid not in output_data}
    paper_ids = list(papers_to_process.keys())
    
    if not paper_ids:
        print("✅ No new papers to process. Exiting.")
        return

    total_papers = len(paper_ids)
    total_chunks = math.ceil(total_papers / CONFIG["DATA_CHUNK_SIZE"])
    
    print(f"--> Processing {total_papers} papers in {total_chunks} chunks of {CONFIG['DATA_CHUNK_SIZE']}...")

    # Process the dataset in manageable outer chunks
    for chunk_idx in tqdm(range(total_chunks), desc="Overall Progress (Chunks)"):
        start_idx = chunk_idx * CONFIG["DATA_CHUNK_SIZE"]
        end_idx = start_idx + CONFIG["DATA_CHUNK_SIZE"]
        chunk_ids = paper_ids[start_idx:end_idx]
        
        flat_inputs = []
        flat_metadata = [] 
        
        # 1. Format pairs specifically for this chunk
        for pid in chunk_ids:
            paper = papers_to_process[pid]
            abstract = str(paper.get("abstract", "")).strip()
            entities = paper.get("entities", [])
            
            # Initialize empty relationships list for this paper
            paper["relationships"] = []
            output_data[pid] = paper
            
            # We need at least 2 entities to form a relationship
            if not abstract or len(entities) < 2:
                continue
                
            entity_names = list(set([e['name'] for e in entities if 'name' in e]))
            all_possible_pairs = list(permutations(entity_names, 2))
            
            for sub, obj in all_possible_pairs:
                text_input = f"{sub} {sep_token} {obj}"
                # Standard HF dict formatting for passing text pairs
                flat_inputs.append({"text": text_input, "text_pair": abstract})
                flat_metadata.append({"pid": pid, "subject": sub, "object": obj})
                
        # If no valid pairs exist in this chunk, just save and move on
        if not flat_inputs:
            with open(CONFIG["OUTPUT_FILE"], 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            continue

        # 2. Run inference on the permutations
        results = re_pipeline(flat_inputs, batch_size=CONFIG["GPU_BATCH_SIZE"])

        # 3. Map predictions back to the papers
        for i, out in enumerate(results):
            # Account for varying pipeline return structures (dict vs list of dicts)
            prediction = out[0] if isinstance(out, list) else out
            label = prediction['label']
            score = prediction['score']
            
            # Discard negative samples and low confidence predictions
            if label != "NO_RELATION" and score >= CONFIG["CONFIDENCE_THRESHOLD"]:
                meta = flat_metadata[i]
                pid = meta["pid"]
                output_data[pid]["relationships"].append({
                    "subject": meta["subject"],
                    "relationship": label,
                    "object": meta["object"],
                    "confidence": round(score, 4)
                })

        # 4. Save checkpoint after every chunk
        with open(CONFIG["OUTPUT_FILE"], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ RE Extraction complete. Saved to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    main()