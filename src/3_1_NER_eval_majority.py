# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
from collections import Counter
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "MODEL_TO_EVALUATE": "./models/ner-gemini-cs_roberta_base-finetuned-mlm", 
    
    "DATASET_1": "./data/gpt40mini-annotated_balanced_papers_2025.json",
    "DATASET_2": "./data/gemini_NER_annotated_balanced_papers_2025.json",
    "DATASET_3": "./data/deepseek_NER_annotated_balanced_papers_2025.json", 
    
    "MAX_LENGTH": 512,
    
    "LABELS": [
        "O",
        "B-Task", "I-Task",
        "B-Model", "I-Model",
        "B-Algorithm", "I-Algorithm",
        "B-Dataset", "I-Dataset",
        "B-Metric", "I-Metric",
        "B-Hardware", "I-Hardware"
    ]
}

LABEL_TO_ID = {label: i for i, label in enumerate(CONFIG["LABELS"])}
ID_TO_LABEL = {i: label for i, label in enumerate(CONFIG["LABELS"])}

# ==============================================================================
# DATASET PREPARATION & ALIGNMENT
# ==============================================================================
def align_labels_with_tokens(tokenizer, text, entities):
    tokenized_inputs = tokenizer(
        text, 
        truncation=True, 
        max_length=CONFIG['MAX_LENGTH'],
        return_offsets_mapping=True, 
        is_split_into_words=False
    )
    
    offsets = tokenized_inputs.offset_mapping
    word_ids = tokenized_inputs.word_ids() 
    labels = [LABEL_TO_ID["O"]] * len(offsets)

    for entity in entities:
        ent_text = entity['name']
        ent_type = entity['type']

        if f"B-{ent_type}" not in LABEL_TO_ID: continue

        for match in re.finditer(re.escape(ent_text), text):
            start_char, end_char = match.span()
            found_start = False
            
            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end == 0: continue 
                
                if token_start >= start_char and token_end <= end_char:
                    if not found_start:
                        labels[idx] = LABEL_TO_ID[f"B-{ent_type}"]
                        found_start = True
                    else:
                        labels[idx] = LABEL_TO_ID[f"I-{ent_type}"]

    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            labels[idx] = -100
        elif word_idx != previous_word_idx:
            pass 
        else:
            labels[idx] = -100 
        previous_word_idx = word_idx
                
    return tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], labels


def extract_and_align_dataset(filepath, tokenizer, base_keys, master_abstracts):
    """
    Aligns entities against a locked set of master abstracts.
    Guarantees perfect token-length matching even if the JSON is missing papers.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dataset_dict = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for pid in base_keys:
        # Use the master text from Dataset 1 to ensure identical tokenization
        abstract = master_abstracts[pid] 
        
        # If this specific model missed the paper, it defaults to an empty entity list []
        entities = data.get(pid, {}).get("entities", [])
        
        input_ids, attention_mask, labels = align_labels_with_tokens(tokenizer, abstract, entities)
        
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["labels"].append(labels)
        
    return Dataset.from_dict(dataset_dict)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [CONFIG["LABELS"][p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [CONFIG["LABELS"][l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    
    flattened_metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
    for k, v in results.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flattened_metrics[f"{k}_{sub_k}"] = sub_v
                
    return flattened_metrics

# ==============================================================================
# MAIN EVALUATION EXECUTION
# ==============================================================================
def main():
    print(f"--> Initializing Tokenizer & Model from: {CONFIG['MODEL_TO_EVALUATE']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_TO_EVALUATE"], use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForTokenClassification.from_pretrained(
        CONFIG["MODEL_TO_EVALUATE"], 
        num_labels=len(CONFIG["LABELS"]),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )

    # 1. Establish the Single Source of Truth for IDs and text
    with open(CONFIG["DATASET_1"], "r", encoding="utf-8") as f:
        base_data = json.load(f)
        
    # Lock in the exact paper IDs and order
    base_keys = [pid for pid, paper in base_data.items() if paper.get("abstract", "")]
    
    # Lock in the master abstract texts
    master_abstracts = {pid: base_data[pid]["abstract"] for pid in base_keys}

    # 2. Extract and align each dataset independently using the master texts
    print("--> Loading, tokenizing, and aligning datasets independently...")
    ds_1 = extract_and_align_dataset(CONFIG["DATASET_1"], tokenizer, base_keys, master_abstracts)
    ds_2 = extract_and_align_dataset(CONFIG["DATASET_2"], tokenizer, base_keys, master_abstracts)
    ds_3 = extract_and_align_dataset(CONFIG["DATASET_3"], tokenizer, base_keys, master_abstracts)
    
    # 3. Slicing datasets individually to isolate the exact 20% test partitions
    print("--> Slicing datasets to isolate the 20% evaluation splits...")
    test_1 = ds_1.train_test_split(test_size=0.2, seed=42)["test"]
    test_2 = ds_2.train_test_split(test_size=0.2, seed=42)["test"]
    test_3 = ds_3.train_test_split(test_size=0.2, seed=42)["test"]
    
    # 4. Merge the isolated test splits via majority voting
    print("--> Merging test splits via token-level majority voting...")
    voted_labels_list = []
    
    for labels_1, labels_2, labels_3 in zip(test_1["labels"], test_2["labels"], test_3["labels"]):
        voted_labels = []
        for l1, l2, l3 in zip(labels_1, labels_2, labels_3):
            if l1 == -100:  
                voted_labels.append(-100)
                continue
                
            counts = Counter([l1, l2, l3])
            most_common = counts.most_common()
            
            if most_common[0][1] >= 2:
                # A majority consensus is reached
                voted_labels.append(most_common[0][0])
            else:
                # No consensus reached. Discard the annotation by marking it as 'O' (Outside).
                voted_labels.append(LABEL_TO_ID["O"])
                
        voted_labels_list.append(voted_labels)
        
    unified_test_dataset = Dataset.from_dict({
        "input_ids": test_1["input_ids"],
        "attention_mask": test_1["attention_mask"],
        "labels": voted_labels_list
    })
    print(f"Unified test set finalized with {len(unified_test_dataset)} abstracts.")

    # 5. Evaluate the model
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    evaluator = Trainer(
        model=model,
        eval_dataset=unified_test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\n--> Running evaluation pass on the unified blind test set...")
    eval_results = evaluator.evaluate()

    model_name = os.path.basename(os.path.normpath(CONFIG["MODEL_TO_EVALUATE"]))
    
    print("\n" + "=" * 60)
    print(f"UNIFIED NER EVALUATION REPORT FOR: {model_name}")
    print("=" * 60 + "\n")
    print("GLOBAL METRICS:")
    print(f"  Accuracy:  {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"  Recall:    {eval_results.get('eval_recall', 0):.4f}")
    print(f"  F1-Score:  {eval_results.get('eval_f1', 0):.4f}\n")
    print("PER-CLASS METRICS:")
    print(f"{'Entity Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    
    unique_classes = set([lbl.split("-")[1] for lbl in CONFIG["LABELS"] if lbl != "O"])
    for cls in sorted(unique_classes):
        p = eval_results.get(f"eval_{cls}_precision", 0.0)
        r = eval_results.get(f"eval_{cls}_recall", 0.0)
        f1 = eval_results.get(f"eval_{cls}_f1", 0.0)
        print(f"{cls:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    
    print("-" * 55)
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()