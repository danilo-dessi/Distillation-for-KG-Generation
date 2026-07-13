# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Swap out as needed: "KISTI-AI/Scideberta-full", "allenai/scibert_scivocab_uncased", "microsoft/deberta-v3-small"
    "BASE_MODEL": "models/cs_roberta_base-finetuned-mlm", 
    "INPUT_DATA_FILE": "./data/gpt40mini-annotated_balanced_papers_2025.json",
    "OUTPUT_DIR": "./models/ner-gpt40mini-cs_roberta_base-finetuned-mlm",
    
    "EPOCHS": 50,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 3e-5,
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

# Mapping utilities
LABEL_TO_ID = {label: i for i, label in enumerate(CONFIG["LABELS"])}
ID_TO_LABEL = {i: label for i, label in enumerate(CONFIG["LABELS"])}

# ==============================================================================
# DATASET PREPARATION & ROBUST ALIGNMENT
# ==============================================================================
def align_labels_with_tokens(tokenizer, text, entities):
    """
    Uses character offsets to precisely align LLM-extracted text spans 
    to subword tokens, completely bypassing punctuation splitting bugs.
    """
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

    # 1. Match occurrences of the entity using exact character bounds
    for entity in entities:
        ent_text = entity['name']
        ent_type = entity['type']

        if f"B-{ent_type}" not in LABEL_TO_ID: continue

        for match in re.finditer(re.escape(ent_text), text):
            start_char, end_char = match.span()
            found_start = False
            
            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end == 0: continue 
                
                # If the token falls within the character bounds of the entity
                if token_start >= start_char and token_end <= end_char:
                    if not found_start:
                        labels[idx] = LABEL_TO_ID[f"B-{ent_type}"]
                        found_start = True
                    else:
                        labels[idx] = LABEL_TO_ID[f"I-{ent_type}"]

    # 2. Mask special tokens and continuation subwords with -100
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            labels[idx] = -100
        elif word_idx != previous_word_idx:
            pass # Keep the assigned label for the start of a word
        else:
            labels[idx] = -100 # Mask subword continuations
        previous_word_idx = word_idx
                
    return tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], labels


def load_and_tokenize_dataset(filepath, tokenizer):
    print(f"--> Loading and aligning data from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    hf_data = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for pid, paper in data.items():
        abstract = paper.get("abstract", "")
        entities = paper.get("entities", [])
        
        if not abstract: continue
            
        input_ids, attention_mask, labels = align_labels_with_tokens(tokenizer, abstract, entities)
        
        hf_data["input_ids"].append(input_ids)
        hf_data["attention_mask"].append(attention_mask)
        hf_data["labels"].append(labels)
        
    print(f"✅ Successfully aligned {len(hf_data['input_ids'])} abstracts.")
    return Dataset.from_dict(hf_data)

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
# MAIN TRAINING EXECUTION
# ==============================================================================
def main():
    print(f"--> Initializing Tokenizer & Model: {CONFIG['BASE_MODEL']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_MODEL"], use_fast=True)
    
    # Handle tokenizers missing a padding token definition
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForTokenClassification.from_pretrained(
        CONFIG["BASE_MODEL"], 
        num_labels=len(CONFIG["LABELS"]),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )

    # Load, Tokenize, and Align in one step
    raw_dataset = load_and_tokenize_dataset(CONFIG["INPUT_DATA_FILE"], tokenizer)
    
    # 1. First split: 80% Train+Val, 20% strictly for Final Test
    train_val_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_val_split["test"]
    
    # 2. Second split: Split the 80% to get a standard Train/Val ratio
    # (0.125 of 80% = 10% of total data for validation during epochs)
    train_val = train_val_split["train"].train_test_split(test_size=0.125, seed=42)
    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    
    print(f"--> Dataset Splits Ready:")
    print(f"    Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=CONFIG["OUTPUT_DIR"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=CONFIG["LEARNING_RATE"],
        per_device_train_batch_size=CONFIG["BATCH_SIZE"],
        per_device_eval_batch_size=CONFIG["BATCH_SIZE"],
        num_train_epochs=CONFIG["EPOCHS"],
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("--> Beginning Fine-Tuning Loop...")
    trainer.train()

    print(f"--> Execution complete. Saving optimal weights to {CONFIG['OUTPUT_DIR']}")
    trainer.save_model(CONFIG["OUTPUT_DIR"])
    tokenizer.save_pretrained(CONFIG["OUTPUT_DIR"])

    # Run a final evaluation round to extract exact scores on the BLIND TEST set
    print("--> Running final evaluation generation on blind test set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)

    # Write metrics report to a .txt file alongside the weights
    report_path = os.path.join(CONFIG["OUTPUT_DIR"], "precision_recall_f1_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(f"NER EVALUATION REPORT FOR: {CONFIG['BASE_MODEL']}\n")
        f.write("============================================================\n\n")
        f.write("GLOBAL METRICS:\n")
        f.write(f"  Accuracy:  {eval_results.get('eval_accuracy', 0):.4f}\n")
        f.write(f"  Precision: {eval_results.get('eval_precision', 0):.4f}\n")
        f.write(f"  Recall:    {eval_results.get('eval_recall', 0):.4f}\n")
        f.write(f"  F1-Score:  {eval_results.get('eval_f1', 0):.4f}\n\n")
        f.write("PER-CLASS METRICS:\n")
        f.write(f"{'Entity Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n")
        f.write("-" * 55 + "\n")
        
        unique_classes = set([lbl.split("-")[1] for lbl in CONFIG["LABELS"] if lbl != "O"])
        for cls in sorted(unique_classes):
            p = eval_results.get(f"eval_{cls}_precision", 0.0)
            r = eval_results.get(f"eval_{cls}_recall", 0.0)
            f1 = eval_results.get(f"eval_{cls}_f1", 0.0)
            f.write(f"{cls:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}\n")

    print(f"✅ Training completed successfully. Metrics log generated at {report_path}")

if __name__ == "__main__":
    main()
