# -*- coding: utf-8 -*-

import os
import json
import torch
import random
import numpy as np
from itertools import permutations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import datasets
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback
)

# ==============================================================================
# MULTI-GPU SAFETY SETUP
# ==============================================================================
os.environ["WANDB_DISABLED"] = "true"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

set_seed(42)
random.seed(42)
np.random.seed(42)

def main_print(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Swap out as needed: "KISTI-AI/Scideberta-full", "allenai/scibert_scivocab_uncased", "microsoft/deberta-v3-small"
    "BASE_MODEL": "models/Scideberta-full-finetuned-mlm", 
    "INPUT_DATA_FILE": "./data/gemini_RE_annotated_balanced_papers_2025.json", 
    "ONTOLOGY_FILE": "../resources/relationships.json",
    "OUTPUT_DIR": "./models/re-gemini-Scideberta-full-finetuned-mlm",

    # Training Hyperparameters
    "MAX_LEN": 512,
    "BATCH_SIZE": 16, 
    "EPOCHS": 50,           
    "LEARNING_RATE": 2e-5,
    "NEGATIVE_RATIO": 5.0, 
    "EVAL_STEPS": 500,     
}

# ==============================================================================
# ONTOLOGY & DATASET PREPARATION
# ==============================================================================
def load_ontology(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ontology file not found at {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = list(data.get('relationships', {}).keys())

    if "NO_RELATION" not in labels:
        labels.append("NO_RELATION")

    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    return labels, id2label, label2id

def prepare_dataset(data_file, label2id, negative_ratio):
    samples = []
    main_print(f"--> Compiling training dataset from {data_file}...")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input data file not found at {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    for paper_id, paper in papers.items():
        if 'relationships' not in paper:
            continue

        abstract = paper.get('abstract')
        entities = paper.get('entities', [])
        relationships = paper['relationships'] 

        if not abstract or len(entities) < 2:
            continue

        entity_names = list(set([e['name'] for e in entities if 'name' in e])) 

        existing_pairs = {}
        for rel in relationships:
            sub = rel.get('subject')
            obj = rel.get('object')
            lbl = rel.get('relationship')

            if lbl in label2id and sub and obj:
                existing_pairs[(sub, obj)] = lbl
                samples.append({
                    "text": abstract,
                    "subject": sub,
                    "object": obj,
                    "label": label2id[lbl]
                })

        all_possible_pairs = list(permutations(entity_names, 2))
        negative_pairs = [p for p in all_possible_pairs if p not in existing_pairs]

        num_positives = len(existing_pairs)
        num_negatives_to_keep = int(num_positives * negative_ratio)

        if num_negatives_to_keep == 0 and len(negative_pairs) > 0:
            num_negatives_to_keep = 1

        selected_negatives = random.sample(negative_pairs, min(len(negative_pairs), num_negatives_to_keep))

        for sub, obj in selected_negatives:
            samples.append({
                "text": abstract,
                "subject": sub,
                "object": obj,
                "label": label2id["NO_RELATION"]
            })

    main_print(f"✅ Generated {len(samples)} total relation pair samples.")
    return samples

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    LABELS, id2label, label2id = load_ontology(CONFIG["ONTOLOGY_FILE"])
    main_print(f"--> Ontology loaded. Classes: {LABELS}")

    raw_data = prepare_dataset(CONFIG["INPUT_DATA_FILE"], label2id, CONFIG["NEGATIVE_RATIO"])
    if not raw_data:
        main_print("❌ Error: No valid data generated.")
        return

    hf_dataset = Dataset.from_list(raw_data)
    
    # ---------------------------------------------------------
    # NEW SPLIT LOGIC: 70% Train, 10% Validation, 20% Test
    # ---------------------------------------------------------
    # 1. Separate the 20% Test set from the rest of the pool
    train_val_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_val_split["test"]
    
    # 2. Extract 10% Validation from the remaining 80% pool
    # (0.125 of 80% = exactly 10% of the total original dataset)
    train_val = train_val_split["train"].train_test_split(test_size=0.125, seed=42)
    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    
    # 3. Group into a single dictionary for easier mapping
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })
    
    main_print(f"--> Dataset Splits Ready:")
    main_print(f"    Train: {len(dataset_dict['train'])} | Val: {len(dataset_dict['val'])} | Test: {len(dataset_dict['test'])}")

    main_print(f"\n--> Loading tokenizer from: {CONFIG['BASE_MODEL']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_MODEL"], use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sep_token = tokenizer.sep_token if tokenizer.sep_token else " "

    def preprocess_function(examples):
        inputs = [f"{s} {sep_token} {o}" for s, o in zip(examples["subject"], examples["object"])]
        return tokenizer(
            inputs,                         
            examples["text"],       
            truncation=True,
            max_length=CONFIG["MAX_LEN"],
            padding=False 
        )

    if LOCAL_RANK != 0:
        datasets.disable_progress_bar()

    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, desc="Tokenizing dataset")

    main_print(f"--> Loading sequence classification model: {CONFIG['BASE_MODEL']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["BASE_MODEL"],
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    training_args = TrainingArguments(
        output_dir=CONFIG["OUTPUT_DIR"],
        eval_strategy="steps",
        eval_steps=CONFIG["EVAL_STEPS"],
        save_strategy="steps",
        save_steps=CONFIG["EVAL_STEPS"],
        learning_rate=CONFIG["LEARNING_RATE"],
        per_device_train_batch_size=CONFIG["BATCH_SIZE"],
        per_device_eval_batch_size=CONFIG["BATCH_SIZE"],
        num_train_epochs=CONFIG["EPOCHS"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none", 
        ddp_find_unused_parameters=False 
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        
        # UPDATED: The Trainer now evaluates against the 10% val_dataset
        eval_dataset=tokenized_datasets["val"],
        
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    main_print("\n--> Starting Training...")
    trainer.train()

    main_print(f"\n--> Saving relation classifier to {CONFIG['OUTPUT_DIR']}")
    trainer.save_model(CONFIG["OUTPUT_DIR"])
    if LOCAL_RANK == 0:
        tokenizer.save_pretrained(CONFIG["OUTPUT_DIR"])

    main_print("✅ Training Complete.")

    # UPDATED: The final prediction report is generated strictly on the 20% blind test_dataset
    main_print("\n--> Running detailed evaluation on the blind Test Set...")
    test_results = trainer.predict(tokenized_datasets["test"])
    
    if LOCAL_RANK == 0:
        logits = test_results.predictions
        true_labels = test_results.label_ids
        predictions = np.argmax(logits, axis=1)
        
        report = classification_report(
            true_labels,
            predictions,
            target_names=LABELS,
            digits=4
        )
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT (Per-Class Metrics)")
        print("="*60)
        print(report)
        print("="*60)
        
        with open(os.path.join(CONFIG["OUTPUT_DIR"], "classification_report.txt"), "w", encoding='utf-8') as f:
            f.write(report)

if __name__ == "__main__":
    main()
