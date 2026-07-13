# -*- coding: utf-8 -*-
import os
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./eval/pipeline_eval.json"
}

def compute_statistics():
    if not os.path.exists(CONFIG["INPUT_FILE"]):
        print(f"❌ Error: Could not find {CONFIG['INPUT_FILE']}")
        return

    print(f"--> Loading evaluation dataset: {CONFIG['INPUT_FILE']}")
    df = pd.read_json(CONFIG["INPUT_FILE"])
    
    if df.empty:
        print("⚠️ The dataset is empty.")
        return

    # --- 1. Overall Statistics ---
    print("\n" + "="*40)
    print(" 📊 OVERALL DATASET STATISTICS")
    print("="*40)
    print(f"Total Triples Extracted : {len(df)}")
    print(f"Total Unique Papers     : {df['paper_id'].nunique()}")

    # --- 2. Relationship (Predicate) Distribution ---
    print("\n" + "="*40)
    print(" 🔗 RELATIONSHIP DISTRIBUTION")
    print("="*40)
    rel_counts = df['predicate'].value_counts()
    rel_percentages = (df['predicate'].value_counts(normalize=True) * 100).round(2)
    
    rel_df = pd.DataFrame({'Count': rel_counts, 'Percentage (%)': rel_percentages})
    print(rel_df.to_string())

    # --- 3. Entity Category Distribution ---
    print("\n" + "="*40)
    print(" 🏷️ ENTITY CATEGORY DISTRIBUTION")
    print("="*40)
    
    # Isolate subjects and objects, then stack them to analyze all entities together
    subjects = df[['subject', 'subject_type']].rename(columns={'subject': 'entity', 'subject_type': 'type'})
    objects = df[['object', 'object_type']].rename(columns={'object': 'entity', 'object_type': 'type'})
    all_entities = pd.concat([subjects, objects])

    # A) By Total Occurrences (How often a category appears as a subject/object)
    print("\n--- By Total Occurrences in Triples ---")
    occ_counts = all_entities['type'].value_counts()
    occ_percentages = (all_entities['type'].value_counts(normalize=True) * 100).round(2)
    occ_df = pd.DataFrame({'Occurrences': occ_counts, 'Percentage (%)': occ_percentages})
    print(occ_df.to_string())

    # B) By Unique Entities (How many distinct terms belong to each category)
    print("\n--- By Unique Entities ---")
    unique_entities = all_entities.drop_duplicates(subset=['entity', 'type'])
    unique_counts = unique_entities['type'].value_counts()
    unique_percentages = (unique_entities['type'].value_counts(normalize=True) * 100).round(2)
    unique_df = pd.DataFrame({'Unique Count': unique_counts, 'Percentage (%)': unique_percentages})
    print(unique_df.to_string())
    print("\n✅ Statistics computed successfully.\n")

if __name__ == "__main__":
    compute_statistics()
