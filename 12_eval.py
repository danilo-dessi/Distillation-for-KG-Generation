# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ==============================================================================
# CONFIGURATION & ONTOLOGY
# ==============================================================================
CONFIG = {
    "JUDGE_FILE": "./eval/gemini_judge_results.json",
    "DATA_DIR": "./eval/"
}

# The strictly enforced ontology. Any extracted predicate NOT in this list 
# will be treated as an invalid hallucination (False Positive).
VALID_PREDICATES = {
    "USES", 
    "PRODUCES", 
    "ANALYZES", 
    "INCLUDES", 
    "IS_A", 
    "COMPARES_TO", 
    "INFLUENCES", 
    "SHOWS", 
    "IDENTIFIES", 
    "ADDRESSES", 
    "EVALUATED_ON", 
    "ACHIEVES"
}

def main():
    judge_filename = os.path.basename(CONFIG['JUDGE_FILE'])
    judge_name = os.path.splitext(judge_filename)[0] 

    print(f"\n--- Starting Pipeline Evaluation ---")
    print(f"--> Using Judge File: {CONFIG['JUDGE_FILE']}")

    if not os.path.exists(CONFIG['JUDGE_FILE']):
        print(f"❌ Error: Judge file not found at {CONFIG['JUDGE_FILE']}")
        return

    # 1. Load the Judge's Scores
    judge_scores = {}
    with open(CONFIG["JUDGE_FILE"], 'r', encoding='utf-8') as f:
        for item in json.load(f):
            sig = item.get("signature_id")
            score = item.get("judge_score", 0)
            if sig:
                judge_scores[sig] = score

    # 2. Find all extraction files
    extraction_files = [f for f in glob.glob(f"{CONFIG['DATA_DIR']}*.json") if "judge" not in f]
    print(f"--> Found {len(extraction_files)} extraction datasets to evaluate.")

    source_extractions = defaultdict(lambda: defaultdict(str))
    gold_standard = set()

    # 3. Build the Pooled Gold Standard
    for filepath in extraction_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                pid = item.get('paper_id')
                sub = str(item.get('subject', '')).strip().lower()
                pred = str(item.get('predicate', '')).strip().upper()
                obj = str(item.get('object', '')).strip().lower()
                source = os.path.basename(filepath).replace('.json', '')

                sig = f"{pid}|{sub}|{pred}|{obj}"
                source_extractions[source][sig] = pred

                if judge_scores.get(sig, 0) == 1:
                    if pred in VALID_PREDICATES:
                        gold_standard.add((sig, pred))

    print(f"--> Pooled Gold Standard contains {len(gold_standard)} verified triples across the {len(VALID_PREDICATES)} valid predicates.")

    # 4. Calculate Metrics & Prepare Sets for Overlap Analysis
    results = []
    detailed_results = [] 
    
    # Dictionary to store sets for pairwise overlap analysis
    all_extracted_sets = {}
    
    for source, extractions in source_extractions.items():
        ext_set = {(sig, pred) for sig, pred in extractions.items()}
        
        # Extracted set strictly adhering to valid ontology (for overlap analysis)
        valid_ext_set = {(sig, pred) for sig, pred in extractions.items() if pred in VALID_PREDICATES}
        all_extracted_sets[source] = valid_ext_set
        
        # --- GLOBAL MICRO METRICS (Using strict set theory) ---
        global_tp = len(gold_standard.intersection(ext_set))
        global_fp = len(ext_set - gold_standard)
        global_fn = len(gold_standard - ext_set)

        micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        # --- PER-RELATION MACRO METRICS ---
        macro_p, macro_r, macro_f1 = [], [], []

        for p in VALID_PREDICATES:
            gold_p = {sig for sig, pr in gold_standard if pr == p}
            ext_p = {sig for sig, pr in extractions.items() if pr == p}

            tp = len(gold_p.intersection(ext_p))
            fp = len(ext_p - gold_p)
            fn = len(gold_p - ext_p)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            macro_p.append(precision)
            macro_r.append(recall)
            macro_f1.append(f1)

            detailed_results.append({
                "Source": source,
                "Predicate": p,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Gold_Support": len(gold_p)
            })

        # Calculate hallucination ratio
        total_extracted = len(ext_set)
        num_hallucinations = len([pr for pr in extractions.values() if pr not in VALID_PREDICATES])
        hallucination_rate = (num_hallucinations / total_extracted) if total_extracted > 0 else 0.0

        results.append({
            "Source": source,
            "Macro Precision": np.mean(macro_p),
            "Macro Recall": np.mean(macro_r),
            "Macro F1": np.mean(macro_f1),
            "Micro Precision": micro_p,  
            "Micro Recall": micro_r,     
            "Micro F1": micro_f1,
            "Total Extracted": total_extracted,
            "True Positives": global_tp,
            "Hallucinations (Count)": num_hallucinations,
            "Hallucination Rate": hallucination_rate
        })

    # 5. Build Pairwise Overlap Matrix (Jaccard Similarity)
    sources_list = list(source_extractions.keys())
    overlap_jaccard_df = pd.DataFrame(index=sources_list, columns=sources_list, dtype=object)

    for s1 in sources_list:
        for s2 in sources_list:
            intersection = len(all_extracted_sets[s1].intersection(all_extracted_sets[s2]))
            union = len(all_extracted_sets[s1].union(all_extracted_sets[s2]))
            
            jaccard_sim = (intersection / union) if union > 0 else 0.0
            
            # Format as "Count (Jaccard%)" for maximum readability
            overlap_jaccard_df.loc[s1, s2] = f"{intersection} ({jaccard_sim * 100:.1f}%)"

    # 6. Format and Save Summary Results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    
    for col in ["Macro Precision", "Macro Recall", "Macro F1", "Micro Precision", "Micro Recall", "Micro F1", "Hallucination Rate"]:
        df_results[col] = (df_results[col] * 100).round(2).astype(str) + "%"

    report_text = "="*130 + "\n"
    report_text += f" 🏆 KNOWLEDGE GRAPH EXTRACTION SUMMARY REPORT ({judge_name.upper()})\n"
    report_text += "="*130 + "\n"
    report_text += df_results.to_string(index=False) + "\n"
    
    # 7. Format Detailed Per-Relation Results
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed = df_detailed.sort_values(by=["Source", "Gold_Support"], ascending=[True, False])
    
    for col in ["Precision", "Recall", "F1"]:
        df_detailed[col] = (df_detailed[col] * 100).round(2).astype(str) + "%"

    report_text += "\n" + "="*130 + "\n"
    report_text += f" 📊 PER-RELATION BREAKDOWN (STRICT ONTOLOGY)\n"
    report_text += "="*130 + "\n"
    report_text += df_detailed.to_string(index=False) + "\n"
    
    # 8. Append Jaccard Overlap Matrix to Report
    report_text += "\n" + "="*130 + "\n"
    report_text += f" 🔗 PAIRWISE JACCARD SIMILARITY MATRIX\n"
    report_text += "="*130 + "\n"
    report_text += "(Format: Raw Intersection Count (Jaccard Overlap %))\n\n"
    report_text += overlap_jaccard_df.to_string() + "\n"
    report_text += "="*130 + "\n"

    # Print full report to console
    print("\n" + report_text)
    
    # Save the combined text report
    txt_path = f"./eval/{judge_name}_summary_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Save DataFrames to CSV
    csv_path = f"./eval/{judge_name}_summary_report.csv"
    df_results.to_csv(csv_path, index=False)
    
    detailed_csv_path = f"./eval/{judge_name}_per_relation_report.csv"
    df_detailed.to_csv(detailed_csv_path, index=False)
    
    # Save Overlap Matrix to CSV
    overlap_jaccard_path = f"./eval/{judge_name}_overlap_jaccard.csv"
    overlap_jaccard_df.to_csv(overlap_jaccard_path)
    
    print(f"✅ Summary CSV saved to {csv_path}")
    print(f"✅ Detailed Per-Relation CSV saved to {detailed_csv_path}")
    print(f"✅ Jaccard Overlap matrix saved to {overlap_jaccard_path}")

if __name__ == "__main__":
    main()
