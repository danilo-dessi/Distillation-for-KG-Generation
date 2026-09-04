# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Pointing to the EXCEL file your researchers are actively filling out
    "COMPLETED_EXCEL": "./eval/human_annotation_sample.xlsx", 
    
    # The JSON files containing the automated judge scores
    "JUDGE_FILES": {
        "GPT-4o-mini": "./eval/openai_judge_results.json",
        "Gemini-2.5-Flash": "./eval/gemini_judge_results.json",
        "DeepSeek": "./eval/deepseek_judge_results.json"
    },
    "RESEARCHER_COLS": [f"Researcher_{i}" for i in range(1, 6)]
}

def load_judge_data(filepath):
    """Loads a judge JSON and returns a dictionary of {signature_id: is_correct}"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['signature_id']: item.get('judge_score', 0) for item in data if 'signature_id' in item}

def analyze_evaluations():
    print("\n--- Starting Human vs. LLM Judge Evaluation ---")
    
    if not os.path.exists(CONFIG["COMPLETED_EXCEL"]):
        print(f"❌ Error: Could not find {CONFIG['COMPLETED_EXCEL']}")
        print("Please ensure your researchers have saved the Excel file to this path.")
        return

    # 1. Load Human Annotations from Excel
    print(f"--> Loading human annotations from: {CONFIG['COMPLETED_EXCEL']}")
    df = pd.read_excel(CONFIG["COMPLETED_EXCEL"], engine='openpyxl')
    
    # Convert researcher columns to numeric, forcing blanks/strings to NaN
    for col in CONFIG["RESEARCHER_COLS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Count how many human annotations exist per row
    df['num_annotations'] = df[CONFIG["RESEARCHER_COLS"]].notna().sum(axis=1)
    
    # Isolate only the rows that have at least one human grade
    df_valid = df[df['num_annotations'] > 0].copy()
    
    if df_valid.empty:
        print("❌ No human annotations found in the Excel file yet. Exiting.")
        return
        
    print(f"--> Found {len(df_valid)} distinct triples with at least one human annotation.")

    # 2. Establish Human Ground Truth (Dynamic Majority Vote)
    # Pandas .mean() automatically ignores NaN values. 
    df_valid['Human_Consensus'] = (df_valid[CONFIG["RESEARCHER_COLS"]].mean(axis=1) >= 0.5).astype(int)
    
    consensus_dist = df_valid['Human_Consensus'].value_counts()
    print(f"\n[1] HUMAN GROUND TRUTH DISTRIBUTION")
    print(f"--> Correct Triples (1): {consensus_dist.get(1, 0)}")
    print(f"--> Incorrect Triples (0): {consensus_dist.get(0, 0)}")

    # 3. Calculate Inter-Annotator Agreement (If Possible)
    print("\n[2] INTER-ANNOTATOR AGREEMENT (IAA)")
    active_cols = [col for col in CONFIG["RESEARCHER_COLS"] if col in df_valid.columns and df_valid[col].notna().any()]
    
    if len(active_cols) >= 2:
        # To calculate Fleiss' Kappa, we need rows where ALL active researchers provided a vote
        df_iaa = df_valid.dropna(subset=active_cols)
        if len(df_iaa) > 0:
            ratings_matrix = df_iaa[active_cols].values
            agg_ratings, _ = aggregate_raters(ratings_matrix)
            kappa = fleiss_kappa(agg_ratings)
            
            print(f"--> Fleiss' Kappa ({len(active_cols)} active raters on {len(df_iaa)} overlapping triples): {kappa:.4f}")
            if kappa < 0: print("    Interpretation: Poor agreement")
            elif kappa <= 0.20: print("    Interpretation: Slight agreement")
            elif kappa <= 0.40: print("    Interpretation: Fair agreement")
            elif kappa <= 0.60: print("    Interpretation: Moderate agreement")
            elif kappa <= 0.80: print("    Interpretation: Substantial agreement")
            else: print("    Interpretation: Almost perfect agreement")
        else:
            print("--> Cannot compute IAA: Active researchers have not graded the same overlapping triples yet.")
    else:
        print(f"--> Skipping IAA: Only {len(active_cols)} active researcher(s) found. Need at least 2.")

    # Reconstruct the signature ID to match the JSON files
    df_valid['signature_id'] = df_valid.apply(
        lambda row: f"{row['paper_id']}|{row['subject']}|{row['predicate']}|{row['object']}", 
        axis=1
    )

    # 4. Compare LLM Judges to Human Consensus
    print("\n[3] LLM JUDGE EVALUATION vs. HUMAN CONSENSUS")
    y_true = df_valid['Human_Consensus'].values

    for judge_name, filepath in CONFIG["JUDGE_FILES"].items():
        judge_dict = load_judge_data(filepath)
        
        if not judge_dict:
            print(f"\n⚠️ Skipping {judge_name}: File not found ({filepath})")
            continue
            
        df_valid[judge_name] = df_valid['signature_id'].map(lambda sig: judge_dict.get(sig, None))
        
        missing_count = df_valid[judge_name].isna().sum()
        if missing_count > 0:
            valid_idx = df_valid[judge_name].notna()
            y_true_valid = y_true[valid_idx]
            y_pred = df_valid.loc[valid_idx, judge_name].astype(int).values
        else:
            y_true_valid = y_true
            y_pred = df_valid[judge_name].astype(int).values

        if len(y_true_valid) == 0:
            continue

        acc = accuracy_score(y_true_valid, y_pred)
        prec = precision_score(y_true_valid, y_pred, zero_division=0)
        rec = recall_score(y_true_valid, y_pred, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred, zero_division=0)

        # McNemar's Test
        table = [[0, 0], [0, 0]]
        for t, p in zip(y_true_valid, y_pred):
            if t == 1 and p == 1: table[0][0] += 1
            elif t == 1 and p == 0: table[0][1] += 1
            elif t == 0 and p == 1: table[1][0] += 1
            elif t == 0 and p == 0: table[1][1] += 1

        try:
            mcnemar_result = mcnemar(table, exact=True)
            p_value = mcnemar_result.pvalue
        except ValueError:
            p_value = 1.0

        print(f"\n--- {judge_name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        sig_marker = "*" if p_value < 0.05 else ""
        print(f"McNemar's p-value: {p_value:.10f} {sig_marker}")
        if p_value < 0.05:
            print("  -> Significant difference from human consensus (p < 0.05).")
        else:
            print("  -> NO significant difference from human consensus.")

    # 5. Extraction Framework Performance
    print("\n[4] EXTRACTION FRAMEWORK PERFORMANCE (Human-Evaluated Precision)")
    if 'source' in df_valid.columns:
        # Explode comma-separated sources so each model gets full credit for shared triples
        df_exploded = df_valid.assign(source=df_valid['source'].astype(str).str.split(', ')).explode('source')
        
        # Group by the individual extraction model
        source_perf = df_exploded.groupby('source')['Human_Consensus'].agg(['count', 'sum', 'mean'])
        source_perf = source_perf.rename(columns={'count': 'Total Sampled', 'sum': 'Correct Triples', 'mean': 'Precision'})
        
        # Sort by Precision descending
        source_perf = source_perf.sort_values(by='Precision', ascending=False)
        
        for source, row in source_perf.iterrows():
            print(f"\n--- {source} ---")
            print(f"Sample Size:     {int(row['Total Sampled'])}")
            print(f"Correct Triples: {int(row['Correct Triples'])}")
            print(f"Precision:       {row['Precision']:.4f} ({row['Precision'] * 100:.1f}%)")

        # 6. Statistical Significance of Precision Differences (Table 11)
        print("\n[5] STATISTICAL SIGNIFICANCE OF PRECISION DIFFERENCES")

        # Wilson 95% CIs for each source (better small-sample behaviour than a normal-approx CI)
        print("\n--> 95% Wilson confidence intervals:")
        ci_lookup = {}
        for source, row in source_perf.iterrows():
            n_correct = int(row['Correct Triples'])
            n_total = int(row['Total Sampled'])
            lo, hi = proportion_confint(n_correct, n_total, alpha=0.05, method='wilson')
            ci_lookup[source] = (n_correct, n_total)
            print(f"    {source}: {row['Precision']*100:.1f}%  [{lo*100:.1f}%, {hi*100:.1f}%]  (n={n_total})")

        if len(ci_lookup) >= 2:
            # Global test: are precision rates homogeneous across all sources?
            contingency = np.array([[n, N - n] for n, N in ci_lookup.values()])
            chi2, p_global, dof, _ = chi2_contingency(contingency)
            print(f"\n--> Global chi-square test across all {len(ci_lookup)} sources: "
                  f"chi2={chi2:.3f}, dof={dof}, p={p_global:.2e}")

            # Pairwise Fisher's exact tests: top performer vs. every other source,
            # Holm-Bonferroni corrected for multiple comparisons
            best_source = source_perf['Precision'].idxmax()
            best_n, best_N = ci_lookup[best_source]
            print(f"\n--> Pairwise Fisher's exact tests vs. top performer "
                  f"({best_source}, {best_n}/{best_N} = {best_n/best_N:.1%}):")

            pvals, comparisons = [], []
            for source, (n, N) in ci_lookup.items():
                if source == best_source:
                    continue
                table = [[best_n, best_N - best_n], [n, N - n]]
                _, p = fisher_exact(table)
                pvals.append(p)
                comparisons.append(source)

            if pvals:
                reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='holm')
                for source, p_raw, p_corr, sig in zip(comparisons, pvals, p_adj, reject):
                    marker = "*" if sig else ""
                    print(f"    {best_source} vs {source}: p={p_raw:.6f}, "
                          f"Holm-corrected p={p_corr:.6f} {marker}")
        else:
            print("--> Need at least 2 sources to run comparative significance tests.")
    else:
        print("⚠️ 'source' column not found in the Excel file.")

if __name__ == "__main__":
    analyze_evaluations()