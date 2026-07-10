# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./data/re_applied_papers_2026_01.json",
    "OUTPUT_FILE": "./eval/pipeline_eval.json",
    "TARGET_SAMPLE_SIZE": 1000
}

def process_pipeline_data():
    if not os.path.exists(CONFIG["INPUT_FILE"]):
        print(f"❌ Error: Could not find {CONFIG['INPUT_FILE']}")
        return

    print(f"--> Loading pipeline data: {CONFIG['INPUT_FILE']}")
    with open(CONFIG["INPUT_FILE"], 'r', encoding='utf-8') as f:
        data = json.load(f)

    papers_metrics = []
    all_triples = []

    for pid, paper_data in tqdm(data.items(), desc="Analyzing Papers"):
        abstract = paper_data.get('abstract', '')
        rels = paper_data.get('relationships', [])
        ents = paper_data.get('entities', [])

        if not abstract or not rels:
            continue

        ent_map = {str(e.get('name', '')).strip().lower(): e.get('type', 'UNKNOWN') for e in ents}
        paper_relationships = {}

        for rel in rels:
            sub = str(rel.get('subject', '')).strip().lower()
            pred = rel.get('relationship', 'UNKNOWN')
            obj = str(rel.get('object', '')).strip().lower()
            conf = float(rel.get('confidence', 1.0))
            
            if sub and obj and pred != 'UNKNOWN':
                sig = f"{sub}|{pred}|{obj}"
                if sig not in paper_relationships:
                    paper_relationships[sig] = {
                        'paper_id': pid,
                        'abstract_text': abstract,
                        'subject': sub,
                        'subject_type': ent_map.get(sub, 'UNKNOWN'),
                        'predicate': pred,
                        'object': obj,
                        'object_type': ent_map.get(obj, 'UNKNOWN'),
                        'confidence': conf,
                        'source': "SciDeBERTa-Pipeline"
                    }
                else:
                    paper_relationships[sig]['confidence'] = max(paper_relationships[sig]['confidence'], conf)

        unique_preds = set()
        conf_sum = 0
        t_count = len(paper_relationships)

        for stmt in paper_relationships.values():
            unique_preds.add(stmt['predicate'])
            conf_sum += stmt['confidence']
            all_triples.append(stmt)

        if 4 <= t_count <= 15:
            papers_metrics.append({
                'paper_id': pid,
                'triple_count': t_count,
                'unique_predicates': len(unique_preds),
                'avg_confidence': conf_sum / t_count if t_count > 0 else 0
            })

    df_metrics = pd.DataFrame(papers_metrics)
    if df_metrics.empty:
        print("⚠️ No papers fit the richness criteria.")
        return

    df_metrics['norm_predicates'] = df_metrics['unique_predicates'] / df_metrics['unique_predicates'].max()
    df_metrics['norm_confidence'] = df_metrics['avg_confidence'] / df_metrics['avg_confidence'].max()
    df_metrics['norm_count'] = df_metrics['triple_count'] / 15.0 
    df_metrics['richness_score'] = (df_metrics['norm_predicates'] * 0.40) + (df_metrics['norm_confidence'] * 0.40) + (df_metrics['norm_count'] * 0.20)

    best_papers = df_metrics.sort_values(by='richness_score', ascending=False).head(CONFIG["TARGET_SAMPLE_SIZE"])
    target_pids = set(best_papers['paper_id'])
    
    df_triples = pd.DataFrame(all_triples)
    eval_triples = df_triples[df_triples['paper_id'].isin(target_pids)].copy()

    final_export = eval_triples[['paper_id', 'abstract_text', 'subject', 'subject_type', 'predicate', 'object', 'object_type', 'confidence', 'source']]
    
    os.makedirs("./data", exist_ok=True)
    final_export.to_json(CONFIG["OUTPUT_FILE"], orient='records', indent=4)
    print(f"\n✅ Evaluation subset saved to: {CONFIG['OUTPUT_FILE']} (Total triples: {len(final_export)})")

if __name__ == "__main__":
    process_pipeline_data()
