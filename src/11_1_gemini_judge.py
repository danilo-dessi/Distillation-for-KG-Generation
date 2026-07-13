# -*- coding: utf-8 -*-
import os
import glob
import json
import time
import threading
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures
from google import genai
from google.genai import types
from pydantic import BaseModel

class TripleScore(BaseModel):
    signature_id: str
    is_correct: int

class BatchEvaluation(BaseModel):
    evaluations: list[TripleScore]

CONFIG = {
    "OUTPUT_FILE": "./eval/gemini_judge_results.json",
    "MODEL_NAME": "gemini-2.5-flash",
    "MAX_WORKERS": 10,
    "DATA_DIR": "./eval/"
}

class GeminiBatchedJudge:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))

    def evaluate_paper_batch(self, paper_id: str, paper_data: dict) -> list:
        abstract = paper_data['abstract_text']
        triples_to_grade = paper_data['triples']
        triples_formatted = "\n".join([f"ID: {t['signature_id']} | Triple: [{t['subject_type']}] {t['subject']} --({t['predicate']})--> [{t['object_type']}] {t['object']}" for t in triples_to_grade])

        prompt = f"""You are an impartial evaluator grading Knowledge Graph extractions.
Read the abstract. Evaluate every Triple. Return 1 if perfectly correct, 0 if hallucinated/wrong.
Abstract: {abstract}
Triples:\n{triples_formatted}"""

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=CONFIG['MODEL_NAME'], contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=BatchEvaluation, temperature=0.0)
                )
                score_map = {ev['signature_id']: ev['is_correct'] for ev in json.loads(response.text).get('evaluations', [])}
                return [{**t, 'judge_score': score_map.get(t['signature_id'], 0)} for t in triples_to_grade]
            except Exception:
                time.sleep(2 ** attempt)
        return [{**t, 'judge_score': 0} for t in triples_to_grade]

def main():
    print("\n--- Starting Batched Gemini Judge ---")
    
    # --- NEW: RESUME LOGIC ---
    graded_results = []
    graded_signatures = set()
    
    if os.path.exists(CONFIG['OUTPUT_FILE']):
        print(f"--> Found existing judge results at {CONFIG['OUTPUT_FILE']}. Loading...")
        try:
            with open(CONFIG['OUTPUT_FILE'], 'r', encoding='utf-8') as f:
                graded_results = json.load(f)
                graded_signatures = {item['signature_id'] for item in graded_results if 'signature_id' in item}
            print(f"--> Skipping {len(graded_signatures)} previously evaluated triples.")
        except json.JSONDecodeError:
            print("--> Existing judge file is empty or corrupted. Starting fresh.")
    # -------------------------

    json_files = [f for f in glob.glob(f"{CONFIG['DATA_DIR']}*.json") if "judge" not in f]
    
    unique_triples, papers_dict = {}, {}
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # --- Normalization Check ---
            if isinstance(data, dict):
                if 'paper_id' in data:
                    data = [data]
                elif 'data' in data and isinstance(data['data'], list):
                    data = data['data']
                else:
                    data = list(data.values())
            
            for item in data:
                if not isinstance(item, dict) or not item.get('paper_id'):
                    continue
                    
                sig = f"{item.get('paper_id')}|{item.get('subject')}|{item.get('predicate')}|{item.get('object')}"
                
                # --- NEW: FILTER OUT ALREADY GRADED TRIPLES ---
                if sig not in graded_signatures:
                    unique_triples[sig] = {**item, "signature_id": sig}

    if not unique_triples:
        print("✅ All triples in all files have already been judged. Nothing new to process!")
        return

    print(f"--> Found {len(unique_triples)} NEW unique triples to evaluate.")

    for sig, triple in unique_triples.items():
        if triple['paper_id'] not in papers_dict:
            papers_dict[triple['paper_id']] = {'abstract_text': triple.get('abstract_text',''), 'triples': []}
        papers_dict[triple['paper_id']]['triples'].append(triple)

    judge = GeminiBatchedJudge()
    completed = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
        futures = {executor.submit(judge.evaluate_paper_batch, pid, p_data): pid for pid, p_data in papers_dict.items()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(papers_dict)):
            completed.extend(future.result())

    # --- NEW: COMBINE AND SAVE ---
    final_results = graded_results + completed

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    print(f"✅ Saved {len(final_results)} total evaluations to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    main()