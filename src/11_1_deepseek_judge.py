# -*- coding: utf-8 -*-
import os
import glob
import json
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

class TripleScore(BaseModel):
    signature_id: str
    is_correct: int

class BatchEvaluation(BaseModel):
    evaluations: list[TripleScore]

CONFIG = {
    "OUTPUT_FILE": "./eval/deepseek_judge_results.json",
    "MODEL_NAME": "deepseek-chat",
    "BASE_URL": "https://api.deepseek.com",
    "CONCURRENCY_LIMIT": 10,
    "DATA_DIR": "./eval/"
}

async def evaluate_batch(client, paper_id, paper_data, semaphore, completed_list):
    abstract = paper_data['abstract_text']
    triples_to_grade = paper_data['triples']
    triples_formatted = "\n".join([f"ID: {t['signature_id']} | Triple: [{t['subject_type']}] {t['subject']} --({t['predicate']})--> [{t['object_type']}] {t['object']}" for t in triples_to_grade])
    schema_json = json.dumps(BatchEvaluation.model_json_schema(), indent=2)

    prompt = f"""You are an impartial evaluator grading Knowledge Graph extractions.
Read the abstract. Evaluate every Triple. Return 1 if perfectly correct, 0 if hallucinated/wrong.
You MUST respond strictly matching this JSON schema:
{schema_json}

Abstract: {abstract}
Triples:\n{triples_formatted}"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=CONFIG['MODEL_NAME'],
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                result_data = BatchEvaluation.model_validate_json(response.choices[0].message.content)
                score_map = {ev.signature_id: ev.is_correct for ev in result_data.evaluations}
                
                for t in triples_to_grade:
                    t['judge_score'] = score_map.get(t['signature_id'], 0)
                    completed_list.append(t)
                break
            except Exception:
                if attempt == 2:
                    for t in triples_to_grade:
                        t['judge_score'] = 0
                        completed_list.append(t)
                await asyncio.sleep(2 ** attempt)

async def main():
    print("\n--- Starting Batched DeepSeek Judge ---")
    load_dotenv()
    client = AsyncOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url=CONFIG['BASE_URL'])
    
    # --- NEW: RESUME LOGIC ---
    graded_results = []
    graded_signatures = set()
    
    if os.path.exists(CONFIG['OUTPUT_FILE']):
        print(f"--> Found existing judge results at {CONFIG['OUTPUT_FILE']}. Loading...")
        try:
            with open(CONFIG['OUTPUT_FILE'], 'r', encoding='utf-8') as f:
                graded_results = json.load(f)
                # Store the signatures of triples we've already graded
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
        pid = triple.get('paper_id')
        if pid not in papers_dict:
            papers_dict[pid] = {'abstract_text': triple.get('abstract_text', ''), 'triples': []}
        papers_dict[pid]['triples'].append(triple)

    semaphore = asyncio.Semaphore(CONFIG['CONCURRENCY_LIMIT'])
    completed = []
    
    tasks = [evaluate_batch(client, pid, p_data, semaphore, completed) for pid, p_data in papers_dict.items()]
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        await f

    # --- NEW: COMBINE AND SAVE ---
    # Merge the freshly graded triples with the ones we loaded at the start
    final_results = graded_results + completed

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    print(f"✅ Saved {len(final_results)} total evaluations to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
