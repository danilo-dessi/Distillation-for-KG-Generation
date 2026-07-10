# -*- coding: utf-8 -*-

import os
import json
import asyncio
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==============================================================================
# PYDANTIC SCHEMA
# ==============================================================================
class Entity(BaseModel):
    name: str = Field(description="The exact text span of the entity from the abstract.")
    type: str = Field(description="The classification category of the entity.")

class PaperResult(BaseModel):
    paper_id: str = Field(description="The ID of the paper provided in the prompt.")
    entities: list[Entity]

class BatchNERResult(BaseModel):
    results: list[PaperResult]

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./data/balanced_papers_2025.json",
    "OUTPUT_FILE": "./data/gemini_NER_annotated_balanced_papers_2025.json",
    "MODEL_NAME": "gemini-2.5-flash",
    "BATCH_SIZE": 10,
    "CONCURRENCY_LIMIT": 15, # Tier 2 can handle heavy concurrent requests
    
    "ENTITY_DEFINITIONS": {
        "Task": "Applications, goals, or broader fields of study.",
        "Model": "Specific, named architectures or systems.",
        "Algorithm": "Theoretical methods, techniques, or general architecture families.",
        "Dataset": "Named data collections or benchmarks.",
        "Metric": "Performance measures or evaluation standards.",
        "Hardware": "Physical compute resources or chips."
    }
}

def generate_batch_prompt(batch_papers):
    definitions = "\n".join([f"- {k}: {v}" for k, v in CONFIG['ENTITY_DEFINITIONS'].items()])
    abstracts_text = "".join([f"--- PAPER ID: {pid} ---\n{p_data['abstract']}\n\n" for pid, p_data in batch_papers.items()])

    prompt = f"""
    You are an expert AI researcher performing NER.
    Extract all scientific entities from the batch of abstracts below and classify them into these specific types:

    {definitions}

    Rules:
    1. Extract the EXACT text span as it appears in the abstract.
    2. Do not infer entities that are not explicitly written.
    3. Ensure every Paper ID provided below is included in your JSON response.

    ABSTRACTS:
    {abstracts_text}
    """
    return prompt

async def process_batch(client, batch, semaphore, annotated_papers):
    async with semaphore:
        try:
            # Note the use of .aio.models for asynchronous calls
            response = await client.aio.models.generate_content(
                model=CONFIG['MODEL_NAME'],
                contents=generate_batch_prompt(batch),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchNERResult,
                    temperature=0.1 
                )
            )

            result_data = json.loads(response.text)
            
            for paper_result in result_data.get('results', []):
                pid = paper_result.get('paper_id')
                if pid in batch:
                    annotated_papers[pid] = {
                        "abstract": batch[pid]['abstract'],
                        "entities": paper_result.get('entities', []),
                        "title": batch[pid].get('title', ""),
                        "date": batch[pid].get('date', "")
                    }
        except Exception as e:
            print(f"\nError on batch: {e}")

async def main():
    load_dotenv()
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return

    if not os.path.exists(CONFIG['INPUT_FILE']):
        print(f"❌ Input file not found.")
        return

    with open(CONFIG['INPUT_FILE'], 'r', encoding='utf-8') as f:
        papers = json.load(f)

    annotated_papers = {}
    if os.path.exists(CONFIG['OUTPUT_FILE']):
        with open(CONFIG['OUTPUT_FILE'], 'r', encoding='utf-8') as f:
            annotated_papers = json.load(f)

    papers_to_process = {pid: p for pid, p in papers.items() if pid not in annotated_papers}
    paper_items = list(papers_to_process.items())
    
    batches = [dict(paper_items[i:i + CONFIG['BATCH_SIZE']]) for i in range(0, len(paper_items), CONFIG['BATCH_SIZE'])]
    print(f"Total batches to process: {len(batches)}")

    semaphore = asyncio.Semaphore(CONFIG['CONCURRENCY_LIMIT'])
    tasks = [process_batch(client, batch, semaphore, annotated_papers) for batch in batches]

    # Execute all tasks concurrently and track progress
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Async Batches"):
        await f

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(annotated_papers, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Finished! Saved {len(annotated_papers)} annotations.")

if __name__ == "__main__":
    asyncio.run(main())