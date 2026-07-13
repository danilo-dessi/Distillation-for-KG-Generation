# -*- coding: utf-8 -*-

import os
import json
import asyncio
import openai
from tqdm import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class Triple(BaseModel):
    subject: str = Field(description="Must exactly match a provided entity name.")
    relationship: str = Field(description="The canonical relationship type.")
    object: str = Field(description="Must exactly match a provided entity name.")

class RelationExtractionResult(BaseModel):
    relationships: list[Triple]

CONFIG = {
    "INPUT_FILE": "./data/gpt40mini-annotated_balanced_papers_2025.json",
    "OUTPUT_FILE": "./data/gpt40mini_RE_annotated_balanced_papers_2025.json",
    "RELATIONS_FILE": "../resources/relationships.json",
    "MODEL_NAME": "gpt-4o-mini",
    "CONCURRENCY_LIMIT": 4, # Kept lower to respect the 200k TPM limit
    "SAVE_INTERVAL": 100
}

def load_ontology():
    with open(CONFIG['RELATIONS_FILE'], 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('relationships', {})

def build_prompt(abstract, entities, ontology):
    rel_defs = "\n".join([f"- {k}: {v['definition']}" for k, v in ontology.items()])
    ent_list = [f"{e['name']} ({e['type']})" for e in entities]
    
    return f"""You are a meticulous expert in scientific information extraction.
Extract relationships between named entities from the abstract below.

Instructions:
1. Extract relationships as triples (subject, relationship, object).
2. The 'subject' and 'object' MUST EXACTLY match the name of one of the provided entities.
3. Classify each relationship into ONE of the canonical types provided below.

Canonical Relationship Types:
{rel_defs}

Entities Available:
{json.dumps(ent_list, indent=2)}

Abstract:
{abstract}
"""

async def process_paper(client, pid, paper_data, ontology, semaphore, output_data):
    abstract = paper_data.get('abstract', '')
    entities = paper_data.get('entities', [])
    
    if not abstract or len(entities) < 2:
        paper_data['relationships'] = []
        output_data[pid] = paper_data
        return

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.beta.chat.completions.parse(
                    model=CONFIG['MODEL_NAME'],
                    messages=[{"role": "user", "content": build_prompt(abstract, entities, ontology)}],
                    response_format=RelationExtractionResult,
                    temperature=0.0
                )

                result_data = response.choices[0].message.parsed
                paper_data['relationships'] = [r.model_dump() for r in result_data.relationships]
                output_data[pid] = paper_data
                break
                
            except openai.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"\nFailed on {pid}: {e}")
                    paper_data['relationships'] = []
                    output_data[pid] = paper_data
                await asyncio.sleep(2 ** attempt)

async def main():
    load_dotenv()
    client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    ontology = load_ontology()

    with open(CONFIG['INPUT_FILE'], 'r', encoding='utf-8') as f:
        papers = json.load(f)

    output_data = {}
    if os.path.exists(CONFIG['OUTPUT_FILE']):
        with open(CONFIG['OUTPUT_FILE'], 'r', encoding='utf-8') as f:
            output_data = json.load(f)

    papers_to_process = {pid: p for pid, p in papers.items() if pid not in output_data}
    print(f"Papers to process: {len(papers_to_process)}")

    semaphore = asyncio.Semaphore(CONFIG['CONCURRENCY_LIMIT'])
    tasks = [process_paper(client, pid, p_data, ontology, semaphore, output_data) 
             for pid, p_data in papers_to_process.items()]

    processed = 0
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting GPT-4o-mini Relations"):
        await f
        processed += 1
        if processed % CONFIG['SAVE_INTERVAL'] == 0:
            with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as out_f:
                json.dump(output_data, out_f, indent=4, ensure_ascii=False)

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as out_f:
        json.dump(output_data, out_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
