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
# PYDANTIC SCHEMA (Forces Perfect Triples)
# ==============================================================================
class Triple(BaseModel):
    subject: str = Field(description="Must exactly match a provided entity name.")
    relationship: str = Field(description="The canonical relationship type.")
    object: str = Field(description="Must exactly match a provided entity name.")

class RelationExtractionResult(BaseModel):
    relationships: list[Triple]

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "INPUT_FILE": "./data/gemini_NER_annotated_balanced_papers_2025.json",
    "OUTPUT_FILE": "./data/gemini_RE_annotated_balanced_papers_2025.json",
    "RELATIONS_FILE": "../resources/relationships.json",
    "MODEL_NAME": "gemini-2.5-flash", 
    "CONCURRENCY_LIMIT": 15,
    "SAVE_INTERVAL": 60 
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
4. Only extract explicit, positive relationships.

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
                response = await client.aio.models.generate_content(
                    model=CONFIG['MODEL_NAME'],
                    contents=build_prompt(abstract, entities, ontology),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RelationExtractionResult,
                        temperature=0.0 # Zero temp for strict extraction
                    )
                )

                result_data = json.loads(response.text)
                paper_data['relationships'] = result_data.get('relationships', [])
                output_data[pid] = paper_data
                break
                
            except Exception as e:
                if attempt == 4:
                    print(f"\nFailed on {pid}: {e}")
                    paper_data['relationships'] = []
                    output_data[pid] = paper_data
                await asyncio.sleep(2 ** attempt)

async def main():
    load_dotenv()
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
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
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting Gemini Relations"):
        await f
        processed += 1
        if processed % CONFIG['SAVE_INTERVAL'] == 0:
            with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as out_f:
                json.dump(output_data, out_f, indent=4, ensure_ascii=False)

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as out_f:
        json.dump(output_data, out_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
