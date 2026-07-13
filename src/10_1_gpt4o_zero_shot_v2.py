# -*- coding: utf-8 -*-
import os
import json
import asyncio
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

class Triple(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str

class ExtractionResult(BaseModel):
    triples: list[Triple]

CONFIG = {
    "INPUT_FILE": "./eval/pipeline_eval.json",
    "OUTPUT_FILE": "./eval/openai_v2_zero_shot.json",
    "ONTOLOGY_FILE": "../resources/relationships.json", # Added Ontology Path
    "MODEL_NAME": "gpt-4o-mini",
    "CONCURRENCY_LIMIT": 10
}

def load_ontology_prompt():
    """Loads the JSON ontology and formats it into a strict instruction string."""
    if not os.path.exists(CONFIG["ONTOLOGY_FILE"]):
        raise FileNotFoundError(f"Ontology file not found at {CONFIG['ONTOLOGY_FILE']}")
        
    with open(CONFIG["ONTOLOGY_FILE"], 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
        
    relationships = ontology_data.get("relationships", {})
    
    prompt_lines = ["\nALLOWED PREDICATES AND THEIR DEFINITIONS:"]
    for predicate, details in relationships.items():
        definition = details.get("definition", "")
        prompt_lines.append(f"- {predicate}: {definition}")
        
    return "\n".join(prompt_lines)

async def process_abstract(client, paper_id, abstract, semaphore, results_list, ontology_instructions):
    # Dynamic prompt using the loaded ontology
    prompt = f"""You are an expert AI researcher performing Joint Entity and Relation Extraction.
Extract valid knowledge triples from this abstract using standard scientific entities (Task, Model, Algorithm, Dataset, Metric, Hardware).

CRITICAL INSTRUCTION: You must strictly use ONLY the predicates listed below. Do not invent, hallucinate, or modify these predicates in any way.
{ontology_instructions}

Abstract:
{abstract}"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.beta.chat.completions.parse(
                    model=CONFIG['MODEL_NAME'],
                    messages=[{"role": "user", "content": prompt}],
                    response_format=ExtractionResult,
                    temperature=0.0
                )
                
                result_data = response.choices[0].message.parsed
                if result_data:
                    for t in result_data.triples:
                        results_list.append({
                            "paper_id": paper_id, "abstract_text": abstract,
                            "subject": t.subject, "subject_type": t.subject_type,
                            "predicate": t.predicate, "object": t.object,
                            "object_type": t.object_type, "source": "openai-zero-shot"
                        })
                break
            except Exception as e:
                if attempt == 2: print(f"\n⚠️ Skipped {paper_id}: {e}")
                await asyncio.sleep(2 ** attempt)

async def main():
    print("\n--- Starting Zero-Shot OpenAI Extraction ---")
    
    # 1. Load the Ontology constraints
    try:
        ontology_instructions = load_ontology_prompt()
        print("--> Successfully loaded ontology constraints.")
    except Exception as e:
        print(f"❌ Error loading ontology: {e}")
        return

    load_dotenv()
    client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    df_pipeline = pd.read_json(CONFIG['INPUT_FILE'])
    unique_papers = df_pipeline[['paper_id', 'abstract_text']].drop_duplicates().to_dict('records')
    
    semaphore = asyncio.Semaphore(CONFIG['CONCURRENCY_LIMIT'])
    all_triples = []
    
    # Pass the ontology_instructions to each task
    tasks = [process_abstract(client, p['paper_id'], p['abstract_text'], semaphore, all_triples, ontology_instructions) for p in unique_papers]
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        await f

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(all_triples, f, indent=4)
    print(f"✅ Saved {len(all_triples)} triples to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    asyncio.run(main())
