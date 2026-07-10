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
    "OUTPUT_FILE": "./eval/deepseek_zero_shot.json",
    "MODEL_NAME": "deepseek-chat",
    "BASE_URL": "https://api.deepseek.com",
    "CONCURRENCY_LIMIT": 10
}

async def process_abstract(client, paper_id, abstract, semaphore, results_list):
    schema_json = json.dumps(ExtractionResult.model_json_schema(), indent=2)
    prompt = f"""You are an expert AI researcher performing Joint Entity and Relation Extraction.
Extract valid knowledge triples from this abstract using standard scientific entities (Task, Model, Algorithm, Dataset, Metric, Hardware) and predicates (USES, PRODUCES, ANALYZES, INCLUDES, IS_A, COMPARES_TO, INFLUENCES, SHOWS, IDENTIFIES, ADDRESSES, EVALUATED_ON, ACHIEVES).

You MUST respond strictly matching this JSON schema:
{schema_json}

Abstract:
{abstract}"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=CONFIG['MODEL_NAME'],
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                result_data = ExtractionResult.model_validate_json(response.choices[0].message.content)
                for t in result_data.triples:
                    results_list.append({
                        "paper_id": paper_id, "abstract_text": abstract,
                        "subject": t.subject, "subject_type": t.subject_type,
                        "predicate": t.predicate, "object": t.object,
                        "object_type": t.object_type, "source": "deepseek-zero-shot"
                    })
                break
            except Exception as e:
                if attempt == 2: print(f"\n⚠️ Skipped {paper_id}: {e}")
                await asyncio.sleep(2 ** attempt)

async def main():
    print("\n--- Starting Zero-Shot DeepSeek Extraction ---")
    load_dotenv()
    client = AsyncOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url=CONFIG['BASE_URL'])
    
    df_pipeline = pd.read_json(CONFIG['INPUT_FILE'])
    unique_papers = df_pipeline[['paper_id', 'abstract_text']].drop_duplicates().to_dict('records')
    
    semaphore = asyncio.Semaphore(CONFIG['CONCURRENCY_LIMIT'])
    all_triples = []
    
    tasks = [process_abstract(client, p['paper_id'], p['abstract_text'], semaphore, all_triples) for p in unique_papers]
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        await f

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(all_triples, f, indent=4)
    print(f"✅ Saved {len(all_triples)} triples to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    asyncio.run(main())
