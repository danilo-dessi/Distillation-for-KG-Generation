# -*- coding: utf-8 -*-
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
    "OUTPUT_FILE": "./eval/pipeline_eval_gemini_zero_shot.json",
    "MODEL_NAME": "gemini-2.5-flash",
    "MAX_WORKERS": 10
}

import time
from google.api_core import exceptions as google_exceptions

class GeminiZeroShotExtractor:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))

    def process_abstract(self, paper_id: str, abstract: str, retries=3) -> list:
        prompt = f"""You are an expert AI researcher performing Joint Entity and Relation Extraction.
Extract valid knowledge triples from this abstract using standard scientific entities (Task, Model, Algorithm, Dataset, Metric, Hardware) and predicates (USES, PRODUCES, ANALYZES, INCLUDES, IS_A, COMPARES_TO, INFLUENCES, SHOWS, IDENTIFIES, ADDRESSES, EVALUATED_ON, ACHIEVES).

Abstract:
{abstract}"""
        
        attempt = 0
        while attempt < retries:
            try:
                response = self.client.models.generate_content(
                    model=CONFIG['MODEL_NAME'],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                        temperature=0.0 
                    )
                )
                result_data = json.loads(response.text)
                return [{
                    "paper_id": paper_id, "abstract_text": abstract,
                    "subject": t.get("subject", ""), "subject_type": t.get("subject_type", ""),
                    "predicate": t.get("predicate", ""), "object": t.get("object", ""),
                    "object_type": t.get("object_type", ""), "source": "gemini-zero-shot"
                } for t in result_data.get('triples', [])]
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it is a hard billing error
                if "prepayment credits are depleted" in error_msg:
                    print(f"\n❌ FATAL BILLING ERROR on paper {paper_id}. Halting thread.")
                    # Actually raise the error so the script knows to stop
                    raise Exception("Billing Depleted. Please check Google AI Studio.")
                
                # Check if it's a standard rate limit (too many requests)
                elif "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    attempt += 1
                    wait_time = 2 ** attempt # Waits 2s, then 4s, then 8s
                    print(f"\n⏳ Rate limited on paper {paper_id}. Retrying in {wait_time}s... (Attempt {attempt}/{retries})")
                    time.sleep(wait_time)
                
                # For any other weird error (JSON parsing, server timeout, etc)
                else:
                    print(f"\n⚠️ Skipped paper {paper_id} due to API Error: {e}")
                    return []
                    
        # If it runs out of retries
        print(f"\n⚠️ Skipped paper {paper_id} after {retries} failed retries.")
        return []
def main():
    print("\n--- Starting Zero-Shot Gemini Extraction ---")
    df_pipeline = pd.read_json(CONFIG['INPUT_FILE'])
    unique_papers = df_pipeline[['paper_id', 'abstract_text']].drop_duplicates().to_dict('records')
    
    all_triples = []
    processed_paper_ids = set()

    # --- NEW RESUME LOGIC ---
    if os.path.exists(CONFIG['OUTPUT_FILE']):
        print("Found existing checkpoint. Loading previous progress...")
        try:
            with open(CONFIG['OUTPUT_FILE'], 'r', encoding='utf-8') as f:
                all_triples = json.load(f)
                # Collect all paper_ids that we've already successfully processed
                processed_paper_ids = {t['paper_id'] for t in all_triples}
        except json.JSONDecodeError:
            print("Checkpoint file was empty or corrupted. Starting fresh.")
            
    # Filter the papers list to only include ones we haven't done yet
    papers_to_process = [p for p in unique_papers if p['paper_id'] not in processed_paper_ids]
    
    print(f"Total unique papers: {len(unique_papers)}")
    print(f"Already processed: {len(processed_paper_ids)}")
    print(f"Remaining to process: {len(papers_to_process)}")
    # ------------------------

    if not papers_to_process:
        print("✅ All papers have already been processed!")
        return

    extractor = GeminiZeroShotExtractor()
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
        # Use the filtered list (papers_to_process) instead of unique_papers
        future_to_paper = {executor.submit(extractor.process_abstract, p['paper_id'], p['abstract_text']): p['paper_id'] for p in papers_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_paper), total=len(papers_to_process)):
            # Append the new triples to our existing list
            all_triples.extend(future.result())
            completed_count += 1
            
            # Save checkpoint every 10 papers
            if completed_count % 10 == 0:
                with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
                    json.dump(all_triples, f, indent=4)
                    
            time.sleep(0.1)

    # Final save to ensure any remaining triples are written
    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump(all_triples, f, indent=4)
    print(f"✅ Saved {len(all_triples)} total triples to {CONFIG['OUTPUT_FILE']}")

if __name__ == "__main__":
    main()