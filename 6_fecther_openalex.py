# -*- coding: utf-8 -*-
# 6_fetcher.py

from datetime import datetime, timedelta
import requests
import json
import time
import os

# API Settings
BASE_URL = "https://api.openalex.org/works"

class MonthlyOpenAlexFetcher:
    def __init__(self, start_date="2026-01-01", end_date="2026-01-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.select_fields = "id,doi,title,abstract_inverted_index,publication_year,publication_date,type,concepts,topics"
        
        # High-Quality Publishers & Sources
        publishers = [
            "P4310320990", "P4310320595", "P4310319900", "P4310320547", 
            "P4310311648", "P4310320017", "P4310319965", "P4310319908", 
            "P4310319808", "P4310319798"
        ]
        sources = ["S4306400194"] # arXiv
        
        self.pub_filter = "|".join(publishers)
        self.src_filter = "|".join(sources)
        self.ai_concept = "concepts.id:C154945302"
        self.ai_topics = "topics.subfield.id:1702|1707|1703|1711|1710" 

        self.params = {
            "select": self.select_fields,
            "per_page": 200,  # Maximized for bulk download
            "mailto": "ddessi@sharjah.ac.ae"
        }
        
        self.seen_titles = set()
        self.master_dataset = {}

    def reconstruct_abstract(self, inverted_index: dict) -> str:
        if not inverted_index:
            return ""
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        abstract_words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                abstract_words[pos] = word
        return " ".join(abstract_words)

    def is_valid_ai_paper(self, paper: dict) -> bool:
        title = paper.get('title')
        if not title: return False
        
        title_norm = " ".join(title.lower().split())
        if title_norm in self.seen_titles: return False
        if title_norm.startswith("system and method") or title_norm.startswith("apparatus for"): return False

        is_ai = False
        for concept in paper.get('concepts', []):
            if concept.get('id', '').endswith('C154945302') and concept.get('score', 0) >= 0.5:
                is_ai = True
                break 
        
        if not is_ai:
            for topic in paper.get('topics', []):
                subfield_id = topic.get('subfield', {}).get('id', '')
                if any(subfield_id.endswith(str(tid)) for tid in ['1702', '1707', '1703', '1711']):
                    is_ai = True
                    break

        if not is_ai: return False
        
        self.seen_titles.add(title_norm)
        return True

    def fetch_timeframe(self) -> dict:
        date_filter = f"from_publication_date:{self.start_date},to_publication_date:{self.end_date}"
        common_filters = f"{date_filter},is_retracted:False,type:article|proceedings-article|preprint|book-chapter,is_paratext:False"
        
        filter_queries = [
            f"{common_filters},{self.ai_concept},locations.source.publisher_lineage:{self.pub_filter}",
            f"{common_filters},{self.ai_topics},locations.source.publisher_lineage:{self.pub_filter}",
            f"{common_filters},{self.ai_concept},locations.source.id:{self.src_filter}",
            f"{common_filters},{self.ai_topics},locations.source.id:{self.src_filter}"
        ]

        collected_papers = {}

        for query_string in filter_queries:
            self.params['filter'] = query_string
            self.params['cursor'] = "*" 
            has_more_pages = True
            
            while has_more_pages:
                try:
                    response = requests.get(BASE_URL, params=self.params)
                    response.raise_for_status()
                    data = response.json()

                    works = data.get('results', [])
                    next_cursor = data.get('meta', {}).get('next_cursor')

                    for paper in works:
                        if self.is_valid_ai_paper(paper):
                            inv_index = paper.get("abstract_inverted_index")
                            abstract = self.reconstruct_abstract(inv_index)
                            
                            if abstract and len(abstract.split()) >= 60:
                                pid = paper['id']
                                collected_papers[pid] = {
                                    'doi': paper.get('doi'), 
                                    'title': paper.get('title'), 
                                    'abstract': abstract, 
                                    'type': paper.get('type', 'unknown'),
                                    'date': paper.get('publication_date', '')
                                }

                    print(f"--> Fetched batch. Total unique papers so far: {len(collected_papers)}")

                    if next_cursor:
                        self.params['cursor'] = next_cursor
                        time.sleep(0.1)  
                    else:
                        has_more_pages = False

                except Exception as e:
                    print(f"Error encountered: {e}")
                    has_more_pages = False

        return collected_papers

    def run(self):
        print(f"Starting unrestricted fetch for period: {self.start_date} to {self.end_date}")
        self.master_dataset = self.fetch_timeframe()

        save_directory = './data/'
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, f"papers_{self.start_date[:7].replace('-', '_')}.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.master_dataset, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Collection complete! Saved {len(self.master_dataset)} papers to {file_path}")

if __name__ == "__main__":
    fetcher = MonthlyOpenAlexFetcher(start_date="2026-01-01", end_date="2026-01-31")
    fetcher.run()
