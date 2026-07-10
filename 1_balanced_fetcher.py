# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import urllib.parse
import requests
import json
import time
import os

# API Settings
BASE_URL = "https://api.openalex.org/works"
CORE_API_KEY = None 

class BalancedOpenAlexFetcher:
    def __init__(self, target_year=2025, papers_per_day=10):
        self.target_year = target_year
        self.papers_per_day = papers_per_day
        self.select_fields = "id,doi,title,abstract_inverted_index,publication_year,type,concepts,topics"
        
        # High-Quality Publishers & Sources
        publishers = [
            "P4310320990", "P4310320595", "P4310319900", "P4310320547", 
            "P4310311648", "p4310320017", "P4310319965", "P4310319908", 
            "P4310319808", "P4310319798"
        ]
        sources = ["S4306400194"] # arXiv
        
        self.pub_filter = "|".join(publishers)
        self.src_filter = "|".join(sources)
        self.ai_concept = "concepts.id:C154945302"
        self.ai_topics = "topics.subfield.id:1702|1707|1703|1711|1710" 

        self.params = {
            "select": self.select_fields,
            "per_page": 50,  # Reduced since we only need 10 per day
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

        # Check AI concepts/topics
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

    def fetch_day(self, date_str: str) -> dict:
        common_filters = f"publication_date:{date_str},is_retracted:False,type:article|proceedings-article|preprint|book-chapter,is_paratext:False"
        
        filter_queries = [
            f"{common_filters},{self.ai_concept},locations.source.publisher_lineage:{self.pub_filter}",
            f"{common_filters},{self.ai_topics},locations.source.publisher_lineage:{self.pub_filter}",
            f"{common_filters},{self.ai_concept},locations.source.id:{self.src_filter}",
            f"{common_filters},{self.ai_topics},locations.source.id:{self.src_filter}"
        ]

        daily_papers = {}

        for query_string in filter_queries:
            if len(daily_papers) >= self.papers_per_day:
                break

            self.params['filter'] = query_string
            self.params['cursor'] = "*" 
            has_more_pages = True
            
            while has_more_pages and len(daily_papers) < self.papers_per_day:
                try:
                    response = requests.get(BASE_URL, params=self.params)
                    response.raise_for_status()
                    data = response.json()

                    works = data.get('results', [])
                    next_cursor = data.get('meta', {}).get('next_cursor')

                    for paper in works:
                        if len(daily_papers) >= self.papers_per_day:
                            break
                            
                        if self.is_valid_ai_paper(paper):
                            inv_index = paper.get("abstract_inverted_index")
                            abstract = self.reconstruct_abstract(inv_index)
                            
                            # You can integrate the CORE/S2 fallback here if inverted_index is missing, 
                            # but OpenAlex usually has enough native abstracts to hit 10 per day easily.
                            if abstract and len(abstract.split()) >= 60:
                                pid = paper['id']
                                daily_papers[pid] = {
                                    'doi': paper.get('doi'), 
                                    'title': paper.get('title'), 
                                    'abstract': abstract, 
                                    'type': paper.get('type', 'unknown'),
                                    'date': date_str
                                }

                    if next_cursor:
                        self.params['cursor'] = next_cursor
                        time.sleep(0.1)  
                    else:
                        has_more_pages = False

                except Exception as e:
                    has_more_pages = False
                    break

        return daily_papers

    def run(self):
        start_date = datetime(self.target_year, 1, 1)
        end_date = datetime(self.target_year, 12, 31)
        
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        
        print(f"Starting fetch for {self.target_year}. Targeting {self.papers_per_day} papers/day.")
        
        for i in range(total_days):
            date_str = current_date.strftime("%Y-%m-%d")
            daily_papers = self.fetch_day(date_str)
            self.master_dataset.update(daily_papers)
            
            print(f"[{date_str}] Fetched {len(daily_papers)} papers. Total so far: {len(self.master_dataset)}")
            current_date += timedelta(days=1)

        # Save the master dataset
        save_directory = './data/'
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, f"balanced_papers_{self.target_year}.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.master_dataset, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Collection complete! Saved {len(self.master_dataset)} papers to {file_path}")

if __name__ == "__main__":
    fetcher = BalancedOpenAlexFetcher(target_year=2025, papers_per_day=10)
    fetcher.run()
