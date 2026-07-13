# Distillation-for-KG-Generation

This repository contains all the Python files used to create the SMLs as detailed in the paper:

```


```

All the models created and used in this work are available through this Zenodo link:

## Directory src/

- openalex_fetcher.py: A data collection script that queries the OpenAlex API to harvest 10 AI-related scientific papers per day, targeting a total of 3,650 papers across the 2025 calendar year. It reconstructs abstract texts from inverted indices and filters results by high-quality publishers and specific AI concepts to build a balanced dataset.
- deepseek_ner.py: An asynchronous script leveraging the DeepSeek API to perform Named Entity Recognition (NER) on the collected abstracts. It extracts specific scientific entities—such as Task, Model, and Dataset—and relies on Pydantic schemas to validate the JSON output.  
- gemini_ner.py: An asynchronous entity extraction tool utilizing the Gemini API (gemini-2.5-flash). It processes batches of academic abstracts using Google's structured content generation to classify and extract scientific terminology into predefined categories.  
- hf_llama_ner.py: A local inference script that employs Hugging Face's pipeline with the Llama-3.1-8B-Instruct model to run sequential NER extraction. It uses precise prompt engineering and regex-based parsing to enforce strict JSON output without relying on external API structural constraints.  
- openai_ner.py: An asynchronous annotation script powered by the OpenAI API and the gpt-4o-mini model. It utilizes OpenAI's structured outputs feature alongside exponential backoff for rate-limit handling to reliably extract and categorize entities from the papers.