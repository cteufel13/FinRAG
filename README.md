# FinRAG

## Overview
FinRAG is a project that demonstrates the application of Retrieval-Augmented Generation (RAG) for analyzing financial and ESG (Environmental, Social, and Governance) documents. The project uses large language models (LLMs) to extract, analyze, and evaluate ESG metrics from sustainability reports of various companies.

## Features
- **Document Processing Pipeline**: Extract text from PDF reports while filtering out headers and footers
- **ESG Metric Extraction**: Use Google's Gemini API to extract structured ESG metrics from sustainability reports
- **ESG Scoring System**: Evaluate companies based on:
  - Emissions Performance
  - Clean Energy Commitment
  - Circularity & Material Use
  - ESG Governance & Supply Chain Engagement
- **Visualization**: Plot and compare ESG metrics across different companies

## Project Structure
```
FinRAG/
├── data/
│   ├── processed/       # Processed and structured ESG data in JSON format
│   ├── raw/             # Raw sustainability reports in PDF format
├── docs/                # Documentation files
├── examples/            # Example use cases
├── notebooks/           # Jupyter notebooks for exploration and demonstration
│   ├── ESG_Gemini.ipynb # ESG metric extraction using Gemini
│   ├── RAG_Pipeline.ipynb # General RAG pipeline implementation
├── src/                 # Source code
│   ├── common/          # Core pipeline implementations
│   │   ├── gemini_pipeline.py # Gemini-based processing pipeline
│   │   ├── rag_pipeline.py    # General RAG pipeline implementation
│   ├── utils/           # Utility functions
```

## Getting Started

### Prerequisites
- Python 3.12+
- Required packages listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/yourusername/FinRAG.git
cd FinRAG
pip install -r requirements.txt
```

### Usage
1. Place ESG reports in PDF format in `data/raw/ESG`
2. Set up your API key for Gemini:
```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```
3. Run the extraction pipeline:
```python
from src.common.gemini_pipeline import GeminiPipeline

pipeline = GeminiPipeline(
    prompt_template=prompt, 
    source_path='data/raw/ESG', 
    target_path='data/processed/ESG'
)
pipeline.run()
```

## Examples
See the `notebooks` directory for detailed examples of:
- ESG metric extraction from sustainability reports
- RAG implementation for question answering on financial documents

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.