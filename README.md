### Installation and Setup
```bash
# 1. Clone/create the project directory
git clone https://github.com/pratiksonune/QA-RAG-LLM.git

# 2. Create virtual environment
uv venv ThoR --python 3.10
source ThoR/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your Gemini API key

# 5. Create necessary directories
mkdir -p data/vector_store logs documents

# 6. Run the application
python api/main.py
```

### Usage Examples

#### 1. Ingest Documents
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["documents/doc1.md", "documents/doc2.md"],
    "overwrite": false
  }'
```

#### 2. Query the System
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## Key Features:

1. **Document Ingestion**: Processes markdown files with metadata extraction
2. **Smart Chunking**: Semantic chunking by sections with fallback to sliding window
3. **Embeddings**: Uses Qwen3-Embedding-0.6B model via sentence-transformers
4. **Vector Store**: FAISS-based vector database with persistence
5. **Two-Stage Retrieval**: ANN search followed by TF-IDF reranking
6. **LLM Integration**: Gemini 2.5 Flash for answer generation
7. **FastAPI Service**: RESTful API with validation and error handling
8. **Background Processing**: Async document processing
9. **Monitoring**: Prometheus metrics and health checks
10. **Testing**: Comprehensive test suite
