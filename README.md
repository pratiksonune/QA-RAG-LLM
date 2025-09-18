### Installation and Setup
```bash
# 1. Clone/create the project directory
git clone https://github.com/pratiksonune/QA-RAG-LLM.git

# 2. Create virtual environment
uv venv ThoR --python 3.10
source venv/bin/activate

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

#### 3. Search Documents
```bash
curl "http://localhost:8000/api/v1/search?query=neural%20networks&top_k=5"
```

#### 4. Health Check
```bash
curl "http://localhost:8000/api/v1/health"
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

## Key Features Implemented

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

This implementation provides a production-ready RAG Q/A system with all the components you specified!

```python
async def generate_answer(query: str, contexts: List[dict]) -> str:
    """Generate answer using Gemini API"""
    try:
        # Prepare context from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1} (Score: {ctx.get('final_score', 0):.3f}):\n{ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Create prompt
        prompt = f"""Based on the following context documents, please answer the user's question. If the answer is not found in the context, please say so.

Context Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided. If you need to reference specific documents, mention them in your response."""

        # Call Gemini API
        model = genai.GenerativeModel(settings.LLM_MODEL)
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"I apologize, but I encountered an error while generating the answer: {str(e)}"

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest documents into the system"""
    try:
        # Validate file paths
        for file_path in request.file_paths:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        
        initialize_vector_store()
        
        if request.overwrite:
            vector_store.clear()
            logger.info("Vector store cleared for fresh ingestion")
        
        # Process documents in background
        background_tasks.add_task(
            process_documents_task,
            request.file_paths,
            vector_store,
            model_manager
        )
        
        return IngestResponse(
            success=True,
            message="Document ingestion started. Processing in background.",
            processed_files=len(request.file_paths),
            total_chunks=0  # Will be updated after processing
        )
        
    except Exception as e:
        logger.error(f"Error in ingest_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=SearchResponse)
async def search_documents(query: str, top_k: int = 10):
    """Search documents without LLM generation"""
    try:
        initialize_vector_store()
        
        if vector_store.get_stats()['total_documents'] == 0:
            raise HTTPException(status_code=400, detail="No documents in the system.")
        
        # Retrieve relevant documents
        results = retriever.retrieve(query, rerank_top_k=top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "source": result["source"],
                "similarity": result.get("final_score", 0.0),
                "metadata": result["metadata"]
            })
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in search_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        initialize_vector_store()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model_info=model_manager.get_model_info(),
            vector_store_stats=vector_store.get_stats()
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_info={"status": "error", "error": str(e)},
            vector_store_stats={"status": "error"}
        )

@router.post("/clear")
async def clear_vector_store():
    """Clear all data from vector store"""
    try:
        initialize_vector_store()
        vector_store.clear()
        
        # Rebuild retriever
        global retriever
        retriever = TwoStageRetriever(vector_store)
        
        return {"message": "Vector store cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```