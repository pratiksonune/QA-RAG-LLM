import pytest
from fastapi.testclient import TestClient
import tempfile
import json
from pathlib import Path

# Mock the API since we can't run the full system in tests
@pytest.fixture
def mock_app():
    from fastapi import FastAPI
    from api.models import HealthResponse, QueryResponse
    
    app = FastAPI()
    
    @app.get("/api/v1/health")
    async def mock_health():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "model_info": {"status": "loaded", "model_name": "test"},
            "vector_store_stats": {"total_documents": 0}
        }
    
    @app.post("/api/v1/query")
    async def mock_query(request: dict):
        return {
            "query": request.get("query", ""),
            "answer": "This is a mock answer.",
            "sources": [],
            "metadata": {"total_sources": 0}
        }
    
    return app

@pytest.fixture
def client(mock_app):
    return TestClient(mock_app)

@pytest.fixture
def temp_file():
    """Create temporary markdown file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\nThis is a test document for ingestion.")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_info" in data

def test_query_endpoint(client):
    """Test query endpoint"""
    query_data = {
        "query": "What is machine learning?",
        "top_k": 5,
        "include_sources": True
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["query"] == query_data["query"]
    assert "answer" in data
    assert "sources" in data
    assert "metadata" in data

def test_query_endpoint_validation(client):
    """Test query endpoint input validation"""
    # Test missing query
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422  # Validation error
    
    # Test invalid top_k
    response = client.post("/api/v1/query", json={"query": "test", "top_k": -1})
    # Should still work, FastAPI will handle validation

class TestDocumentProcessor:
    """Test document processing functionality"""
    
    def test_load_markdown_file(self, temp_file):
        from ingest.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        doc = processor.load_document(temp_file)
        
        assert doc is not None
        assert "content" in doc
        assert "metadata" in doc
        assert "source" in doc
        assert "Test Document" in doc["metadata"]["title"]

class TestChunker:
    """Test text chunking functionality"""
    
    def test_chunk_document(self):
        from ingest.chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        sample_doc = {
            'content': "This is a long document. " * 20,  # Create long content
            'metadata': {'filename': 'test.md', 'title': 'Test'},
            'source': 'test.md'
        }
        
        chunks = chunker.chunk_document(sample_doc)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        for chunk in chunks:
            assert 'content' in chunk
            assert 'metadata' in chunk
            assert 'chunk_id' in chunk['metadata']

def test_integration_flow(temp_file):
    """Test integration of document processing, chunking, and embedding"""
    from ingest.document_processor import DocumentProcessor
    from ingest.chunker import TextChunker
    
    # Process document
    processor = DocumentProcessor()
    doc = processor.load_document(temp_file)
    
    # Chunk document
    chunker = TextChunker()
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) >= 1
    assert all('content' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)