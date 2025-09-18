import pytest
import numpy as np
from embeddings.embedding_service import EmbeddingService
from embeddings.model_manager import ModelManager

@pytest.fixture
def embedding_service():
    return EmbeddingService()

@pytest.fixture
def sample_texts():
    return [
        "This is a test document about machine learning.",
        "Natural language processing is a branch of AI.",
        "Vector databases store high-dimensional embeddings."
    ]

def test_embedding_service_initialization(embedding_service):
    """Test embedding service initialization"""
    assert embedding_service.model is not None
    assert embedding_service.embedding_dim > 0

def test_encode_texts(embedding_service, sample_texts):
    """Test text encoding functionality"""
    embeddings = embedding_service.encode_texts(sample_texts)
    
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] == embedding_service.embedding_dim
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

def test_encode_empty_list(embedding_service):
    """Test encoding empty list"""
    embeddings = embedding_service.encode_texts([])
    assert len(embeddings) == 0

def test_compute_similarity(embedding_service, sample_texts):
    """Test similarity computation"""
    embeddings = embedding_service.encode_texts(sample_texts)
    query_embedding = embeddings[0]
    
    similarities = embedding_service.compute_similarity(query_embedding, embeddings)
    
    assert len(similarities) == len(sample_texts)
    assert similarities[0] == pytest.approx(1.0, rel=1e-5)  # Self-similarity
    assert all(0 <= sim <= 1 for sim in similarities)

def test_model_manager_singleton():
    """Test ModelManager singleton pattern"""
    manager1 = ModelManager()
    manager2 = ModelManager()
    
    assert manager1 is manager2

def test_encode_chunks(embedding_service):
    """Test chunk encoding with metadata"""
    chunks = [
        {
            'content': "Test content 1",
            'metadata': {'id': 1},
            'source': 'test1.md'
        },
        {
            'content': "Test content 2", 
            'metadata': {'id': 2},
            'source': 'test2.md'
        }
    ]
    
    enriched_chunks = embedding_service.encode_chunks(chunks)
    
    assert len(enriched_chunks) == 2
    for chunk in enriched_chunks:
        assert 'embedding' in chunk
        assert len(chunk['embedding']) == embedding_service.embedding_dim