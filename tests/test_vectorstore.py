import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from vectorstore.faiss_store import FAISSVectorStore
from vectorstore.retriever import TwoStageRetriever

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def vector_store(temp_dir):
    """Create vector store for testing"""
    return FAISSVectorStore(embedding_dim=384, store_path=temp_dir)

@pytest.fixture
def sample_chunks():
    """Sample document chunks with embeddings"""
    return [
        {
            'content': 'Machine learning is a subset of artificial intelligence.',
            'embedding': np.random.rand(384).astype(np.float32),
            'metadata': {'chunk_id': 'doc1_chunk1', 'title': 'ML Basics'},
            'source': 'doc1.md'
        },
        {
            'content': 'Deep learning uses neural networks with multiple layers.',
            'embedding': np.random.rand(384).astype(np.float32),
            'metadata': {'chunk_id': 'doc1_chunk2', 'title': 'ML Basics'},
            'source': 'doc1.md'
        },
        {
            'content': 'Natural language processing enables computers to understand text.',
            'embedding': np.random.rand(384).astype(np.float32),
            'metadata': {'chunk_id': 'doc2_chunk1', 'title': 'NLP Guide'},
            'source': 'doc2.md'
        }
    ]

def test_vector_store_initialization(vector_store):
    """Test vector store initialization"""
    assert vector_store.embedding_dim == 384
    assert vector_store.index is not None
    assert len(vector_store.documents) == 0

def test_add_documents(vector_store, sample_chunks):
    """Test adding documents to vector store"""
    vector_store.add_documents(sample_chunks)
    
    assert len(vector_store.documents) == 3
    assert len(vector_store.metadata) == 3
    assert vector_store.index.ntotal == 3

def test_search(vector_store, sample_chunks):
    """Test vector store search"""
    vector_store.add_documents(sample_chunks)
    
    # Create query embedding
    query_embedding = np.random.rand(384).astype(np.float32)
    
    similarities, results = vector_store.search(query_embedding, k=2)
    
    assert len(similarities) == 2
    assert len(results) == 2
    assert all('content' in result for result in results)
    assert all('similarity' in result for result in results)

def test_save_and_load(vector_store, sample_chunks, temp_dir):
    """Test saving and loading vector store"""
    # Add documents and save
    vector_store.add_documents(sample_chunks)
    vector_store.save()
    
    # Create new vector store and load
    new_vector_store = FAISSVectorStore(embedding_dim=384, store_path=temp_dir)
    success = new_vector_store.load()
    
    assert success
    assert len(new_vector_store.documents) == 3
    assert new_vector_store.index.ntotal == 3

def test_clear(vector_store, sample_chunks):
    """Test clearing vector store"""
    vector_store.add_documents(sample_chunks)
    assert len(vector_store.documents) == 3
    
    vector_store.clear()
    assert len(vector_store.documents) == 0
    assert vector_store.index.ntotal == 0

def test_get_stats(vector_store, sample_chunks):
    """Test getting vector store statistics"""
    stats = vector_store.get_stats()
    assert stats['total_documents'] == 0
    
    vector_store.add_documents(sample_chunks)
    stats = vector_store.get_stats()
    assert stats['total_documents'] == 3
    assert stats['embedding_dim'] == 384

def test_two_stage_retriever(vector_store, sample_chunks):
    """Test two-stage retrieval"""
    vector_store.add_documents(sample_chunks)
    retriever = TwoStageRetriever(vector_store)
    
    # Test retrieval (note: this will work but similarity scores may be random due to random embeddings)
    results = retriever.retrieve("machine learning", top_k=5, rerank_top_k=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2  # Should return at most rerank_top_k results
    
    if results:  # If any results were returned above threshold
        for result in results:
            assert 'content' in result
            assert 'final_score' in result
            assert 'ann_score' in result