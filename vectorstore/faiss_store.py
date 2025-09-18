import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from config import settings

class FAISSVectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_dim: int, store_path: str = None):
        self.embedding_dim = embedding_dim
        self.store_path = Path(store_path or settings.VECTOR_STORE_PATH)
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Create store directory
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Using IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract embeddings and metadata
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        for chunk in chunks:
            # Store document content and metadata separately
            doc_data = {
                'content': chunk['content'],
                'source': chunk['source']
            }
            metadata = chunk['metadata']
            
            self.documents.append(doc_data)
            self.metadata.append(metadata)
        
        logger.info(f"Successfully added {len(chunks)} chunks. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return [], []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than available
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        similarities = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                result = {
                    'content': self.documents[idx]['content'],
                    'source': self.documents[idx]['source'],
                    'metadata': self.metadata[idx],
                    'similarity': float(score)
                }
                results.append(result)
                similarities.append(float(score))
        
        return similarities, results
    
    def save(self):
        """Save the vector store to disk"""
        try:
            # Save FAISS index
            index_path = self.store_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents and metadata
            docs_path = self.store_path / "documents.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            metadata_path = self.store_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save store info
            info_path = self.store_path / "store_info.json"
            info = {
                'embedding_dim': self.embedding_dim,
                'total_documents': len(self.documents),
                'index_type': 'IndexFlatIP'
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Vector store saved to {self.store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self) -> bool:
        """Load the vector store from disk"""
        try:
            index_path = self.store_path / "faiss_index.bin"
            docs_path = self.store_path / "documents.pkl"
            metadata_path = self.store_path / "metadata.json"
            info_path = self.store_path / "store_info.json"
            
            # Check if all required files exist
            if not all(p.exists() for p in [index_path, docs_path, metadata_path, info_path]):
                logger.info("Vector store files not found, starting fresh")
                return False
            
            # Load store info
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            if info['embedding_dim'] != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {info['embedding_dim']}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Vector store loaded: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def clear(self):
        """Clear the vector store"""
        self._initialize_index()
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0,
            'is_trained': self.index.is_trained if self.index else False
        }