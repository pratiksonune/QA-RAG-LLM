from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from loguru import logger
from config import settings
import torch

class EmbeddingService:
    """Service for generating embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            sample_embedding = self.model.encode(["test"])
            self.embedding_dim = len(sample_embedding[0])
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into embeddings"""
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            
            # Encode with batching for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encode document chunks and add embeddings to metadata"""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.encode_texts(texts)
        
        # Add embeddings to chunks
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embeddings[i]
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings"""
        return self.model.similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(document_embeddings)
        ).numpy().flatten()
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim