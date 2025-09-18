from typing import Optional, Dict, Any
from embeddings.embedding_service import EmbeddingService
from loguru import logger

class ModelManager:
    """Manage embedding model lifecycle and caching"""
    
    _instance: Optional['ModelManager'] = None
    _embedding_service: Optional[EmbeddingService] = None
    
    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create embedding service"""
        if self._embedding_service is None:
            logger.info("Initializing embedding service...")
            self._embedding_service = EmbeddingService()
        return self._embedding_service
    
    def reload_model(self, model_name: str = None):
        """Reload the embedding model"""
        logger.info("Reloading embedding model...")
        self._embedding_service = EmbeddingService(model_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self._embedding_service is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self._embedding_service.model_name,
            "embedding_dim": self._embedding_service.embedding_dim
        }