import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Model Configuration
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    LLM_MODEL: str = "gemini-2.5-flash"
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = "data/vector_store"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create necessary directories
Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)