from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    include_sources: Optional[bool] = Field(True, description="Whether to include source information")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    overwrite: Optional[bool] = Field(False, description="Whether to overwrite existing data")

class IngestResponse(BaseModel):
    success: bool
    message: str
    processed_files: int
    total_chunks: int

class HealthResponse(BaseModel):
    status: str
    version: str
    model_info: Dict[str, Any]
    vector_store_stats: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(10, description="Number of results to return")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int