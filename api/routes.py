from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import os
from loguru import logger

from api.models import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, SearchRequest, SearchResponse
)
from ingest.document_processor import DocumentProcessor
from ingest.chunker import TextChunker
from embeddings.model_manager import ModelManager
from vectorstore.faiss_store import FAISSVectorStore
from vectorstore.retriever import TwoStageRetriever
from workers.background_tasks import process_documents_task
from config import settings

import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

router = APIRouter()

# Global instances
model_manager = ModelManager()
vector_store = None
retriever = None

def initialize_vector_store():
    """Initialize vector store and retriever"""
    global vector_store, retriever
    
    if vector_store is None:
        embedding_dim = model_manager.embedding_service.get_embedding_dim()
        vector_store = FAISSVectorStore(embedding_dim)
        vector_store.load()
        retriever = TwoStageRetriever(vector_store)

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest new documents into the system"""
    try:
        initialize_vector_store()

        # Process documents in background
        background_tasks.add_task(process_documents_task, request.file_paths, model_manager, vector_store)

        return IngestResponse(
            message="Ingestion started",
            total_files=len(request.file_paths),
            overwrite=request.overwrite
        )

    except Exception as e:
        logger.error(f"Error in ingest_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""
    try:
        initialize_vector_store()
        
        if vector_store.get_stats()['total_documents'] == 0:
            raise HTTPException(status_code=400, detail="No documents in the system. Please ingest documents first.")
        
        # Retrieve relevant documents
        results = retriever.retrieve(request.query, rerank_top_k=request.top_k)
        
        if not results:
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                metadata={"total_sources": 0, "retrieval_method": "two_stage"}
            )
        
        # Generate answer using Gemini
        answer = await generate_answer(request.query, results)
        
        # Prepare sources
        sources = []
        if request.include_sources:
            for result in results:
                source = {
                    "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                    "source": result["source"],
                    "similarity": result.get("final_score", 0.0),
                    "metadata": {
                        "chunk_id": result["metadata"].get("chunk_id"),
                        "title": result["metadata"].get("title")
                    }
                }
                sources.append(source)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metadata={
                "total_sources": len(results),
                "retrieval_method": "two_stage",
                "model": settings.LLM_MODEL
            }
        )
        
    except Exception as e:
        logger.error(f"Error in query_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))