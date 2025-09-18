from typing import List
from loguru import logger
from ingest.document_processor import DocumentProcessor
from ingest.chunker import TextChunker
from vectorstore.faiss_store import FAISSVectorStore
from embeddings.model_manager import ModelManager

async def process_documents_task(
    file_paths: List[str],
    vector_store: FAISSVectorStore,
    model_manager: ModelManager
):
    """Background task to process and ingest documents"""
    try:
        logger.info(f"Starting background processing of {len(file_paths)} files")
        
        # Initialize processors
        doc_processor = DocumentProcessor()
        chunker = TextChunker()
        
        # Load documents
        documents = doc_processor.load_documents(file_paths)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        embedding_service = model_manager.embedding_service
        enriched_chunks = embedding_service.encode_chunks(chunks)
        logger.info("Generated embeddings for all chunks")
        
        # Add to vector store
        vector_store.add_documents(enriched_chunks)
        
        # Save vector store
        vector_store.save()
        
        logger.info(f"Successfully processed {len(file_paths)} files, {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in background document processing: {e}")
        raise

def process_single_document_task(
    file_path: str,
    vector_store: FAISSVectorStore,
    model_manager: ModelManager
):
    """Process a single document"""
    return process_documents_task([file_path], vector_store, model_manager)