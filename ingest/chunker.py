from typing import List, Dict, Any
import re
from config import settings

class TextChunker:
    """Chunk text into smaller pieces for embedding"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document"""
        content = document['content']
        metadata = document['metadata']
        source = document['source']
        
        # Try semantic chunking first (by sections)
        semantic_chunks = self._semantic_chunk(content)
        
        # If semantic chunking produces chunks that are too large, split further
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large semantic chunks
                sub_chunks = self._sliding_window_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(final_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': f"{metadata['filename']}_chunk_{i}",
                'chunk_index': i,
                'chunk_size': len(chunk_text),
                'total_chunks': len(final_chunks)
            })
            
            chunks.append({
                'content': chunk_text.strip(),
                'metadata': chunk_metadata,
                'source': source
            })
        
        return chunks
    
    def _semantic_chunk(self, content: str) -> List[str]:
        """Chunk by markdown sections"""
        # Split by markdown headers
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        # Clean and filter empty sections
        chunks = []
        for section in sections:
            section = section.strip()
            if section and len(section) > 50:  # Minimum chunk size
                chunks.append(section)
        
        # If no clear sections, fall back to paragraph splitting
        if len(chunks) <= 1:
            chunks = self._paragraph_chunk(content)
        
        return chunks
    
    def _paragraph_chunk(self, content: str) -> List[str]:
        """Chunk by paragraphs"""
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, start new chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """Sliding window chunking for large texts"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence ending in the last 100 characters
                sentence_end = text.rfind('.', end - 100, end)
                if sentence_end != -1 and sentence_end > start + 100:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks