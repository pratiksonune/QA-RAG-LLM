from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import re

class DocumentProcessor:
    """Process and load markdown documents"""
    
    def __init__(self):
        self.supported_extensions = {'.md', '.markdown'}
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents successfully")
        return documents
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from markdown
        metadata = self._extract_metadata(content, path)
        
        return {
            'content': content,
            'metadata': metadata,
            'source': str(path)
        }
    
    def _extract_metadata(self, content: str, path: Path) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {
            'filename': path.name,
            'file_path': str(path),
            'file_size': len(content),
        }
        
        # Extract title from markdown (first # heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        else:
            metadata['title'] = path.stem
        
        # Count headings for structure analysis
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        metadata['heading_count'] = len(headings)
        metadata['sections'] = headings[:10]  # First 10 headings
        
        return metadata