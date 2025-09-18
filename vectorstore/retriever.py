from typing import List, Dict, Any, Tuple
from vectorstore.faiss_store import FAISSVectorStore
from embeddings.model_manager import ModelManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
from config import settings

class TwoStageRetriever:
    """Two-stage retrieval: ANN search + reranking"""
    
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.model_manager = ModelManager()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._build_tfidf_index()
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for reranking"""
        if not self.vector_store.documents:
            return
        
        logger.info("Building TF-IDF index for reranking...")
        
        # Extract all document contents
        contents = [doc['content'] for doc in self.vector_store.documents]
        
        # Build TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
        logger.info(f"TF-IDF index built with shape: {self.tfidf_matrix.shape}")
    
    def retrieve(self, query: str, top_k: int = None, rerank_top_k: int = None) -> List[Dict[str, Any]]:
        """Perform two-stage retrieval"""
        top_k = top_k or settings.TOP_K_RETRIEVAL
        rerank_top_k = rerank_top_k or settings.TOP_K_RERANK
        
        # Stage 1: ANN retrieval
        ann_results = self._ann_retrieve(query, top_k)
        
        if not ann_results:
            return []
        
        # Stage 2: Reranking
        reranked_results = self._rerank(query, ann_results, rerank_top_k)
        
        return reranked_results
    
    def _ann_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Stage 1: Approximate Nearest Neighbor retrieval"""
        # Get query embedding
        embedding_service = self.model_manager.embedding_service
        query_embedding = embedding_service.encode_texts([query])[0]
        
        # Search vector store
        similarities, results = self.vector_store.search(query_embedding, top_k)
        
        # Filter by similarity threshold
        filtered_results = []
        for result, similarity in zip(results, similarities):
            if similarity >= settings.SIMILARITY_THRESHOLD:
                result['ann_score'] = similarity
                filtered_results.append(result)
        
        logger.info(f"ANN retrieval: {len(filtered_results)}/{len(results)} results above threshold")
        return filtered_results
    
    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Stage 2: Rerank candidates using TF-IDF similarity"""
        if not candidates or self.tfidf_vectorizer is None:
            return candidates[:top_k]
        
        # Get TF-IDF representation of query
        query_tfidf = self.tfidf_vectorizer.transform([query])
        
        # Find indices of candidates in the original document list
        candidate_indices = []
        for candidate in candidates:
            # Find the index based on content matching
            for i, doc in enumerate(self.vector_store.documents):
                if doc['content'] == candidate['content']:
                    candidate_indices.append(i)
                    break
        
        if not candidate_indices:
            return candidates[:top_k]
        
        # Get TF-IDF vectors for candidates
        candidate_tfidf = self.tfidf_matrix[candidate_indices]
        
        # Compute TF-IDF similarities
        tfidf_similarities = cosine_similarity(query_tfidf, candidate_tfidf).flatten()
        
        # Combine ANN and TF-IDF scores
        for i, candidate in enumerate(candidates):
            ann_score = candidate.get('ann_score', 0.0)
            tfidf_score = tfidf_similarities[i] if i < len(tfidf_similarities) else 0.0
            
            # Weighted combination (you can tune these weights)
            combined_score = 0.7 * ann_score + 0.3 * tfidf_score
            
            candidate['tfidf_score'] = float(tfidf_score)
            candidate['combined_score'] = float(combined_score)
            candidate['final_score'] = float(combined_score)
        
        # Sort by combined score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Reranking: returning top {min(top_k, len(reranked))} results")
        return reranked[:top_k]
    
    def update_tfidf_index(self):
        """Update TF-IDF index after adding new documents"""
        self._build_tfidf_index()