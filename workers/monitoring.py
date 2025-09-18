from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from loguru import logger

# Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
query_count = Counter('rag_queries_total', 'Total RAG queries')
ingestion_count = Counter('rag_ingestions_total', 'Total document ingestions')
retrieval_duration = Histogram('rag_retrieval_duration_seconds', 'RAG retrieval duration')

def setup_monitoring(app: FastAPI):
    """Setup monitoring and metrics collection"""
    
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        request_count.labels(method=request.method, endpoint=request.url.path).inc()
        request_duration.observe(duration)
        
        return response
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    logger.info("Monitoring setup complete")

class MetricsCollector:
    """Utility class for collecting custom metrics"""
    
    @staticmethod
    def record_query():
        query_count.inc()
    
    @staticmethod
    def record_ingestion():
        ingestion_count.inc()
    
    @staticmethod
    def record_retrieval_time(duration: float):
        retrieval_duration.observe(duration)