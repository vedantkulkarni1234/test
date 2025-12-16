"""
Configuration management for the Agentic RAG System.
Handles API keys, file paths, search parameters, and system settings.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Centralized configuration management."""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    
    # File System Paths
    PDF_MONITOR_PATH: Path = Path(os.getenv("PDF_MONITOR_PATH", "./knowledge_hub"))
    EMBEDDING_CACHE_PATH: Path = Path("./embeddings_cache")
    VECTOR_INDEX_PATH: Path = Path("./vector_index.json")
    SYSTEM_DB_PATH: Path = Path("./rag_system.db")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384
    
    # A* Search Configuration
    ASTAR_HEURISTIC_WEIGHT: float = 1.0  # Weight for heuristic in A* search
    MAX_SEARCH_DEPTH: int = 10  # Maximum search depth in A* algorithm
    RELEVANCE_THRESHOLD: float = 0.7  # Minimum relevance score for document inclusion
    
    # RAG Pipeline Configuration
    TOP_K_DOCUMENTS: int = 5  # Number of top documents to return
    CHUNK_SIZE: int = 512  # Text chunk size for embedding
    CHUNK_OVERLAP: int = 50  # Overlap between chunks
    CONTEXT_WINDOW_SIZE: int = 8000  # Maximum context tokens for Gemini
    
    # Agent Configuration
    MAX_AGENT_STEPS: int = 10  # Maximum steps in agent execution
    AGENT_TIMEOUT: int = 300  # Timeout in seconds for agent tasks
    CONCURRENCY_LIMIT: int = 4  # Maximum concurrent operations
    
    # Monitoring & Logging
    MONITOR_INTERVAL: float = 5.0  # PDF monitoring interval in seconds
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_METRICS: bool = True
    
    # Performance Thresholds
    MAX_EMBEDDING_BATCH_SIZE: int = 32
    MAX_MEMORY_USAGE_MB: int = 2048  # Maximum memory usage in MB
    CACHE_TTL_HOURS: int = 24  # Cache time-to-live in hours
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY environment variable is required")
        
        if not cls.PDF_MONITOR_PATH.exists():
            try:
                cls.PDF_MONITOR_PATH.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create PDF monitor path: {e}")
        
        # Create necessary directories
        cls.EMBEDDING_CACHE_PATH.mkdir(exist_ok=True)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }
    
    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini API configuration."""
        return {
            "api_key": cls.GEMINI_API_KEY,
            "base_url": cls.GEMINI_BASE_URL,
            "model": "gemini-2.5-flash",
            "max_tokens": cls.CONTEXT_WINDOW_SIZE,
            "temperature": 0.1,  # Lower temperature for more deterministic results
            "top_p": 0.95,
            "top_k": 40
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get A* search configuration."""
        return {
            "heuristic_weight": cls.ASTAR_HEURISTIC_WEIGHT,
            "max_depth": cls.MAX_SEARCH_DEPTH,
            "relevance_threshold": cls.RELEVANCE_THRESHOLD,
            "top_k": cls.TOP_K_DOCUMENTS
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding model configuration."""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "batch_size": cls.MAX_EMBEDDING_BATCH_SIZE
        }


# Global configuration instance
config = Config()