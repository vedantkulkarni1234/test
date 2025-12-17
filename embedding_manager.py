"""
Local embedding generation and management module.
Implements sentence-transformers for embedding generation and local storage.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


class EmbeddingStorageError(Exception):
    """Exception raised when embedding storage operations fail."""
    pass


class EmbeddingGenerator:
    """Local embedding generation using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        
        # Lazy import to avoid heavy startup
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded sentence-transformers model: {self.model_name}")
            except ImportError:
                raise EmbeddingStorageError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)  # Default dimension
        
        self._load_model()
        try:
            embedding = self._model.encode(text.strip(), convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []
        
        self._load_model()
        
        try:
            embeddings = self._model.encode(
                texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [np.zeros(384, dtype=np.float32) for _ in texts]


class LocalVectorIndex:
    """Local vector storage and retrieval system."""
    
    def __init__(self, index_path: Path, embedding_dimension: int = 384):
        """
        Initialize local vector index.
        
        Args:
            index_path: Path to store the vector index
            embedding_dimension: Dimension of embeddings
        """
        self.index_path = index_path
        self.embedding_dimension = embedding_dimension
        self.index_lock = threading.RLock()
        
        # Initialize SQLite database for metadata
        self.db_path = index_path.with_suffix('.db')
        self._init_database()
        
        # In-memory cache
        self._embeddings_cache = {}
        self._metadata_cache = {}
        self._usage_stats = {}
        
        # Load existing index
        self._load_index()
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    relevance_scores TEXT,
                    semantic_tags TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    source_pdf TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON document_metadata(file_path)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count ON document_metadata(access_count DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_source ON image_metadata(source_pdf)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_page ON image_metadata(page_number)
            """)
            
            conn.commit()
    
    def _load_index(self):
        """Load existing vector index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    index_data = pickle.load(f)
                    self._embeddings_cache = index_data.get('embeddings', {})
                    self._metadata_cache = index_data.get('metadata', {})
                    self._usage_stats = index_data.get('usage_stats', {})
                logger.info(f"Loaded vector index with {len(self._embeddings_cache)} embeddings")
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")
                self._clear_index()
        else:
            self._clear_index()
    
    def _save_index(self):
        """Save vector index to disk."""
        try:
            with self.index_lock:
                index_data = {
                    'embeddings': self._embeddings_cache,
                    'metadata': self._metadata_cache,
                    'usage_stats': self._usage_stats,
                    'embedding_dimension': self.embedding_dimension
                }
                
                with open(self.index_path, 'wb') as f:
                    pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.debug(f"Saved vector index with {len(self._embeddings_cache)} embeddings")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
    
    def _clear_index(self):
        """Clear the vector index."""
        self._embeddings_cache = {}
        self._metadata_cache = {}
        self._usage_stats = {}
        logger.info("Initialized empty vector index")
    
    def add_document_chunks(self, document_data: Dict[str, Any]) -> List[str]:
        """
        Add document chunks to the vector index.
        
        Args:
            document_data: Dictionary containing document information and chunks
            
        Returns:
            List of chunk IDs added
        """
        chunk_ids = []
        file_path = document_data['file_path']
        file_hash = document_data['file_hash']
        
        with self.index_lock:
            for chunk in document_data['chunks']:
                # Create unique chunk ID
                chunk_id = f"{file_hash}_{chunk['id']}"
                chunk_ids.append(chunk_id)
                
                # Store chunk information
                self._metadata_cache[chunk_id] = {
                    'file_path': file_path,
                    'filename': document_data['filename'],
                    'chunk_text': chunk['text'],
                    'chunk_index': chunk['id'],
                    'created_at': document_data['extracted_at'],
                    'file_hash': file_hash,
                    'sections': document_data.get('sections', {}),
                    'metadata': document_data.get('metadata', {})
                }
                
                # Initialize usage statistics
                if chunk_id not in self._usage_stats:
                    self._usage_stats[chunk_id] = {
                        'access_count': 0,
                        'last_accessed': None,
                        'relevance_scores': [],
                        'semantic_tags': set()
                    }
        
        # Update database
        self._update_database_metadata(document_data, chunk_ids)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(chunk_ids)} chunks from {document_data['filename']}")
        return chunk_ids
    
    def _update_database_metadata(self, document_data: Dict[str, Any], chunk_ids: List[str]):
        """Update SQLite database with metadata."""
        with sqlite3.connect(self.db_path) as conn:
            for chunk_id, chunk in zip(chunk_ids, document_data['chunks']):
                conn.execute("""
                    INSERT OR REPLACE INTO document_metadata 
                    (chunk_id, file_path, chunk_index, text_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    document_data['file_path'],
                    chunk['id'],
                    hashlib.sha256(chunk['text'].encode()).hexdigest(),
                    document_data['extracted_at']
                ))
            
            conn.commit()
    
    def store_embeddings(self, chunk_ids: List[str], embeddings: List[np.ndarray]):
        """
        Store embeddings for chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding arrays
        """
        with self.index_lock:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                self._embeddings_cache[chunk_id] = embedding
        
        logger.debug(f"Stored {len(embeddings)} embeddings")
    
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Embedding array or None if not found
        """
        with self.index_lock:
            embedding = self._embeddings_cache.get(chunk_id)
            
            if embedding is not None:
                # Update usage statistics
                self._update_usage_stats(chunk_id)
            
            return embedding
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Metadata dictionary or None if not found
        """
        with self.index_lock:
            return self._metadata_cache.get(chunk_id)
    
    def _update_usage_stats(self, chunk_id: str):
        """Update usage statistics for a chunk."""
        if chunk_id in self._usage_stats:
            self._usage_stats[chunk_id]['access_count'] += 1
            self._usage_stats[chunk_id]['last_accessed'] = datetime.now().isoformat()
    
    def update_relevance_score(self, chunk_id: str, score: float, tags: List[str] = None):
        """
        Update relevance score and tags for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            score: Relevance score (0-1)
            tags: Optional semantic tags
        """
        with self.index_lock:
            if chunk_id in self._usage_stats:
                self._usage_stats[chunk_id]['relevance_scores'].append(score)
                
                if tags:
                    self._usage_stats[chunk_id]['semantic_tags'].update(tags)
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE document_metadata 
                        SET access_count = ?, last_accessed = ?, relevance_scores = ?
                        WHERE chunk_id = ?
                    """, (
                        self._usage_stats[chunk_id]['access_count'],
                        self._usage_stats[chunk_id]['last_accessed'],
                        json.dumps(self._usage_stats[chunk_id]['relevance_scores']),
                        chunk_id
                    ))
                    conn.commit()
    
    def get_usage_stats(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently accessed chunks.
        
        Args:
            top_k: Number of top chunks to return
            
        Returns:
            List of usage statistics
        """
        with self.index_lock:
            sorted_chunks = sorted(
                self._usage_stats.items(),
                key=lambda x: x[1]['access_count'],
                reverse=True
            )[:top_k]
            
            return [
                {
                    'chunk_id': chunk_id,
                    'access_count': stats['access_count'],
                    'last_accessed': stats['last_accessed'],
                    'avg_relevance': np.mean(stats['relevance_scores']) if stats['relevance_scores'] else 0.0,
                    'semantic_tags': list(stats['semantic_tags'])
                }
                for chunk_id, stats in sorted_chunks
            ]
    
    def store_images(self, document_data: Dict[str, Any]) -> List[str]:
        """
        Store image metadata in the database.
        
        Args:
            document_data: Document information containing images
            
        Returns:
            List of stored image IDs
        """
        if 'images' not in document_data or not document_data['images']:
            return []
        
        image_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for image in document_data['images']:
                image_id = image['id']
                image_ids.append(image_id)
                
                conn.execute("""
                    INSERT OR REPLACE INTO image_metadata 
                    (image_id, file_path, source_pdf, page_number, image_path, 
                     width, height, format, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id,
                    image['file_path'],
                    document_data['file_path'],
                    image['page_number'],
                    image['file_path'],
                    image['width'],
                    image['height'],
                    image['format'],
                    image.get('description', ''),
                    image['extracted_at']
                ))
            
            conn.commit()
        
        logger.info(f"Stored metadata for {len(image_ids)} images")
        return image_ids
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image metadata.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Image metadata dictionary or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM image_metadata WHERE image_id = ?
            """, (image_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        
        return None
    
    def search_images_by_source(self, source_pdf: str) -> List[Dict[str, Any]]:
        """
        Retrieve all images from a specific PDF.
        
        Args:
            source_pdf: Path to the source PDF
            
        Returns:
            List of image metadata dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM image_metadata WHERE source_pdf = ?
                ORDER BY page_number, image_id
            """, (source_pdf,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def update_image_description(self, image_id: str, description: str):
        """
        Update the description of an image.
        
        Args:
            image_id: ID of the image
            description: Generated description text
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE image_metadata 
                SET description = ?
                WHERE image_id = ?
            """, (description, image_id))
            conn.commit()
        
        logger.debug(f"Updated description for image {image_id}")


class EmbeddingManager:
    """High-level embedding management with local storage."""
    
    def __init__(self, cache_path: Path, embedding_dimension: int = 384):
        """
        Initialize embedding manager.
        
        Args:
            cache_path: Path to store embeddings cache
            embedding_dimension: Dimension of embeddings
        """
        self.cache_path = cache_path
        self.cache_path.mkdir(exist_ok=True)
        
        self.generator = EmbeddingGenerator()
        self.vector_index = LocalVectorIndex(cache_path / "vector_index.pkl", embedding_dimension)
        
        # Processing queue for batch operations
        self._processing_queue = []
        self._queue_lock = threading.Lock()
        
        logger.info("Initialized EmbeddingManager")
    
    def process_document(self, document_data: Dict[str, Any], gemini_client=None) -> Tuple[List[str], List[str]]:
        """
        Process a document and generate embeddings for text and images.
        
        Args:
            document_data: Document information with chunks and images
            gemini_client: Optional Gemini client for image description
            
        Returns:
            Tuple of (chunk_ids, image_ids)
        """
        logger.info(f"Processing document: {document_data['filename']}")
        
        # Add document chunks to index
        chunk_ids = self.vector_index.add_document_chunks(document_data)
        
        # Generate embeddings for all chunks
        if document_data['chunks']:
            chunk_texts = [chunk['text'] for chunk in document_data['chunks']]
            embeddings = self.generator.generate_embeddings_batch(chunk_texts)
            
            # Store embeddings
            self.vector_index.store_embeddings(chunk_ids, embeddings)
            
            logger.info(f"Generated embeddings for {len(chunk_texts)} chunks")
        
        # Process images if available
        image_ids = []
        if 'images' in document_data and document_data['images']:
            image_ids = self.process_images(document_data, gemini_client)
        
        return chunk_ids, image_ids
    
    def process_images(self, document_data: Dict[str, Any], gemini_client=None) -> List[str]:
        """
        Process images from a document, optionally generating descriptions.
        
        Args:
            document_data: Document information containing images
            gemini_client: Optional Gemini client for image description
            
        Returns:
            List of processed image IDs
        """
        if 'images' not in document_data or not document_data['images']:
            return []
        
        logger.info(f"Processing {len(document_data['images'])} images from {document_data['filename']}")
        
        # Generate descriptions for images if Gemini client is provided
        if gemini_client:
            for image in document_data['images']:
                try:
                    # Determine MIME type from format
                    mime_type = f"image/{image['format']}"
                    if image['format'] == 'jpg':
                        mime_type = 'image/jpeg'
                    
                    # Generate description using Gemini
                    response = gemini_client.describe_image(
                        image['base64'],
                        mime_type=mime_type
                    )
                    
                    image['description'] = response.text
                    logger.debug(f"Generated description for image {image['id']}")
                    
                    # Generate embedding for the description
                    if response.text:
                        description_embedding = self.generator.generate_embedding(response.text)
                        # Store with special image ID prefix
                        image_embedding_id = f"img_{image['id']}"
                        self.vector_index.store_embeddings([image_embedding_id], [description_embedding])
                        
                        # Update metadata to link image description to embedding
                        self.vector_index._metadata_cache[image_embedding_id] = {
                            'file_path': document_data['file_path'],
                            'filename': document_data['filename'],
                            'chunk_text': response.text,
                            'chunk_index': -1,  # Special index for images
                            'created_at': image['extracted_at'],
                            'file_hash': document_data['file_hash'],
                            'is_image': True,
                            'image_id': image['id'],
                            'image_path': image['file_path'],
                            'page_number': image['page_number']
                        }
                        
                except Exception as e:
                    logger.error(f"Failed to generate description for image {image['id']}: {e}")
                    image['description'] = ''
        
        # Store image metadata
        image_ids = self.vector_index.store_images(document_data)
        
        logger.info(f"Processed {len(image_ids)} images")
        return image_ids
    
    def get_document_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunk information
        """
        chunks = []
        
        with self.vector_index.index_lock:
            for chunk_id, metadata in self.vector_index._metadata_cache.items():
                if metadata['file_path'] == file_path:
                    chunk_data = metadata.copy()
                    chunk_data['chunk_id'] = chunk_id
                    
                    # Add embedding
                    embedding = self.vector_index.get_embedding(chunk_id)
                    if embedding is not None:
                        chunk_data['embedding'] = embedding
                    
                    chunks.append(chunk_data)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x['chunk_index'])
        return chunks
    
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to a query embedding using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar chunks to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        similarities = []
        
        with self.vector_index.index_lock:
            for chunk_id, embedding in self.vector_index._embeddings_cache.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                metadata = self.vector_index._metadata_cache.get(chunk_id)
                if metadata:
                    similarities.append({
                        'chunk_id': chunk_id,
                        'similarity': float(similarity),
                        'chunk_text': metadata['chunk_text'],
                        'file_path': metadata['file_path'],
                        'filename': metadata['filename']
                    })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def search_images(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for images based on description similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of matching images with metadata
        """
        # Generate query embedding
        query_embedding = self.generator.generate_embedding(query)
        
        # Search among image embeddings
        image_results = []
        
        with self.vector_index.index_lock:
            for chunk_id, embedding in self.vector_index._embeddings_cache.items():
                # Only process image embeddings
                if not chunk_id.startswith('img_'):
                    continue
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                metadata = self.vector_index._metadata_cache.get(chunk_id)
                if metadata and metadata.get('is_image'):
                    # Get full image metadata from database
                    image_id = metadata['image_id']
                    image_data = self.vector_index.get_image_metadata(image_id)
                    
                    if image_data:
                        image_results.append({
                            'image_id': image_id,
                            'similarity': float(similarity),
                            'description': metadata['chunk_text'],
                            'filename': metadata['filename'],
                            'file_path': metadata['file_path'],
                            'image_path': metadata['image_path'],
                            'page_number': metadata['page_number'],
                            'width': image_data.get('width'),
                            'height': image_data.get('height'),
                            'format': image_data.get('format')
                        })
        
        # Sort by similarity and return top_k
        image_results.sort(key=lambda x: x['similarity'], reverse=True)
        return image_results[:top_k]