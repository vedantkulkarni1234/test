"""
A* Search Algorithm implementation for ultra-fast document ranking and retrieval.
Implements intelligent document navigation with relevance scoring and path optimization.
"""

import heapq
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class SearchState:
    """Represents a state in the A* search space."""
    document_set: frozenset  # Set of document IDs in current state
    relevance_score: float   # Total relevance score
    cost_so_far: float       # Actual cost to reach this state
    heuristic: float         # Heuristic estimate to goal
    path: List[str]          # Path taken to reach this state
    depth: int               # Search depth
    
    @property
    def total_cost(self) -> float:
        """Total cost including heuristic."""
        return self.cost_so_far + self.heuristic
    
    def __lt__(self, other) -> bool:
        """Compare states by total cost."""
        return self.total_cost < other.total_cost


@dataclass
class DocumentNode:
    """Represents a document in the search graph."""
    chunk_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    usage_frequency: float = 0.0
    recency_score: float = 0.0
    semantic_cohesion: float = 0.0
    neighbors: Set[str] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()


class AStarRetriever:
    """
    A* Search Algorithm implementation for document retrieval and ranking.
    
    Features:
    - Multi-criteria scoring (relevance, usage, recency, cohesion)
    - Intelligent document graph construction
    - Optimal path finding for document selection
    - Dynamic relevance estimation
    """
    
    def __init__(self, vector_index, search_config: Dict[str, Any]):
        """
        Initialize A* retriever.
        
        Args:
            vector_index: LocalVectorIndex instance
            search_config: Configuration parameters for search
        """
        self.vector_index = vector_index
        self.config = search_config
        
        # Search parameters
        self.heuristic_weight = search_config.get('heuristic_weight', 1.0)
        self.max_depth = search_config.get('max_depth', 10)
        self.relevance_threshold = search_config.get('relevance_threshold', 0.7)
        self.top_k = search_config.get('top_k', 5)
        
        # Document graph
        self.document_graph = defaultdict(set)
        self.embedding_cache = {}
        self.query_history = deque(maxlen=100)  # Track recent queries
        
        logger.info("Initialized AStarRetriever")
    
    def search(self, query: str, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform A* search to find optimal document set.
        
        Args:
            query: Text query
            query_embedding: Embedding of the query
            top_k: Number of documents to retrieve
            
        Returns:
            List of ranked documents with search metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Starting A* search for query: '{query[:50]}...'")
        
        # Get initial candidate documents using similarity search
        candidates = self._get_initial_candidates(query_embedding, top_k * 3)
        
        if not candidates:
            logger.warning("No candidate documents found")
            return []
        
        # Build document graph
        self._build_document_graph(candidates)
        
        # Perform A* search
        result = self._astar_search(candidates, query_embedding, top_k)
        
        # Update usage statistics
        self._update_usage_statistics(result)
        
        # Store query in history
        self.query_history.append({
            'query': query,
            'query_embedding': query_embedding,
            'timestamp': datetime.now(),
            'result_count': len(result)
        })
        
        logger.info(f"A* search completed. Found {len(result)} documents")
        return result
    
    def _get_initial_candidates(self, query_embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """
        Get initial candidate documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of candidates
            
        Returns:
            List of candidate documents
        """
        # Use the vector index's similarity search
        candidates = self.vector_index.search_similar_chunks(
            query_embedding, 
            top_k=limit
        )
        
        # Filter by relevance threshold
        filtered_candidates = [
            doc for doc in candidates 
            if doc['similarity'] >= self.relevance_threshold
        ]
        
        logger.debug(f"Found {len(filtered_candidates)} candidates above threshold")
        return filtered_candidates
    
    def _build_document_graph(self, candidates: List[Dict[str, Any]]):
        """
        Build document graph based on semantic similarity and co-occurrence.
        
        Args:
            candidates: List of candidate documents
        """
        self.document_graph.clear()
        
        # Build edges between documents based on semantic similarity
        for i, doc1 in enumerate(candidates):
            chunk_id1 = doc1['chunk_id']
            
            # Get embedding for document 1
            if chunk_id1 not in self.embedding_cache:
                self.embedding_cache[chunk_id1] = self.vector_index.get_embedding(chunk_id1)
            
            embedding1 = self.embedding_cache[chunk_id1]
            if embedding1 is None:
                continue
            
            for j, doc2 in enumerate(candidates[i+1:], i+1):
                chunk_id2 = doc2['chunk_id']
                
                # Get embedding for document 2
                if chunk_id2 not in self.embedding_cache:
                    self.embedding_cache[chunk_id2] = self.vector_index.get_embedding(chunk_id2)
                
                embedding2 = self.embedding_cache[chunk_id2]
                if embedding2 is None:
                    continue
                
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    embedding1.reshape(1, -1),
                    embedding2.reshape(1, -1)
                )[0][0]
                
                # Add edge if similarity is above threshold
                if similarity > 0.5:  # Semantic similarity threshold
                    self.document_graph[chunk_id1].add(chunk_id2)
                    self.document_graph[chunk_id2].add(chunk_id1)
        
        logger.debug(f"Built document graph with {len(self.document_graph)} nodes")
    
    def _astar_search(self, candidates: List[Dict[str, Any]], query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform A* search to find optimal document subset.
        
        Args:
            candidates: Initial candidate documents
            query_embedding: Query embedding
            top_k: Target number of documents
            
        Returns:
            List of optimal documents
        """
        # Priority queue for A* search
        open_set = []
        
        # Create initial search states
        for candidate in candidates:
            chunk_id = candidate['chunk_id']
            
            # Calculate initial costs
            relevance = self._calculate_relevance_score(chunk_id, query_embedding)
            heuristic = self._calculate_heuristic(chunk_id, query_embedding, top_k)
            cost = 1.0  # Base cost for including this document
            
            state = SearchState(
                document_set=frozenset([chunk_id]),
                relevance_score=relevance,
                cost_so_far=cost,
                heuristic=heuristic,
                path=[chunk_id],
                depth=0
            )
            
            heapq.heappush(open_set, state)
        
        # Track visited states to avoid cycles
        visited = set()
        
        while open_set and len(visited) < 1000:  # Safety limit
            current_state = heapq.heappop(open_set)
            
            # Check if we've reached our goal
            if len(current_state.document_set) >= top_k:
                return self._format_search_results(current_state.document_set, query_embedding)
            
            # Create state key for visited tracking
            state_key = tuple(sorted(current_state.document_set))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Generate successor states
            successors = self._generate_successors(current_state, candidates, query_embedding, top_k)
            
            for successor in successors:
                successor_key = tuple(sorted(successor.document_set))
                if successor_key not in visited:
                    heapq.heappush(open_set, successor)
        
        # If we can't reach the goal, return the best found
        if open_set:
            best_state = min(open_set, key=lambda s: s.total_cost)
            return self._format_search_results(best_state.document_set, query_embedding)
        else:
            logger.warning("A* search failed to find solution")
            return self._format_search_results(set(), query_embedding)
    
    def _generate_successors(self, current_state: SearchState, candidates: List[Dict[str, Any]], 
                           query_embedding: np.ndarray, top_k: int) -> List[SearchState]:
        """
        Generate successor states by adding new documents.
        
        Args:
            current_state: Current search state
            candidates: Available candidate documents
            query_embedding: Query embedding
            top_k: Target number of documents
            
        Returns:
            List of successor states
        """
        successors = []
        
        # Don't exceed target size
        if len(current_state.document_set) >= top_k:
            return successors
        
        # Find documents not yet in the set
        current_docs = set(current_state.document_set)
        available_docs = [doc for doc in candidates if doc['chunk_id'] not in current_docs]
        
        # Consider adding documents based on relevance and graph connectivity
        for doc in available_docs:
            chunk_id = doc['chunk_id']
            
            # Calculate new costs
            additional_relevance = self._calculate_relevance_score(chunk_id, query_embedding)
            
            # Factor in graph connectivity (semantic cohesion)
            cohesion_bonus = self._calculate_cohesion_bonus(current_docs, chunk_id)
            
            # Update costs
            new_relevance = current_state.relevance_score + additional_relevance + cohesion_bonus
            additional_cost = 1.0 + (len(current_docs) * 0.1)  # Increasing cost for larger sets
            
            # Calculate heuristic for new state
            remaining_needed = top_k - len(current_docs) - 1
            new_heuristic = self._calculate_heuristic(chunk_id, query_embedding, remaining_needed)
            
            # Create new state
            new_state = SearchState(
                document_set=frozenset(current_docs | {chunk_id}),
                relevance_score=new_relevance,
                cost_so_far=current_state.cost_so_far + additional_cost,
                heuristic=new_heuristic,
                path=current_state.path + [chunk_id],
                depth=current_state.depth + 1
            )
            
            successors.append(new_state)
        
        # Limit number of successors to avoid explosion
        successors.sort(key=lambda s: s.total_cost, reverse=True)
        return successors[:5]  # Keep top 5 successors
    
    def _calculate_relevance_score(self, chunk_id: str, query_embedding: np.ndarray) -> float:
        """
        Calculate relevance score for a document chunk.
        
        Args:
            chunk_id: ID of the chunk
            query_embedding: Query embedding
            
        Returns:
            Relevance score (0-1)
        """
        # Get embedding
        if chunk_id not in self.embedding_cache:
            self.embedding_cache[chunk_id] = self.vector_index.get_embedding(chunk_id)
        
        embedding = self.embedding_cache[chunk_id]
        if embedding is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            embedding.reshape(1, -1)
        )[0][0]
        
        # Get usage statistics
        usage_score = self._get_usage_score(chunk_id)
        
        # Get recency score
        recency_score = self._get_recency_score(chunk_id)
        
        # Combine scores
        relevance = (similarity * 0.5) + (usage_score * 0.3) + (recency_score * 0.2)
        
        return max(0.0, min(1.0, relevance))
    
    def _get_usage_score(self, chunk_id: str) -> float:
        """Get normalized usage frequency score."""
        try:
            usage_stats = self.vector_index.get_usage_stats(top_k=1000)
            usage_map = {stat['chunk_id']: stat['access_count'] for stat in usage_stats}
            
            max_usage = max(usage_map.values()) if usage_map else 1
            usage_count = usage_map.get(chunk_id, 0)
            
            return usage_count / max_usage
        except:
            return 0.0
    
    def _get_recency_score(self, chunk_id: str) -> float:
        """Get normalized recency score based on last access."""
        try:
            usage_stats = self.vector_index.get_usage_stats(top_k=1000)
            usage_map = {stat['chunk_id']: stat['last_accessed'] for stat in usage_stats}
            
            last_accessed = usage_map.get(chunk_id)
            if not last_accessed:
                return 0.0
            
            # Calculate days since last access
            last_time = datetime.fromisoformat(last_accessed)
            days_old = (datetime.now() - last_time).days
            
            # Exponential decay: newer documents get higher scores
            recency_score = math.exp(-days_old / 30.0)  # 30-day half-life
            
            return min(1.0, recency_score)
        except:
            return 0.0
    
    def _calculate_cohesion_bonus(self, current_docs: Set[str], new_doc: str) -> float:
        """
        Calculate cohesion bonus for adding a document to the current set.
        
        Args:
            current_docs: Current document set
            new_doc: New document to add
            
        Returns:
            Cohesion bonus score
        """
        if not current_docs:
            return 0.0
        
        # Calculate average similarity to current documents
        similarities = []
        
        for current_doc in current_docs:
            if new_doc in self.document_graph and current_doc in self.document_graph[new_doc]:
                # Documents are directly connected
                similarities.append(0.8)  # High similarity
            else:
                # Calculate embedding similarity
                emb1 = self.embedding_cache.get(new_doc)
                emb2 = self.embedding_cache.get(current_doc)
                
                if emb1 is not None and emb2 is not None:
                    similarity = cosine_similarity(
                        emb1.reshape(1, -1),
                        emb2.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity * 0.2  # Up to 20% bonus
        
        return 0.0
    
    def _calculate_heuristic(self, current_doc: str, query_embedding: np.ndarray, remaining_needed: int) -> float:
        """
        Calculate heuristic estimate for A* search.
        
        Args:
            current_doc: Current document
            query_embedding: Query embedding
            remaining_needed: Number of additional documents needed
            
        Returns:
            Heuristic estimate
        """
        # Estimate maximum possible relevance for remaining documents
        max_relevance = self.heuristic_weight * remaining_needed
        
        # Factor in distance to goal
        distance_penalty = remaining_needed * 0.1
        
        return max(0.0, max_relevance - distance_penalty)
    
    def _format_search_results(self, document_set: Set[str], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Format search results for output.
        
        Args:
            document_set: Set of document IDs
            query_embedding: Query embedding
            
        Returns:
            List of formatted results
        """
        results = []
        
        for chunk_id in document_set:
            # Get document metadata
            metadata = self.vector_index.get_chunk_metadata(chunk_id)
            if not metadata:
                continue
            
            # Calculate final relevance score
            relevance_score = self._calculate_relevance_score(chunk_id, query_embedding)
            
            # Format result
            result = {
                'chunk_id': chunk_id,
                'text': metadata['chunk_text'],
                'file_path': metadata['file_path'],
                'filename': metadata['filename'],
                'relevance_score': relevance_score,
                'sections': metadata.get('sections', {}),
                'metadata': metadata.get('metadata', {}),
                'timestamp': metadata['created_at']
            }
            
            results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:self.top_k]
    
    def _update_usage_statistics(self, results: List[Dict[str, Any]]):
        """Update usage statistics for retrieved documents."""
        for result in results:
            chunk_id = result['chunk_id']
            relevance_score = result['relevance_score']
            
            self.vector_index.update_relevance_score(chunk_id, relevance_score)
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about search performance and patterns.
        
        Returns:
            Dictionary containing search analytics
        """
        # Analyze query history
        recent_queries = list(self.query_history)
        
        query_patterns = defaultdict(int)
        for query_data in recent_queries:
            query = query_data['query']
            # Extract key terms
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    query_patterns[word] += 1
        
        # Most frequent query terms
        top_terms = sorted(query_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_queries': len(recent_queries),
            'average_results': np.mean([q['result_count'] for q in recent_queries]) if recent_queries else 0,
            'top_query_terms': top_terms,
            'document_graph_nodes': len(self.document_graph),
            'embedding_cache_size': len(self.embedding_cache)
        }