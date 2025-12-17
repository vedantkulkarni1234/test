"""
A* Search Algorithm implementation for ultra-fast document ranking and retrieval.
Implements intelligent document navigation with relevance scoring and path optimization.
"""

import heapq
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Generator
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from datetime import datetime
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


@dataclass
class AStarSearchEvent:
    """Serializable event describing the internal progress of the A* search."""

    step: int
    event_type: str  # 'graph', 'expand', 'enqueue', 'result'

    expanded_chunk_id: Optional[str] = None
    added_chunk_id: Optional[str] = None

    state_documents: Optional[List[str]] = None
    path: Optional[List[str]] = None

    open_set_size: int = 0
    visited_states: int = 0

    cost_so_far: float = 0.0
    heuristic: float = 0.0
    total_cost: float = 0.0
    relevance_score: float = 0.0

    additional_relevance: Optional[float] = None
    cohesion_bonus: Optional[float] = None
    connected_to: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            'step': self.step,
            'type': self.event_type,
            'open_set_size': self.open_set_size,
            'visited_states': self.visited_states,
            'cost_so_far': self.cost_so_far,
            'heuristic': self.heuristic,
            'total_cost': self.total_cost,
            'relevance_score': self.relevance_score,
        }

        if self.expanded_chunk_id is not None:
            data['expanded_chunk_id'] = self.expanded_chunk_id
        if self.added_chunk_id is not None:
            data['added_chunk_id'] = self.added_chunk_id
        if self.state_documents is not None:
            data['state_documents'] = self.state_documents
        if self.path is not None:
            data['path'] = self.path
        if self.additional_relevance is not None:
            data['additional_relevance'] = self.additional_relevance
        if self.cohesion_bonus is not None:
            data['cohesion_bonus'] = self.cohesion_bonus
        if self.connected_to is not None:
            data['connected_to'] = self.connected_to

        return data


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
        self.document_graph_weights: Dict[Tuple[str, str], float] = {}
        self.embedding_cache = {}
        self.query_history = deque(maxlen=100)  # Track recent queries

        # Debug/visualization state
        self.last_search_debug: Optional[Dict[str, Any]] = None
        
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

    def search_with_trace(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = None,
        max_steps: int = 250,
    ) -> Dict[str, Any]:
        """Run the search and return graph + trace data suitable for a UI."""
        graph: Optional[Dict[str, Any]] = None
        trace: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        final_path: List[str] = []

        for event in self.search_stream(query, query_embedding, top_k=top_k, max_steps=max_steps):
            if event['type'] == 'graph':
                graph = event['graph']
            elif event['type'] == 'result':
                results = event.get('results', [])
                final_path = event.get('final_path', [])
            else:
                trace.append(event)

        debug = {
            'query': query,
            'graph': graph,
            'trace': trace,
            'results': results,
            'final_path': final_path,
        }
        self.last_search_debug = debug
        return debug

    def search_stream(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = None,
        max_steps: int = 250,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream A* events for "glass box" visualization.

        Yields a first event of type 'graph' containing the document graph, followed by
        step events ('expand' / 'enqueue'), and finally a 'result' event.
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Starting A* search stream for query: '{query[:50]}...'")

        candidates = self._get_initial_candidates(query_embedding, top_k * 3)
        if not candidates:
            yield {
                'type': 'graph',
                'graph': {'nodes': [], 'edges': []},
                'candidates_count': 0,
            }
            yield {'type': 'result', 'results': [], 'final_path': [], 'final_document_set': []}
            return

        self._build_document_graph(candidates)
        graph = self._export_document_graph(candidates, query_embedding)

        yield {'type': 'graph', 'graph': graph, 'candidates_count': len(candidates)}

        final_event: Optional[Dict[str, Any]] = None
        for event in self._astar_search_stream(candidates, query_embedding, top_k=top_k, max_steps=max_steps):
            if event.get('type') == 'result':
                final_event = event
            yield event

        if final_event and final_event.get('results'):
            self._update_usage_statistics(final_event['results'])

        self.query_history.append({
            'query': query,
            'query_embedding': query_embedding,
            'timestamp': datetime.now(),
            'result_count': len(final_event.get('results', [])) if final_event else 0,
        })

        self.last_search_debug = {
            'query': query,
            'graph': graph,
            'trace': [],
            'results': final_event.get('results', []) if final_event else [],
            'final_path': final_event.get('final_path', []) if final_event else [],
        }

    def get_last_search_debug(self) -> Optional[Dict[str, Any]]:
        """Return the most recent graph/trace payload produced by search_with_trace."""
        return self.last_search_debug
    
    def _get_initial_candidates(self, query_embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """
        Get initial candidate documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of candidates
            
        Returns:
            List of candidate documents
        """
        # Use the vector index's similarity search (preferred)
        if hasattr(self.vector_index, 'search_similar_chunks'):
            candidates = self.vector_index.search_similar_chunks(
                query_embedding,
                top_k=limit
            )
        else:
            # Fallback: compute similarities from cached embeddings/metadata
            candidates = []
            for chunk_id, embedding in getattr(self.vector_index, '_embeddings_cache', {}).items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                metadata = getattr(self.vector_index, '_metadata_cache', {}).get(chunk_id)
                if metadata:
                    candidates.append({
                        'chunk_id': chunk_id,
                        'similarity': float(similarity),
                        'chunk_text': metadata.get('chunk_text', ''),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', '')
                    })
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            candidates = candidates[:limit]
        
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
        self.document_graph_weights.clear()
        
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

                    weight = float(similarity)
                    self.document_graph_weights[(chunk_id1, chunk_id2)] = weight
                    self.document_graph_weights[(chunk_id2, chunk_id1)] = weight
        
        logger.debug(f"Built document graph with {len(self.document_graph)} nodes")

    def _export_document_graph(self, candidates: List[Dict[str, Any]], query_embedding: np.ndarray) -> Dict[str, Any]:
        """Export the current candidate graph in a JSON-friendly format."""
        nodes: List[Dict[str, Any]] = []
        candidate_map = {c['chunk_id']: c for c in candidates}

        for chunk_id, candidate in candidate_map.items():
            metadata = self.vector_index.get_chunk_metadata(chunk_id) or {}
            filename = metadata.get('filename') or candidate.get('filename') or chunk_id
            text = metadata.get('chunk_text') or candidate.get('chunk_text') or candidate.get('text') or ''
            preview = (text[:200] + '...') if len(text) > 200 else text

            nodes.append({
                'id': chunk_id,
                'label': filename,
                'similarity': float(candidate.get('similarity', 0.0)),
                'relevance_score': float(self._calculate_relevance_score(chunk_id, query_embedding)),
                'file_path': metadata.get('file_path') or candidate.get('file_path'),
                'filename': filename,
                'text_preview': preview,
            })

        edges: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()

        for src, neighbors in self.document_graph.items():
            for dst in neighbors:
                if (dst, src) in seen:
                    continue
                seen.add((src, dst))

                edges.append({
                    'source': src,
                    'target': dst,
                    'weight': float(self.document_graph_weights.get((src, dst), 0.0)),
                })

        # Ensure isolated candidates are still represented as nodes
        return {'nodes': nodes, 'edges': edges}

    def _astar_search_stream(
        self,
        candidates: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        top_k: int,
        max_steps: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """A* search as a generator that yields internal events."""
        open_set: List[SearchState] = []

        for candidate in candidates:
            chunk_id = candidate['chunk_id']
            relevance = self._calculate_relevance_score(chunk_id, query_embedding)
            heuristic = self._calculate_heuristic(chunk_id, query_embedding, top_k)
            cost = 1.0

            heapq.heappush(
                open_set,
                SearchState(
                    document_set=frozenset([chunk_id]),
                    relevance_score=relevance,
                    cost_so_far=cost,
                    heuristic=heuristic,
                    path=[chunk_id],
                    depth=0,
                ),
            )

        visited: Set[Tuple[str, ...]] = set()
        step = 0

        while open_set and len(visited) < 1000 and step < max_steps:
            current_state = heapq.heappop(open_set)
            state_key = tuple(sorted(current_state.document_set))
            if state_key in visited:
                continue
            visited.add(state_key)

            expanded_chunk_id = current_state.path[-1] if current_state.path else None

            yield AStarSearchEvent(
                step=step,
                event_type='expand',
                expanded_chunk_id=expanded_chunk_id,
                state_documents=list(state_key),
                path=current_state.path,
                open_set_size=len(open_set),
                visited_states=len(visited),
                cost_so_far=current_state.cost_so_far,
                heuristic=current_state.heuristic,
                total_cost=current_state.total_cost,
                relevance_score=current_state.relevance_score,
            ).to_dict()

            if len(current_state.document_set) >= top_k:
                results = self._format_search_results(current_state.document_set, query_embedding)
                yield {
                    'type': 'result',
                    'results': results,
                    'final_document_set': list(state_key),
                    'final_path': current_state.path,
                    'visited_states': len(visited),
                }
                return

            if current_state.depth >= self.max_depth:
                step += 1
                continue

            successors = self._generate_successors_with_details(current_state, candidates, query_embedding, top_k)
            for successor_state, details in successors:
                successor_key = tuple(sorted(successor_state.document_set))
                if successor_key in visited:
                    continue

                heapq.heappush(open_set, successor_state)

                yield AStarSearchEvent(
                    step=step,
                    event_type='enqueue',
                    expanded_chunk_id=expanded_chunk_id,
                    added_chunk_id=details['added_chunk_id'],
                    state_documents=list(successor_key),
                    path=successor_state.path,
                    open_set_size=len(open_set),
                    visited_states=len(visited),
                    cost_so_far=successor_state.cost_so_far,
                    heuristic=successor_state.heuristic,
                    total_cost=successor_state.total_cost,
                    relevance_score=successor_state.relevance_score,
                    additional_relevance=details['additional_relevance'],
                    cohesion_bonus=details['cohesion_bonus'],
                    connected_to=details['connected_to'],
                ).to_dict()

            step += 1

        if open_set:
            best_state = min(open_set, key=lambda s: s.total_cost)
            results = self._format_search_results(best_state.document_set, query_embedding)
            yield {
                'type': 'result',
                'results': results,
                'final_document_set': list(sorted(best_state.document_set)),
                'final_path': best_state.path,
                'visited_states': len(visited),
            }
            return

        logger.warning("A* search failed to find solution")
        yield {
            'type': 'result',
            'results': [],
            'final_document_set': [],
            'final_path': [],
            'visited_states': len(visited),
        }

    def _astar_search(self, candidates: List[Dict[str, Any]], query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Perform A* search to find optimal document subset."""
        results: List[Dict[str, Any]] = []

        for event in self._astar_search_stream(
            candidates,
            query_embedding,
            top_k=top_k,
            max_steps=1000,
        ):
            if event.get('type') == 'result':
                results = event.get('results', [])

        return results

    def _generate_successors(
        self,
        current_state: SearchState,
        candidates: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[SearchState]:
        """Generate successor states by adding new documents."""
        return [
            state for state, _ in self._generate_successors_with_details(
                current_state, candidates, query_embedding, top_k
            )
        ]

    def _generate_successors_with_details(
        self,
        current_state: SearchState,
        candidates: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[SearchState, Dict[str, Any]]]:
        """Generate successor states plus metadata for "glass box" explanations."""
        if len(current_state.document_set) >= top_k or current_state.depth >= self.max_depth:
            return []

        current_docs = set(current_state.document_set)
        available_docs = [doc for doc in candidates if doc['chunk_id'] not in current_docs]

        successors: List[Tuple[SearchState, Dict[str, Any]]] = []

        for doc in available_docs:
            chunk_id = doc['chunk_id']

            if chunk_id not in self.embedding_cache:
                self.embedding_cache[chunk_id] = self.vector_index.get_embedding(chunk_id)

            additional_relevance = self._calculate_relevance_score(chunk_id, query_embedding)
            connected_to = sorted(list(current_docs & self.document_graph.get(chunk_id, set())))
            cohesion_bonus = self._calculate_cohesion_bonus(current_docs, chunk_id)

            new_relevance = current_state.relevance_score + additional_relevance + cohesion_bonus
            additional_cost = 1.0 + (len(current_docs) * 0.1)

            remaining_needed = top_k - len(current_docs) - 1
            new_heuristic = self._calculate_heuristic(chunk_id, query_embedding, remaining_needed)

            new_state = SearchState(
                document_set=frozenset(current_docs | {chunk_id}),
                relevance_score=new_relevance,
                cost_so_far=current_state.cost_so_far + additional_cost,
                heuristic=new_heuristic,
                path=current_state.path + [chunk_id],
                depth=current_state.depth + 1,
            )

            successors.append(
                (
                    new_state,
                    {
                        'added_chunk_id': chunk_id,
                        'additional_relevance': float(additional_relevance),
                        'cohesion_bonus': float(cohesion_bonus),
                        'connected_to': connected_to,
                    },
                )
            )

        # Keep lowest-cost successors (best according to A* evaluation)
        successors.sort(key=lambda item: item[0].total_cost)
        return successors[:5]

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