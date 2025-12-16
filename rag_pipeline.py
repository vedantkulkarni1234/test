"""
Main RAG pipeline orchestration module.
Integrates all components: PDF monitoring, embedding, A* search, Gemini client, and agent execution.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import threading
import time
import json
from dataclasses import dataclass

from config import Config, config
from pdf_monitor import PDFMonitor
from pdf_parser import PDFProcessor
from embedding_manager import EmbeddingManager
from astar_retriever import AStarRetriever
from gemini_client import GeminiClient, GeminiRequest
from agent_executor import AgentOrchestrator, TaskExecutor, AgentTask, TaskPriority


logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """Represents a RAG query with options."""
    query: str
    context_mode: str = "full"  # 'full', 'concise', 'minimal'
    agentic_mode: bool = True
    max_documents: int = 5
    search_depth: int = 5
    output_format: str = "comprehensive"  # 'comprehensive', 'brief', 'structured'
    
    def __post_init__(self):
        """Validate query parameters."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        valid_context_modes = ["full", "concise", "minimal"]
        if self.context_mode not in valid_context_modes:
            raise ValueError(f"Context mode must be one of: {valid_context_modes}")
        
        valid_output_formats = ["comprehensive", "brief", "structured"]
        if self.output_format not in valid_output_formats:
            raise ValueError(f"Output format must be one of: {valid_output_formats}")


@dataclass
class RAGResponse:
    """Response from RAG system."""
    query: str
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    search_analytics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'answer': self.answer,
            'confidence_score': self.confidence_score,
            'sources': self.sources,
            'search_analytics': self.search_analytics,
            'agent_metrics': self.agent_metrics,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }


class AgenticRAGSystem:
    """
    Complete Agentic RAG System with Gemini 2.5 Flash, A* Search, and PDF Monitoring.
    
    Features:
    - Real-time PDF monitoring and ingestion
    - Local vector storage and A* search
    - Gemini 2.5 Flash for advanced reasoning
    - Agentic task execution
    - Production-ready error handling
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 pdf_directory: Optional[Path] = None,
                 embedding_model: str = None,
                 enable_monitoring: bool = True,
                 config_override: Dict[str, Any] = None):
        """
        Initialize the Agentic RAG System.
        
        Args:
            gemini_api_key: API key for Gemini 2.5 Flash
            pdf_directory: Directory to monitor for PDFs (optional)
            embedding_model: Sentence transformer model name (optional)
            enable_monitoring: Whether to enable PDF monitoring (optional)
            config_override: Configuration overrides (optional)
        """
        self.start_time = datetime.now()
        
        # Initialize configuration
        self._setup_configuration(gemini_api_key, pdf_directory, embedding_model, config_override)
        
        # Validate configuration
        validation_result = config.validate_config()
        if not validation_result['valid']:
            raise ValueError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Initialize core components
        self._initialize_components()
        
        # Initialize monitoring (if enabled)
        self.monitoring_active = False
        if enable_monitoring:
            self._initialize_monitoring()
        
        # Performance tracking
        self.performance_metrics = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_query_time': 0.0,
            'documents_processed': 0,
            'embeddings_generated': 0,
            'searches_performed': 0
        }
        self.metrics_lock = threading.RLock()
        
        logger.info("Agentic RAG System initialized successfully")
    
    def _setup_configuration(self, gemini_api_key: str, pdf_directory: Optional[Path], 
                           embedding_model: str, config_override: Dict[str, Any]):
        """Setup and configure system parameters."""
        # Set configuration values
        if gemini_api_key:
            config.GEMINI_API_KEY = gemini_api_key
        
        if pdf_directory:
            config.PDF_MONITOR_PATH = Path(pdf_directory)
        
        if embedding_model:
            config.EMBEDDING_MODEL = embedding_model
        
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration option: {key}")
    
    def _initialize_components(self):
        """Initialize all core RAG components."""
        logger.info("Initializing RAG components...")
        
        # PDF Processor
        self.pdf_processor = PDFProcessor()
        
        # Embedding Manager
        self.embedding_manager = EmbeddingManager(config.EMBEDDING_CACHE_PATH)
        
        # A* Retriever
        search_config = config.get_search_config()
        self.retriever = AStarRetriever(self.embedding_manager.vector_index, search_config)
        
        # Gemini Client
        gemini_config = config.get_gemini_config()
        self.gemini_client = GeminiClient(gemini_config['api_key'])
        
        # Agent Executor and Orchestrator
        task_executor = TaskExecutor(self.gemini_client, self.embedding_manager, self.retriever)
        self.agent_orchestrator = AgentOrchestrator(task_executor)
        
        # Create extracted texts directory
        self.pdf_processor.output_dir.mkdir(exist_ok=True)
        
        logger.info("All RAG components initialized")
    
    def _initialize_monitoring(self):
        """Initialize PDF monitoring."""
        try:
            self.pdf_monitor = PDFMonitor(
                config.PDF_MONITOR_PATH,
                self.pdf_processor,
                self.embedding_manager
            )
            
            # Add monitoring callbacks
            self.pdf_monitor.add_ingestion_callback(self._on_pdf_ingested)
            self.pdf_monitor.add_error_callback(self._on_pdf_ingestion_error)
            
            logger.info("PDF monitoring initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PDF monitoring: {e}")
            self.pdf_monitor = None
    
    def process_query(self, query: str, **kwargs) -> RAGResponse:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: Query string
            **kwargs: Additional query parameters
            
        Returns:
            RAGResponse object
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # Create query object
            rag_query = RAGQuery(query=query, **kwargs)
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.generator.generate_embedding(query)
            
            # Step 2: Perform A* search for relevant documents
            search_results = self.retriever.search(
                query, 
                query_embedding, 
                top_k=rag_query.max_documents
            )
            
            if not search_results:
                # No relevant documents found
                return self._create_empty_response(query, start_time, "No relevant documents found")
            
            # Step 3: Prepare context for Gemini
            context_documents = self._prepare_context(rag_query, search_results)
            
            # Step 4: Agentic execution (if enabled)
            if rag_query.agentic_mode:
                response_data = self._execute_agentic_pipeline(rag_query, context_documents)
            else:
                # Simple RAG pipeline
                response_data = self._execute_simple_rag(rag_query, context_documents)
            
            # Step 5: Calculate confidence score
            confidence_score = self._calculate_confidence_score(search_results, response_data)
            
            # Step 6: Update performance metrics
            self._update_performance_metrics(time.time() - start_time)
            
            # Step 7: Create response
            response = RAGResponse(
                query=query,
                answer=response_data['answer'],
                confidence_score=confidence_score,
                sources=search_results,
                search_analytics=self.retriever.get_search_analytics(),
                agent_metrics=response_data.get('agent_metrics', {}),
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
            logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Create error response
            return self._create_empty_response(query, start_time, f"Processing error: {str(e)}")
    
    def _execute_agentic_pipeline(self, rag_query: RAGQuery, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute agentic task pipeline."""
        logger.info("Executing agentic pipeline")
        
        # Create task plan
        tasks = self.agent_orchestrator.create_task_plan(rag_query.query, context_documents)
        
        # Execute task plan
        execution_results = self.agent_orchestrator.execute_task_plan(tasks)
        
        # Extract final output
        final_output = execution_results['final_outputs']
        
        # Prepare response data
        if final_output['primary_output']:
            answer = final_output['primary_output'].get('content', str(final_output['primary_output']))
        else:
            # Fallback to synthesized response from supporting outputs
            supporting_content = []
            for output in final_output['supporting_outputs']:
                if isinstance(output['output'], dict):
                    supporting_content.append(str(output['output']))
                else:
                    supporting_content.append(output['output'])
            
            if supporting_content:
                synthesis_prompt = f"Synthesize the following information to answer: {rag_query.query}\n\n" + "\n\n".join(supporting_content)
                synthesis_response = self.gemini_client.generate(synthesis_prompt)
                answer = synthesis_response.text
            else:
                answer = "I couldn't find sufficient information to answer your query."
        
        agent_metrics = {
            'tasks_executed': execution_results['completed_tasks'],
            'total_tasks': execution_results['total_tasks'],
            'success_rate': execution_results['success_rate'],
            'execution_time': execution_results['execution_time'],
            'final_outputs': final_output
        }
        
        return {
            'answer': answer,
            'agent_metrics': agent_metrics
        }
    
    def _execute_simple_rag(self, rag_query: RAGQuery, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute simple RAG pipeline without agentic execution."""
        logger.info("Executing simple RAG pipeline")
        
        # Prepare context text
        context_texts = [doc['text'] for doc in context_documents]
        
        # Generate response using Gemini
        response = self.gemini_client.generate(
            rag_query.query,
            context_texts
        )
        
        agent_metrics = {
            'pipeline_type': 'simple_rag',
            'context_documents': len(context_documents),
            'gemini_usage': response.usage
        }
        
        return {
            'answer': response.text,
            'agent_metrics': agent_metrics
        }
    
    def _prepare_context(self, rag_query: RAGQuery, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare context documents based on query mode.
        
        Args:
            rag_query: Query configuration
            search_results: Search results from A* retriever
            
        Returns:
            Prepared context documents
        """
        context_documents = []
        
        for result in search_results:
            doc_context = {
                'text': result['text'],
                'filename': result['filename'],
                'relevance_score': result['relevance_score'],
                'source': result['file_path']
            }
            
            # Add metadata based on context mode
            if rag_query.context_mode == "full":
                doc_context['metadata'] = result.get('metadata', {})
                doc_context['sections'] = result.get('sections', {})
            elif rag_query.context_mode == "concise":
                # Add summary of sections
                sections = result.get('sections', {})
                if sections:
                    doc_context['section_summary'] = {
                        k: v[:100] + "..." if len(str(v)) > 100 else v 
                        for k, v in sections.items()
                    }
            
            context_documents.append(doc_context)
        
        return context_documents
    
    def _calculate_confidence_score(self, search_results: List[Dict[str, Any]], response_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            search_results: Search results
            response_data: Response data from pipeline
            
        Returns:
            Confidence score (0-1)
        """
        if not search_results:
            return 0.0
        
        # Base score from document relevance
        avg_relevance = sum(doc['relevance_score'] for doc in search_results) / len(search_results)
        
        # Adjust based on agentic execution success
        agent_metrics = response_data.get('agent_metrics', {})
        
        if 'success_rate' in agent_metrics:
            agent_success_bonus = agent_metrics['success_rate'] * 0.2
        else:
            agent_success_bonus = 0.1  # Simple RAG gets small bonus
        
        # Combine scores
        confidence = min(1.0, avg_relevance + agent_success_bonus)
        
        return confidence
    
    def _create_empty_response(self, query: str, start_time: float, error_message: str) -> RAGResponse:
        """Create response for empty or error cases."""
        processing_time = time.time() - start_time
        
        return RAGResponse(
            query=query,
            answer=f"I apologize, but I couldn't find relevant information to answer your query. {error_message}",
            confidence_score=0.0,
            sources=[],
            search_analytics={},
            agent_metrics={},
            processing_time=processing_time,
            timestamp=datetime.now()
        )
    
    def _update_performance_metrics(self, query_time: float):
        """Update system performance metrics."""
        with self.metrics_lock:
            self.performance_metrics['queries_processed'] += 1
            self.performance_metrics['total_processing_time'] += query_time
            self.performance_metrics['average_query_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['queries_processed']
            )
    
    def _on_pdf_ingested(self, ingestion_event: Dict[str, Any]):
        """Callback for successful PDF ingestion."""
        logger.info(f"PDF ingested: {ingestion_event['filename']}")
        
        with self.metrics_lock:
            self.performance_metrics['documents_processed'] += 1
            self.performance_metrics['embeddings_generated'] += ingestion_event['chunks_count']
    
    def _on_pdf_ingestion_error(self, error_event: Dict[str, Any]):
        """Callback for PDF ingestion errors."""
        logger.error(f"PDF ingestion error: {error_event['filename']} - {error_event['error']}")
    
    def start_monitoring(self):
        """Start PDF monitoring."""
        if self.pdf_monitor and not self.monitoring_active:
            self.pdf_monitor.start_monitoring()
            self.monitoring_active = True
            logger.info("PDF monitoring started")
        else:
            logger.warning("PDF monitoring already active or not available")
    
    def stop_monitoring(self):
        """Stop PDF monitoring."""
        if self.pdf_monitor and self.monitoring_active:
            self.pdf_monitor.stop_monitoring()
            self.monitoring_active = False
            logger.info("PDF monitoring stopped")
        else:
            logger.warning("PDF monitoring not active")
    
    def process_existing_pdfs(self):
        """Process all existing PDF files in the monitoring directory."""
        if self.pdf_monitor:
            self.pdf_monitor.process_existing_files()
            logger.info("Processed existing PDF files")
        else:
            logger.warning("PDF monitoring not available")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status dictionary
        """
        uptime = datetime.now() - self.start_time
        
        status = {
            'system_info': {
                'uptime_seconds': uptime.total_seconds(),
                'uptime_hours': uptime.total_seconds() / 3600,
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat()
            },
            'performance_metrics': self.performance_metrics.copy(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'monitor_path': str(config.PDF_MONITOR_PATH),
                'pdf_monitor_available': self.pdf_monitor is not None
            },
            'component_status': {
                'pdf_processor': 'active',
                'embedding_manager': 'active',
                'astar_retriever': 'active',
                'gemini_client': 'active' if self.gemini_client.health_check()['status'] == 'healthy' else 'error',
                'agent_orchestrator': 'active'
            }
        }
        
        # Add monitoring statistics if available
        if self.pdf_monitor:
            status['monitoring_statistics'] = self.pdf_monitor.get_monitoring_stats()
            status['monitored_files'] = len(self.pdf_monitor.get_monitored_files())
        
        # Add Gemini client health check
        try:
            gemini_health = self.gemini_client.health_check()
            status['component_status']['gemini_client'] = gemini_health
        except:
            status['component_status']['gemini_client'] = {'status': 'error', 'error': 'Health check failed'}
        
        return status
    
    def get_usage_analytics(self) -> Dict[str, Any]:
        """
        Get detailed usage analytics.
        
        Returns:
            Usage analytics dictionary
        """
        analytics = {
            'performance_summary': self.performance_metrics.copy(),
            'search_analytics': self.retriever.get_search_analytics(),
            'monitoring_analytics': {}
        }
        
        # Add monitoring analytics if available
        if self.pdf_monitor:
            monitoring_stats = self.pdf_monitor.get_monitoring_stats()
            analytics['monitoring_analytics'] = monitoring_stats
        
        # Calculate derived metrics
        with self.metrics_lock:
            if self.performance_metrics['queries_processed'] > 0:
                analytics['efficiency_metrics'] = {
                    'queries_per_hour': (self.performance_metrics['queries_processed'] / 
                                       max(1, (datetime.now() - self.start_time).total_seconds() / 3600)),
                    'average_processing_time': self.performance_metrics['average_query_time'],
                    'system_utilization': min(1.0, self.performance_metrics['queries_processed'] / 100.0)
                }
        
        return analytics
    
    def export_knowledge_base(self, export_path: Path, include_embeddings: bool = False):
        """
        Export the knowledge base for backup or migration.
        
        Args:
            export_path: Path to export file
            include_embeddings: Whether to include embeddings (large file)
        """
        logger.info(f"Exporting knowledge base to {export_path}")
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_version': '1.0.0',
            'config': {
                'embedding_model': config.EMBEDDING_MODEL,
                'embedding_dimension': config.EMBEDDING_DIMENSION,
                'pdf_monitor_path': str(config.PDF_MONITOR_PATH)
            },
            'documents': {},
            'statistics': self.performance_metrics.copy()
        }
        
        # Export document metadata and text
        document_chunks = self.embedding_manager.vector_index._metadata_cache
        
        for chunk_id, metadata in document_chunks.items():
            export_data['documents'][chunk_id] = {
                'metadata': metadata,
                'file_hash': metadata.get('file_hash', ''),
                'created_at': metadata.get('created_at', ''),
                'chunk_text': metadata.get('chunk_text', '')
            }
            
            # Optionally include embeddings
            if include_embeddings:
                embedding = self.embedding_manager.vector_index.get_embedding(chunk_id)
                if embedding is not None:
                    export_data['documents'][chunk_id]['embedding'] = embedding.tolist()
        
        # Save to file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base exported successfully to {export_path}")
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Direct document search without full RAG pipeline.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        logger.info(f"Direct document search: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generator.generate_embedding(query)
        
        # Perform A* search
        search_results = self.retriever.search(query, query_embedding, top_k)
        
        with self.metrics_lock:
            self.performance_metrics['searches_performed'] += 1
        
        return search_results
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a specific document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document information
        """
        chunks = self.embedding_manager.get_document_chunks(file_path)
        
        if not chunks:
            return {'error': 'Document not found'}
        
        # Calculate document statistics
        total_text_length = sum(len(chunk.get('chunk_text', '')) for chunk in chunks)
        avg_relevance = sum(chunk.get('relevance_score', 0.0) for chunk in chunks) / len(chunks)
        
        return {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'chunks_count': len(chunks),
            'total_text_length': total_text_length,
            'average_relevance': avg_relevance,
            'chunks': chunks
        }
    
    def shutdown(self):
        """Shutdown the RAG system and clean up resources."""
        logger.info("Shutting down Agentic RAG System")
        
        # Stop monitoring
        if self.monitoring_active:
            self.stop_monitoring()
        
        # Shutdown PDF monitor
        if self.pdf_monitor:
            self.pdf_monitor.shutdown()
        
        # Close Gemini client
        if self.gemini_client:
            self.gemini_client.close()
        
        logger.info("Agentic RAG System shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()