"""
Main entry point for the Agentic RAG System.
Provides easy-to-use interface for initializing and operating the system.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import argparse
import sys

from rag_pipeline import AgenticRAGSystem, RAGQuery, RAGResponse
from config import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)


class RAGSystemCLI:
    """Command-line interface for the Agentic RAG System."""
    
    def __init__(self):
        self.rag_system = None
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Agentic RAG System with Gemini 2.5 Flash & A* Search",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s query "Analyze market trends in the uploaded documents"
  %(prog)s --api-key YOUR_KEY query "What are the key findings?"
  %(prog)s --pdf-dir ./docs monitor
  %(prog)s status
  %(prog)s search "machine learning"
  %(prog)s export ./backup.json
            """
        )
        
        # Required arguments
        parser.add_argument(
            '--api-key', 
            required=False,
            help='Gemini API key (or set GEMINI_API_KEY environment variable)'
        )
        
        # Optional arguments
        parser.add_argument(
            '--pdf-dir',
            type=Path,
            default='./knowledge_hub',
            help='Directory to monitor for PDF files (default: ./knowledge_hub)'
        )
        
        parser.add_argument(
            '--embedding-model',
            default='all-MiniLM-L6-v2',
            help='Sentence transformer model (default: all-MiniLM-L6-v2)'
        )
        
        parser.add_argument(
            '--disable-monitoring',
            action='store_true',
            help='Disable automatic PDF monitoring'
        )
        
        parser.add_argument(
            '--config-file',
            type=Path,
            help='Configuration file (JSON format)'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        
        # Commands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Query command
        query_parser = subparsers.add_parser('query', help='Process a query')
        query_parser.add_argument('query_text', help='Query text to process')
        query_parser.add_argument(
            '--agentic-mode', 
            action='store_true', 
            help='Use agentic task execution'
        )
        query_parser.add_argument(
            '--context-mode',
            choices=['full', 'concise', 'minimal'],
            default='full',
            help='Context preparation mode (default: full)'
        )
        query_parser.add_argument(
            '--max-docs',
            type=int,
            default=5,
            help='Maximum number of documents (default: 5)'
        )
        query_parser.add_argument(
            '--output-format',
            choices=['comprehensive', 'brief', 'structured'],
            default='comprehensive',
            help='Output format (default: comprehensive)'
        )
        query_parser.add_argument(
            '--output-file',
            type=Path,
            help='Save response to file'
        )
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Search documents directly')
        search_parser.add_argument('search_query', help='Search query')
        search_parser.add_argument(
            '--top-k',
            type=int,
            default=10,
            help='Number of results (default: 10)'
        )
        search_parser.add_argument(
            '--output-file',
            type=Path,
            help='Save results to file'
        )
        
        # Monitor command
        monitor_parser = subparsers.add_parser('monitor', help='Start monitoring mode')
        monitor_parser.add_argument(
            '--process-existing',
            action='store_true',
            help='Process existing PDF files first'
        )
        monitor_parser.add_argument(
            '--interval',
            type=float,
            default=5.0,
            help='Monitoring interval in seconds (default: 5.0)'
        )
        
        # Status command
        subparsers.add_parser('status', help='Show system status')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export knowledge base')
        export_parser.add_argument('export_path', help='Export file path')
        export_parser.add_argument(
            '--include-embeddings',
            action='store_true',
            help='Include embeddings in export (large file)'
        )
        
        # Document info command
        doc_info_parser = subparsers.add_parser('doc-info', help='Get document information')
        doc_info_parser.add_argument('file_path', help='Path to document')
        
        return parser
    
    def initialize_system(self, args) -> AgenticRAGSystem:
        """Initialize the RAG system."""
        # Load config from file if provided
        config_override = {}
        if args.config_file and args.config_file.exists():
            import json
            with open(args.config_file, 'r') as f:
                config_override = json.load(f)
        
        # Get API key
        api_key = args.api_key or config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key is required. Set --api-key or GEMINI_API_KEY environment variable.")
        
        # Initialize system
        rag_system = AgenticRAGSystem(
            gemini_api_key=api_key,
            pdf_directory=args.pdf_dir,
            embedding_model=args.embedding_model,
            enable_monitoring=not args.disable_monitoring,
            config_override=config_override
        )
        
        # Process existing PDFs if requested
        if hasattr(args, 'process_existing') and args.process_existing:
            logger.info("Processing existing PDF files...")
            rag_system.process_existing_pdfs()
        
        return rag_system
    
    def run_query(self, args):
        """Run a query."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        # Process query
        query_params = {
            'agentic_mode': args.agentic_mode,
            'context_mode': args.context_mode,
            'max_documents': args.max_docs,
            'output_format': args.output_format
        }
        
        response = self.rag_system.process_query(args.query_text, **query_params)
        
        # Display results
        self._display_response(response, args.output_file)
    
    def run_search(self, args):
        """Run document search."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        # Search documents
        results = self.rag_system.search_documents(args.search_query, args.top_k)
        
        # Display results
        self._display_search_results(results, args.output_file)
    
    def run_monitor(self, args):
        """Start monitoring mode."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        # Start monitoring
        self.rag_system.start_monitoring()
        
        logger.info(f"Monitoring started on {args.pdf_dir}")
        logger.info("Press Ctrl+C to stop...")
        
        try:
            import time
            while True:
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.rag_system.stop_monitoring()
    
    def show_status(self, args):
        """Show system status."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        status = self.rag_system.get_system_status()
        self._display_status(status)
    
    def run_export(self, args):
        """Export knowledge base."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        self.rag_system.export_knowledge_base(args.export_path, args.include_embeddings)
        logger.info(f"Knowledge base exported to {args.export_path}")
    
    def show_document_info(self, args):
        """Show document information."""
        if not self.rag_system:
            self.rag_system = self.initialize_system(args)
        
        doc_info = self.rag_system.get_document_info(args.file_path)
        self._display_document_info(doc_info)
    
    def _display_response(self, response: RAGResponse, output_file: Optional[Path] = None):
        """Display RAG response."""
        print(f"\n{'='*80}")
        print(f"QUERY: {response.query}")
        print(f"{'='*80}")
        print(f"\nANSWER:")
        print(response.answer)
        print(f"\nCONFIDENCE: {response.confidence_score:.2%}")
        print(f"PROCESSING TIME: {response.processing_time:.2f} seconds")
        print(f"SOURCES: {len(response.sources)} documents")
        
        if response.sources:
            print(f"\nSOURCE DOCUMENTS:")
            for i, source in enumerate(response.sources[:5], 1):
                print(f"  {i}. {source['filename']} (relevance: {source['relevance_score']:.2%})")
        
        # Save to file if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\nResponse saved to: {output_file}")
    
    def _display_search_results(self, results: List[Dict[str, Any]], output_file: Optional[Path] = None):
        """Display search results."""
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS: {len(results)} documents found")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Relevance: {result['relevance_score']:.2%}")
            print(f"   Text Preview: {result['text'][:150]}...")
        
        # Save to file if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSearch results saved to: {output_file}")
    
    def _display_status(self, status: Dict[str, Any]):
        """Display system status."""
        print(f"\n{'='*80}")
        print(f"AGENTIC RAG SYSTEM STATUS")
        print(f"{'='*80}")
        
        # System info
        sys_info = status['system_info']
        print(f"\nSYSTEM INFO:")
        print(f"  Uptime: {sys_info['uptime_hours']:.1f} hours")
        print(f"  Started: {sys_info['start_time']}")
        
        # Performance metrics
        perf = status['performance_metrics']
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Queries Processed: {perf['queries_processed']}")
        print(f"  Average Query Time: {perf['average_query_time']:.2f} seconds")
        print(f"  Documents Processed: {perf['documents_processed']}")
        print(f"  Embeddings Generated: {perf['embeddings_generated']}")
        print(f"  Searches Performed: {perf['searches_performed']}")
        
        # Monitoring status
        monitoring = status['monitoring_status']
        print(f"\nMONITORING STATUS:")
        print(f"  Active: {monitoring['active']}")
        print(f"  Monitor Path: {monitoring['monitor_path']}")
        print(f"  PDF Monitor Available: {monitoring['pdf_monitor_available']}")
        
        # Component status
        components = status['component_status']
        print(f"\nCOMPONENT STATUS:")
        for component, status_val in components.items():
            if isinstance(status_val, dict):
                print(f"  {component}: {status_val.get('status', 'unknown')}")
            else:
                print(f"  {component}: {status_val}")
        
        # Monitoring statistics
        if 'monitoring_statistics' in status:
            stats = status['monitoring_statistics']
            print(f"\nMONITORING STATISTICS:")
            print(f"  Files Processed: {stats['files_processed']}")
            print(f"  Total Chunks: {stats['total_chunks_created']}")
            print(f"  Errors: {stats['errors_encountered']}")
            if stats.get('files_per_hour'):
                print(f"  Files per Hour: {stats['files_per_hour']:.1f}")
    
    def _display_document_info(self, doc_info: Dict[str, Any]):
        """Display document information."""
        if 'error' in doc_info:
            print(f"Error: {doc_info['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"DOCUMENT INFORMATION")
        print(f"{'='*80}")
        print(f"File: {doc_info['filename']}")
        print(f"Path: {doc_info['file_path']}")
        print(f"Chunks: {doc_info['chunks_count']}")
        print(f"Text Length: {doc_info['total_text_length']:,} characters")
        print(f"Average Relevance: {doc_info['average_relevance']:.2%}")
        
        print(f"\nCHUNK PREVIEW:")
        for i, chunk in enumerate(doc_info['chunks'][:3], 1):
            print(f"  {i}. {chunk['chunk_text'][:100]}...")
    
    def run(self):
        """Run the CLI."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        try:
            # Route to appropriate command
            if args.command == 'query':
                self.run_query(args)
            elif args.command == 'search':
                self.run_search(args)
            elif args.command == 'monitor':
                self.run_monitor(args)
            elif args.command == 'status':
                self.show_status(args)
            elif args.command == 'export':
                self.run_export(args)
            elif args.command == 'doc-info':
                self.show_document_info(args)
            else:
                print(f"Unknown command: {args.command}")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            logger.error(f"Command failed: {e}")
            print(f"Error: {e}")
        finally:
            # Cleanup
            if self.rag_system:
                self.rag_system.shutdown()


def main():
    """Main entry point."""
    cli = RAGSystemCLI()
    cli.run()


# Python module usage examples
def create_rag_system(api_key: str, 
                     pdf_directory: Path = None,
                     embedding_model: str = None,
                     config_override: Dict[str, Any] = None) -> AgenticRAGSystem:
    """
    Create and initialize a RAG system.
    
    Args:
        api_key: Gemini API key
        pdf_directory: Directory to monitor for PDFs
        embedding_model: Sentence transformer model
        config_override: Configuration overrides
        
    Returns:
        Initialized AgenticRAGSystem
    """
    return AgenticRAGSystem(
        gemini_api_key=api_key,
        pdf_directory=pdf_directory,
        embedding_model=embedding_model,
        config_override=config_override
    )


def quick_query(rag_system: AgenticRAGSystem, 
               query: str, 
               agentic_mode: bool = True,
               max_documents: int = 5) -> str:
    """
    Quick query function for simple usage.
    
    Args:
        rag_system: Initialized RAG system
        query: Query text
        agentic_mode: Use agentic execution
        max_documents: Maximum documents to consider
        
    Returns:
        Response text
    """
    response = rag_system.process_query(
        query, 
        agentic_mode=agentic_mode,
        max_documents=max_documents
    )
    return response.answer


if __name__ == '__main__':
    main()