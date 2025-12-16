"""
Comprehensive demonstration of the Agentic RAG System.
Shows all major features including PDF monitoring, A* search, agentic execution, and Gemini integration.
"""

import os
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import AgenticRAGSystem, RAGQuery
from config import config


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGSystemDemo:
    """Comprehensive demonstration of the Agentic RAG System."""
    
    def __init__(self):
        self.rag_system = None
        self.demo_results = []
        
    def setup_demo(self, api_key: str):
        """Setup the RAG system for demonstration."""
        logger.info("Setting up Agentic RAG System Demo...")
        
        # Create demo directories
        demo_dir = Path("./demo_knowledge_hub")
        demo_dir.mkdir(exist_ok=True)
        
        # Create sample PDF files for demonstration
        self._create_sample_documents(demo_dir)
        
        # Initialize RAG system
        self.rag_system = AgenticRAGSystem(
            gemini_api_key=api_key,
            pdf_directory=demo_dir,
            embedding_model="all-MiniLM-L6-v2",
            enable_monitoring=True,
            config_override={
                'LOG_LEVEL': 'INFO',
                'TOP_K_DOCUMENTS': 3,  # Smaller for demo
                'MAX_AGENT_STEPS': 5   # Faster execution
            }
        )
        
        # Process existing files
        logger.info("Processing demo PDF files...")
        self.rag_system.process_existing_pdfs()
        
        logger.info("Demo setup completed!")
        return True
    
    def _create_sample_documents(self, demo_dir: Path):
        """Create sample PDF-like documents for demonstration."""
        # Create sample text files (representing PDF content)
        samples = [
            {
                'filename': 'ai_market_analysis.txt',
                'content': """
                AI Market Analysis 2024
                
                Executive Summary:
                The artificial intelligence market is experiencing unprecedented growth, with
                revenue projections reaching $1.8 trillion by 2030. Key growth drivers include
                increased enterprise adoption, advancing hardware capabilities, and improved
                algorithmic efficiency.
                
                Key Findings:
                1. Enterprise AI adoption increased by 67% in 2024
                2. Cloud-based AI services dominate with 73% market share
                3. Healthcare and finance sectors show highest ROI from AI investments
                4. Small-to-medium businesses are the fastest growing AI adopters
                
                Market Trends:
                - Generative AI leading with 45% market growth
                - Edge computing AI solutions gaining traction
                - AI governance and ethics becoming critical
                - Custom model development on the rise
                
                Competitive Landscape:
                Major players include OpenAI, Google, Microsoft, and emerging startups.
                Open source models are democratizing AI capabilities.
                
                Recommendations:
                1. Invest in AI infrastructure and talent
                2. Focus on ethical AI development
                3. Develop industry-specific AI solutions
                4. Establish AI governance frameworks
                """
            },
            {
                'filename': 'machine_learning_research.txt',
                'content': """
                Machine Learning Research Advances 2024
                
                Introduction:
                Recent advances in machine learning have revolutionized how we approach
                complex problem-solving across industries. This research focuses on
                breakthrough techniques and their practical applications.
                
                Breakthrough Techniques:
                
                1. Transformer Architecture Evolution
                - Attention mechanisms improved efficiency by 40%
                - Multi-modal models achieving state-of-the-art results
                - Reduced computational requirements for training
                
                2. Federated Learning Advances
                - Privacy-preserving learning in healthcare
                - Real-world deployment in financial services
                - Edge device optimization techniques
                
                3. Reinforcement Learning
                - Policy gradient methods improved convergence
                - Real-time learning in dynamic environments
                - Applications in autonomous systems
                
                Research Applications:
                - Healthcare: Drug discovery and diagnosis
                - Finance: Risk assessment and fraud detection
                - Transportation: Autonomous vehicle optimization
                - Manufacturing: Predictive maintenance
                
                Challenges and Limitations:
                - Data privacy and security concerns
                - Model interpretability requirements
                - Computational resource constraints
                - Ethical considerations in decision making
                
                Future Directions:
                1. Development of more efficient algorithms
                2. Improved model interpretability
                3. Enhanced privacy preservation techniques
                4. Better integration with domain knowledge
                
                Conclusion:
                Machine learning continues to evolve rapidly, with practical applications
                expanding across all sectors. Success depends on addressing current
                challenges while pushing the boundaries of what's possible.
                """
            },
            {
                'filename': 'data_science_trends.txt',
                'content': """
                Data Science Trends and Future Outlook
                
                Overview:
                Data science continues to evolve as organizations seek to extract maximum
                value from their data assets. This analysis examines current trends and
                future projections for the data science field.
                
                Current Trends:
                
                1. Automated Machine Learning (AutoML)
                - 80% reduction in model development time
                - Democratizing access to advanced analytics
                - Integration with cloud platforms
                
                2. Real-time Analytics
                - Streaming data processing capabilities
                - Edge computing integration
                - IoT data analysis at scale
                
                3. Explainable AI (XAI)
                - Regulatory compliance requirements
                - Business stakeholder transparency needs
                - Trust building in AI systems
                
                4. Data Mesh Architecture
                - Decentralized data ownership
                - Domain-driven design principles
                - Data as a product mindset
                
                Technology Stack Evolution:
                - Python and R maintaining dominance
                - Julia gaining adoption for performance
                - SQL still critical for data access
                - Cloud-native tools preferred
                
                Skills and Roles:
                - Data engineers in high demand
                - MLOps becoming essential
                - Domain expertise increasingly valued
                - Ethics and governance skills required
                
                Industry Applications:
                
                Healthcare:
                - Personalized medicine advancement
                - Real-time patient monitoring
                - Drug discovery acceleration
                
                Finance:
                - Algorithmic trading optimization
                - Risk management enhancement
                - Customer experience improvement
                
                Retail:
                - Recommendation system enhancement
                - Inventory optimization
                - Customer behavior prediction
                
                Challenges:
                - Data quality and governance
                - Talent shortage and skill gaps
                - Privacy and security concerns
                - Integration complexity
                
                Future Projections:
                1. AI-native data platforms
                2. Quantum computing integration
                3. Synthetic data generation
                4. Continuous learning systems
                
                Strategic Recommendations:
                1. Invest in data infrastructure
                2. Develop talent pipeline
                3. Establish data governance
                4. Foster cross-functional collaboration
                """
            }
        ]
        
        # Write sample files
        for sample in samples:
            file_path = demo_dir / sample['filename']
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample['content'])
        
        logger.info(f"Created {len(samples)} sample documents in {demo_dir}")
    
    def demo_simple_query(self) -> Dict[str, Any]:
        """Demonstrate simple RAG query processing."""
        logger.info("=== DEMO: Simple RAG Query ===")
        
        query = "What are the key trends in AI market development?"
        
        response = self.rag_system.process_query(
            query,
            agentic_mode=False,  # Simple RAG mode
            max_documents=3,
            context_mode="full"
        )
        
        demo_result = {
            'demo_type': 'simple_query',
            'query': query,
            'answer': response.answer,
            'confidence': response.confidence_score,
            'sources_count': len(response.sources),
            'processing_time': response.processing_time,
            'sources': [s['filename'] for s in response.sources]
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"Query processed in {response.processing_time:.2f}s")
        logger.info(f"Confidence: {response.confidence_score:.2%}")
        logger.info(f"Sources: {demo_result['sources']}")
        
        return demo_result
    
    def demo_agentic_query(self) -> Dict[str, Any]:
        """Demonstrate agentic RAG with task planning."""
        logger.info("=== DEMO: Agentic RAG Query ===")
        
        query = "Analyze the convergence of AI and data science trends and create a strategic implementation plan"
        
        response = self.rag_system.process_query(
            query,
            agentic_mode=True,  # Agentic execution
            max_documents=3,
            context_mode="full",
            output_format="structured"
        )
        
        demo_result = {
            'demo_type': 'agentic_query',
            'query': query,
            'answer': response.answer,
            'confidence': response.confidence_score,
            'sources_count': len(response.sources),
            'processing_time': response.processing_time,
            'agent_metrics': response.agent_metrics,
            'sources': [s['filename'] for s in response.sources]
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"Agentic query processed in {response.processing_time:.2f}s")
        logger.info(f"Agent metrics: {response.agent_metrics}")
        
        return demo_result
    
    def demo_direct_search(self) -> Dict[str, Any]:
        """Demonstrate direct document search."""
        logger.info("=== DEMO: Direct Document Search ===")
        
        search_queries = [
            "machine learning applications",
            "data science trends",
            "AI market analysis"
        ]
        
        search_results = {}
        for query in search_queries:
            results = self.rag_system.search_documents(query, top_k=3)
            search_results[query] = {
                'results_count': len(results),
                'top_result': results[0]['filename'] if results else 'None',
                'relevance_scores': [r['relevance_score'] for r in results[:3]]
            }
        
        demo_result = {
            'demo_type': 'direct_search',
            'queries': search_results,
            'total_searches': len(search_queries)
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"Performed {len(search_queries)} direct searches")
        for query, results in search_results.items():
            logger.info(f"  '{query}': {results['results_count']} results")
        
        return demo_result
    
    def demo_system_analytics(self) -> Dict[str, Any]:
        """Demonstrate system analytics and monitoring."""
        logger.info("=== DEMO: System Analytics ===")
        
        # Get system status
        status = self.rag_system.get_system_status()
        
        # Get usage analytics
        analytics = self.rag_system.get_usage_analytics()
        
        # Get document info for one file
        sample_file = None
        if status.get('monitored_files', 0) > 0:
            # Find a monitored file
            files = list(Path(config.PDF_MONITOR_PATH).glob("**/*.txt"))
            if files:
                sample_file = str(files[0])
        
        doc_info = {}
        if sample_file:
            doc_info = self.rag_system.get_document_info(sample_file)
        
        demo_result = {
            'demo_type': 'system_analytics',
            'system_status': {
                'uptime_hours': status['system_info']['uptime_hours'],
                'components_healthy': sum(1 for c in status['component_status'].values() 
                                        if isinstance(c, dict) and c.get('status') == 'healthy'),
                'monitoring_active': status['monitoring_status']['active'],
                'monitored_files': status.get('monitored_files', 0)
            },
            'performance_metrics': analytics['performance_summary'],
            'search_analytics': analytics['search_analytics'],
            'document_sample': doc_info
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"System uptime: {status['system_info']['uptime_hours']:.1f} hours")
        logger.info(f"Monitoring active: {status['monitoring_status']['active']}")
        logger.info(f"Documents processed: {analytics['performance_summary']['documents_processed']}")
        
        return demo_result
    
    def demo_monitoring_capabilities(self) -> Dict[str, Any]:
        """Demonstrate PDF monitoring capabilities."""
        logger.info("=== DEMO: PDF Monitoring ===")
        
        if not self.rag_system.pdf_monitor:
            return {'demo_type': 'monitoring', 'error': 'PDF monitor not available'}
        
        # Get monitoring statistics
        stats = self.rag_system.pdf_monitor.get_monitoring_stats()
        
        # Get monitored files
        files = self.rag_system.pdf_monitor.get_monitored_files()
        
        # Create a new file to test monitoring
        new_file_path = Path(config.PDF_MONITOR_PATH) / "real_time_test.txt"
        test_content = """
        Real-time Monitoring Test Document
        
        This document was created to test the real-time monitoring capabilities
        of the Agentic RAG System. The system should automatically detect this
        new file and process it for embeddings.
        
        Key Test Points:
        - File creation detection
        - Automatic text processing
        - Embedding generation
        - Index updating
        - Search integration
        
        The monitoring system should have detected this file and made it
        available for search and retrieval within seconds.
        """
        
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Test search for the new content
        search_results = self.rag_system.search_documents("monitoring test", top_k=5)
        new_file_found = any("real_time_test" in result.get('filename', '') for result in search_results)
        
        demo_result = {
            'demo_type': 'monitoring',
            'monitoring_stats': stats,
            'files_monitored': len(files),
            'new_file_created': str(new_file_path),
            'new_file_processed': new_file_found,
            'test_search_results': len(search_results)
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"Monitoring stats: {stats}")
        logger.info(f"Files monitored: {len(files)}")
        logger.info(f"New file processed: {new_file_found}")
        
        return demo_result
    
    def run_complete_demo(self, api_key: str):
        """Run the complete demonstration suite."""
        logger.info("🚀 Starting Complete Agentic RAG System Demonstration")
        logger.info("=" * 80)
        
        # Setup
        if not self.setup_demo(api_key):
            logger.error("Demo setup failed!")
            return False
        
        try:
            # Run all demonstrations
            demos = [
                ("Simple RAG Query", self.demo_simple_query),
                ("Agentic RAG Query", self.demo_agentic_query),
                ("Direct Document Search", self.demo_direct_search),
                ("System Analytics", self.demo_system_analytics),
                ("PDF Monitoring", self.demo_monitoring_capabilities)
            ]
            
            for demo_name, demo_func in demos:
                logger.info(f"\n🎯 Running: {demo_name}")
                try:
                    result = demo_func()
                    logger.info(f"✅ {demo_name} completed successfully")
                except Exception as e:
                    logger.error(f"❌ {demo_name} failed: {e}")
                    continue
                
                time.sleep(1)  # Brief pause between demos
            
            # Generate summary report
            self._generate_demo_report()
            
            logger.info("🎉 Demonstration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            return False
    
    def _generate_demo_report(self):
        """Generate a comprehensive demo report."""
        report_path = Path("rag_demo_report.json")
        
        report = {
            'demo_timestamp': time.time(),
            'system_info': {
                'version': '1.0.0',
                'components': ['Gemini 2.5 Flash', 'A* Search', 'PDF Monitoring', 'Agentic Execution']
            },
            'demo_results': self.demo_results,
            'summary': {
                'total_demos': len(self.demo_results),
                'successful_demos': len([r for r in self.demo_results if 'error' not in r]),
                'demo_types': list(set(r.get('demo_type', 'unknown') for r in self.demo_results))
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report generated: {report_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"DEMO SUMMARY")
        print(f"{'='*80}")
        print(f"Total demonstrations: {report['summary']['total_demos']}")
        print(f"Successful: {report['summary']['successful_demos']}")
        print(f"Demo types: {', '.join(report['summary']['demo_types'])}")
        print(f"Report saved to: {report_path}")


def main():
    """Main demonstration function."""
    print("Agentic RAG System - Comprehensive Demonstration")
    print("=" * 80)
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable or provide API key:")
        api_key = input("Gemini API Key: ").strip()
    
    if not api_key:
        print("Error: API key is required for demonstration")
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    required_packages = [
        'sentence_transformers', 'numpy', 'httpx', 'watchdog',
        'PyPDF2', 'fitz', 'pdfplumber'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All dependencies available")
    
    # Run demonstration
    demo = RAGSystemDemo()
    
    try:
        success = demo.run_complete_demo(api_key)
        if success:
            print("\n🎉 Demonstration completed successfully!")
            print("Check rag_demo_report.json for detailed results.")
        else:
            print("\n❌ Demonstration encountered errors.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemonstration cancelled by user")
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        sys.exit(1)
    finally:
        if demo.rag_system:
            demo.rag_system.shutdown()


if __name__ == "__main__":
    main()