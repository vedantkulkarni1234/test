# Agentic RAG System with Gemini 2.5 Flash & A* Search

## System Architecture

This production-ready RAG system implements:
- **Gemini 2.5 Flash API** for advanced reasoning and response generation
- **A* Search Algorithm** for ultra-fast document retrieval and ranking
- **Agentic Task Execution** based on retrieved knowledge
- **Local PDF Monitoring** with automatic ingestion and indexing
- **No Cloud Dependencies** - fully local vector operations

## Components

### Core Modules
- `pdf_monitor.py` - File system watcher for PDF ingestion
- `pdf_parser.py` - Text extraction from PDF files
- `embedding_manager.py` - Local embedding generation and storage
- `astar_retriever.py` - A* search implementation for document ranking
- `gemini_client.py` - Gemini 2.5 Flash API integration
- `agent_executor.py` - Agentic task planning and execution
- `rag_pipeline.py` - Main RAG orchestration
- `config.py` - Configuration management

### Data Flow
1. PDF files detected in monitored directory
2. Text extracted and embedded locally
3. A* search ranks documents by relevance
4. Top documents routed to Gemini for reasoning
5. Agentic tasks planned and executed
6. Results integrated into final responses

## Usage

```python
from rag_system import AgenticRAGSystem

# Initialize system
rag = AgenticRAGSystem(
    pdf_directory="knowledge_hub",
    gemini_api_key="your_api_key"
)

# Process a query with agentic execution
result = rag.process_query("Analyze market trends and create a strategy report")
```

## Glass Box A* Visualization (Streamlit)

This repo includes a Streamlit UI that visualizes the document graph and animates which document nodes the A* retriever visits.

```bash
streamlit run glass_box_app.py
```

## Configuration

Environment variables:
- `GEMINI_API_KEY` - Your Gemini API key
- `PDF_MONITOR_PATH` - Directory to monitor for PDFs (default: ./knowledge_hub)
- `EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)

## Production Features

- Continuous PDF monitoring
- Automatic re-indexing on file changes
- Configurable A* search parameters
- Extensible agent task framework
- Comprehensive error handling
- Performance metrics and logging