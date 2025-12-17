"""
Demonstration script for multimodal RAG features.
Shows PDF processing with image extraction, Gemini-based image description, and image search.
"""

import logging
import os
from pathlib import Path
from rag_pipeline import AgenticRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    print("="*80)
    print("Multimodal RAG System Demonstration")
    print("="*80)
    print()
    
    # Check for API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        print("Please set it with: export GEMINI_API_KEY='your_api_key'")
        return
    
    # Create knowledge hub directory
    knowledge_hub = Path("./knowledge_hub_multimodal")
    knowledge_hub.mkdir(exist_ok=True)
    
    print(f"Knowledge Hub: {knowledge_hub.absolute()}")
    print()
    
    # Initialize RAG System with multimodal support
    print("Initializing Multimodal RAG System...")
    rag = AgenticRAGSystem(
        gemini_api_key=gemini_api_key,
        pdf_directory=knowledge_hub,
        enable_monitoring=False  # Manual processing for demo
    )
    print("✓ System initialized successfully")
    print()
    
    # Demonstrate features
    print("-"*80)
    print("Feature 1: PDF Processing with Image Extraction")
    print("-"*80)
    print()
    
    print("Place PDF files with charts/graphs in:", knowledge_hub)
    print("The system will:")
    print("  1. Extract text from PDFs")
    print("  2. Extract images (charts, graphs, diagrams)")
    print("  3. Generate AI descriptions for images using Gemini 2.5 Flash")
    print("  4. Create embeddings for both text and image descriptions")
    print()
    
    # Check for existing PDFs
    pdf_files = list(knowledge_hub.glob("*.pdf"))
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name}")
        print()
        
        # Process PDFs
        print("Processing PDFs...")
        for pdf_file in pdf_files:
            try:
                # Extract text and images
                extracted_data = rag.pdf_processor.extract_text_from_pdf(pdf_file)
                
                print(f"\n✓ Processed: {pdf_file.name}")
                print(f"  - Text: {len(extracted_data['text'])} characters")
                print(f"  - Text Chunks: {len(extracted_data['chunks'])}")
                print(f"  - Images: {len(extracted_data.get('images', []))}")
                
                # Show image details
                if extracted_data.get('images'):
                    print(f"\n  Image Details:")
                    for img in extracted_data['images']:
                        print(f"    • Page {img['page_number']}: {img['width']}x{img['height']}px ({img['format']})")
                        print(f"      Saved to: {img['file_path']}")
                
                # Process with embeddings and generate image descriptions
                print(f"\n  Generating embeddings and image descriptions...")
                chunk_ids, image_ids = rag.embedding_manager.process_document(
                    extracted_data,
                    rag.gemini_client
                )
                
                print(f"  ✓ Created {len(chunk_ids)} text embeddings")
                print(f"  ✓ Processed {len(image_ids)} images with AI descriptions")
                
                # Show image descriptions
                if image_ids:
                    print(f"\n  Image Descriptions:")
                    for image_id in image_ids:
                        image_metadata = rag.embedding_manager.vector_index.get_image_metadata(image_id)
                        if image_metadata and image_metadata.get('description'):
                            desc = image_metadata['description'][:150]
                            print(f"    • {image_id}: {desc}...")
                
            except Exception as e:
                print(f"  ✗ Error processing {pdf_file.name}: {e}")
        
        print()
        print("-"*80)
        print("Feature 2: Image Search")
        print("-"*80)
        print()
        
        # Demonstrate image search
        search_queries = [
            "chart showing trends",
            "graph with data",
            "diagram or flowchart",
            "financial report visualization"
        ]
        
        print("Searching for images using natural language queries...")
        print()
        
        for query in search_queries:
            print(f"Query: \"{query}\"")
            results = rag.search_images(query, top_k=3)
            
            if results:
                print(f"  Found {len(results)} matching images:")
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['filename']} (Page {result['page_number']})")
                    print(f"       Similarity: {result['similarity']:.3f}")
                    print(f"       Path: {result['image_path']}")
                    if result.get('description'):
                        desc = result['description'][:100]
                        print(f"       Description: {desc}...")
                print()
            else:
                print("  No matching images found")
                print()
        
        print("-"*80)
        print("Feature 3: Specific Image Retrieval")
        print("-"*80)
        print()
        
        # Demonstrate specific image retrieval
        specific_queries = [
            "Show me the trend graph from the Q3 report",
            "Find the pie chart showing market share",
            "Display the organizational diagram"
        ]
        
        for query in specific_queries:
            print(f"Request: \"{query}\"")
            image = rag.get_image_by_description(query)
            
            if image:
                print(f"  ✓ Found: {image['filename']} (Page {image['page_number']})")
                print(f"    Similarity Score: {image['similarity']:.3f}")
                print(f"    Image Path: {image['image_path']}")
                print(f"    Size: {image['width']}x{image['height']}px")
                print(f"    Description: {image.get('description', 'N/A')[:150]}...")
            else:
                print("  No matching image found")
            print()
        
    else:
        print("No PDF files found in knowledge hub.")
        print()
        print("To test the multimodal features:")
        print(f"  1. Place PDF files containing charts/graphs in: {knowledge_hub}")
        print("  2. Run this demo script again")
        print()
    
    print("-"*80)
    print("Feature 4: Combined Text and Image Search")
    print("-"*80)
    print()
    
    print("The system can now search across both:")
    print("  • Text content from PDFs")
    print("  • AI-generated descriptions of images")
    print()
    print("Try queries like:")
    print("  - 'What are the market trends?' (finds both text and relevant charts)")
    print("  - 'Show revenue growth' (finds financial data and visualizations)")
    print("  - 'Organizational structure' (finds text descriptions and org charts)")
    print()
    
    print("="*80)
    print("Demonstration Complete!")
    print("="*80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ PDF text extraction")
    print("  ✓ Automatic image extraction from PDFs")
    print("  ✓ AI-powered image description using Gemini 2.5 Flash")
    print("  ✓ Semantic search across images using descriptions")
    print("  ✓ Multimodal RAG combining text and visual content")
    print()
    print("Next Steps:")
    print("  • Add more PDFs with charts and diagrams")
    print("  • Use rag.search_images() to find specific visualizations")
    print("  • Combine text queries with image results for comprehensive answers")
    print()


if __name__ == "__main__":
    main()
