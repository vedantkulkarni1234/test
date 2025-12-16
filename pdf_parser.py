"""
PDF text extraction and preprocessing module.
Handles various PDF formats and extracts structured text for embedding.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import PyPDF2
import fitz  # PyMuPDF
from pdfplumber import PDF as PDFPlumber


logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Exception raised when PDF parsing fails."""
    pass


class PDFProcessor:
    """Advanced PDF text extraction and preprocessing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./extracted_texts")
        self.output_dir.mkdir(exist_ok=True)
        
        # Text preprocessing patterns
        self.cleaning_patterns = [
            (r'\x00', ''),  # Remove null bytes
            (r'[^\x20-\x7E\n\r\t]', ' '),  # Keep only printable ASCII
            (r'\s+', ' '),  # Normalize whitespace
            (r'\n\s*\n', '\n'),  # Remove empty lines
            (r'Page \d+', ''),  # Remove page numbers
            (r'\d+\s*seconds?', 'time'),  # Normalize time references
        ]
        
        # Section detection patterns
        self.section_patterns = {
            'title': r'^(?:Title|Article|Chapter)\s*:?\s*(.+)$',
            'abstract': r'^(?:Abstract|Summary|Overview)\s*:?\s*(.+)$',
            'introduction': r'^(?:Introduction|Background)\s*:?\s*(.+)$',
            'conclusion': r'^(?:Conclusion|Results|Discussion)\s*:?\s*(.+)$',
            'references': r'^(?:References|Bibliography)\s*:?\s*(.+)$',
            'author': r'^(?:Author|By)\s*:?\s*(.+)$',
            'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file using multiple parsers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        if not pdf_path.exists():
            raise PDFParseError(f"PDF file does not exist: {pdf_path}")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(pdf_path)
        
        # Extract using multiple methods and combine results
        extracted_data = {
            'file_path': str(pdf_path),
            'file_hash': file_hash,
            'filename': pdf_path.name,
            'extracted_at': datetime.now().isoformat(),
            'text': '',
            'chunks': [],
            'metadata': {},
            'sections': {},
            'processing_stats': {}
        }
        
        try:
            # Method 1: PyMuPDF (fitz) - best for formatted text
            pymupdf_result = self._extract_with_pymupdf(pdf_path)
            extracted_data.update(pymupdf_result)
            
            # Method 2: pdfplumber - good for tables and structured data
            if not extracted_data['text'] or len(extracted_data['text']) < 100:
                pdfplumber_result = self._extract_with_pdfplumber(pdf_path)
                if pdfplumber_result['text']:
                    extracted_data['text'] = pdfplumber_result['text']
                    extracted_data['metadata'].update(pdfplumber_result['metadata'])
            
            # Method 3: PyPDF2 - fallback
            if not extracted_data['text'] or len(extracted_data['text']) < 100:
                pypdf2_result = self._extract_with_pypdf2(pdf_path)
                if pypdf2_result['text']:
                    extracted_data['text'] = pypdf2_result['text']
                    extracted_data['metadata'].update(pypdf2_result['metadata'])
            
            # Clean and preprocess the extracted text
            extracted_data['text'] = self._clean_text(extracted_data['text'])
            
            # Extract sections
            extracted_data['sections'] = self._extract_sections(extracted_data['text'])
            
            # Chunk the text for embedding
            extracted_data['chunks'] = self._chunk_text(extracted_data['text'])
            
            # Update processing stats
            extracted_data['processing_stats'] = {
                'total_chars': len(extracted_data['text']),
                'total_chunks': len(extracted_data['chunks']),
                'extraction_methods': [k for k, v in [
                    ('pymupdf', len(pymupdf_result['text']) > 0),
                    ('pdfplumber', 'text' in locals() and len(extracted_data['text']) > len(pymupdf_result['text'])),
                    ('pypdf2', 'text' in locals() and len(extracted_data['text']) > len(pymupdf_result.get('text', '')))
                ] if v]
            }
            
            logger.info(f"Successfully extracted {len(extracted_data['text'])} characters from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
            raise PDFParseError(f"Text extraction failed: {e}")
        
        return extracted_data
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
                # Extract page metadata
                if page_num == 0:
                    page_metadata = doc.metadata
                    metadata.update({
                        'title': page_metadata.get('title', ''),
                        'author': page_metadata.get('author', ''),
                        'subject': page_metadata.get('subject', ''),
                        'creator': page_metadata.get('creator', '')
                    })
            
            doc.close()
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
            return {'text': '', 'metadata': {}}
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using pdfplumber."""
        try:
            text = ""
            metadata = {}
            
            with PDFPlumber(pdf_path) as pdf:
                metadata.update({
                    'title': getattr(pdf.metadata, 'Title', ''),
                    'author': getattr(pdf.metadata, 'Author', ''),
                    'subject': getattr(pdf.metadata, 'Subject', ''),
                    'creator': getattr(pdf.metadata, 'Creator', '')
                })
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed for {pdf_path.name}: {e}")
            return {'text': '', 'metadata': {}}
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyPDF2."""
        try:
            text = ""
            metadata = {}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', '')
                    })
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return {'text': '', 'metadata': {}}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace and normalize
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract structured sections from text."""
        sections = {}
        lines = text.split('\n')
        current_section = 'body'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            section_found = False
            for section_name, pattern in self.section_patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = current_content.copy()
                    
                    # Start new section
                    current_section = section_name
                    current_content = [match.group(1) if match.groups() else line]
                    section_found = True
                    break
            
            if not section_found:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = current_content
        
        return sections
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for embedding."""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for i, word in enumerate(words):
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': len(chunks),
                    'text': chunk_text,
                    'start_word': i - len(current_chunk),
                    'end_word': i,
                    'size': current_size
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words
                current_size = sum(len(w) + 1 for w in current_chunk) - 1
                
                # Add current word
                current_chunk.append(word)
                current_size += word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': len(chunks),
                'text': chunk_text,
                'start_word': len(words) - len(current_chunk),
                'end_word': len(words),
                'size': current_size
            })
        
        return chunks
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def save_extracted_text(self, extracted_data: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
        """Save extracted text to JSON file."""
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(exist_ok=True)
        
        # Create filename based on original PDF name
        pdf_name = Path(extracted_data['filename']).stem
        output_file = output_dir / f"{pdf_name}_{extracted_data['file_hash'][:8]}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted text to {output_file}")
        return output_file
    
    def load_extracted_text(self, json_file: Path) -> Dict[str, Any]:
        """Load extracted text from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)