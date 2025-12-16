"""
PDF monitoring and ingestion module.
Watches a directory for new PDF files and automatically ingests them into the RAG system.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
from datetime import datetime
import hashlib


logger = logging.getLogger(__name__)


class PDFIngestionError(Exception):
    """Exception raised when PDF ingestion fails."""
    pass


class PDFHandler(FileSystemEventHandler):
    """Handles file system events for PDF files."""
    
    def __init__(self, pdf_processor, embedding_manager, pdf_monitor):
        """
        Initialize PDF handler.
        
        Args:
            pdf_processor: PDFProcessor instance
            embedding_manager: EmbeddingManager instance
            pdf_monitor: PDFMonitor instance (for callbacks)
        """
        self.pdf_processor = pdf_processor
        self.embedding_manager = embedding_manager
        self.pdf_monitor = pdf_monitor
        
        # Track processed files to avoid duplicate processing
        self._processed_files = set()
        self._processed_files_lock = threading.Lock()
        
        # File change tracking
        self._file_states = {}
        self._file_states_lock = threading.Lock()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._handle_pdf_file(event.src_path, 'created')
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._handle_pdf_file(event.src_path, 'modified')
    
    def _handle_pdf_file(self, file_path: str, event_type: str):
        """
        Handle PDF file events.
        
        Args:
            file_path: Path to the PDF file
            event_type: Type of event ('created' or 'modified')
        """
        file_path = Path(file_path)
        
        # Check if it's a PDF file
        if file_path.suffix.lower() != '.pdf':
            return
        
        # Calculate file hash to detect actual changes
        file_hash = self._calculate_file_hash(file_path)
        
        with self._file_states_lock:
            # Check if file has actually changed
            previous_hash = self._file_states.get(str(file_path))
            if previous_hash == file_hash and event_type == 'modified':
                # File content hasn't changed, ignore
                return
            
            # Update file state
            self._file_states[str(file_path)] = file_hash
        
        # Check if already processed recently
        if self._is_recently_processed(file_path):
            logger.debug(f"File {file_path.name} recently processed, skipping")
            return
        
        logger.info(f"Processing PDF file: {file_path.name} ({event_type})")
        
        # Process the PDF file
        try:
            self._process_pdf_file(file_path)
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path.name}: {e}")
            self.pdf_monitor._notify_ingestion_error(file_path, str(e))
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path.name}: {e}")
            return ""
    
    def _is_recently_processed(self, file_path: Path) -> bool:
        """Check if file was recently processed to avoid duplicates."""
        with self._processed_files_lock:
            file_key = f"{file_path}_{self._calculate_file_hash(file_path)}"
            return file_key in self._processed_files
    
    def _mark_as_processed(self, file_path: Path):
        """Mark file as processed."""
        with self._processed_files_lock:
            file_key = f"{file_path}_{self._calculate_file_hash(file_path)}"
            self._processed_files.add(file_key)
            
            # Limit the size of processed files set
            if len(self._processed_files) > 1000:
                # Remove oldest entries
                processed_list = list(self._processed_files)
                self._processed_files = set(processed_list[-500:])
    
    def _process_pdf_file(self, file_path: Path):
        """Process a single PDF file."""
        try:
            # Extract text from PDF
            extracted_data = self.pdf_processor.extract_text_from_pdf(file_path)
            
            # Process document chunks and generate embeddings
            chunk_ids = self.embedding_manager.process_document(extracted_data)
            
            # Mark as processed
            self._mark_as_processed(file_path)
            
            # Notify monitoring system
            self.pdf_monitor._notify_ingestion_success(file_path, extracted_data, chunk_ids)
            
            logger.info(f"Successfully processed {file_path.name}: {len(chunk_ids)} chunks created")
            
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path.name}: {e}")
            raise


class PDFMonitor:
    """
    Monitors a directory for PDF files and automatically ingests them.
    
    Features:
    - Real-time file system monitoring
    - Automatic PDF text extraction
    - Embedding generation and indexing
    - Change detection and deduplication
    - Error handling and retry logic
    """
    
    def __init__(self, monitor_path: Path, pdf_processor, embedding_manager):
        """
        Initialize PDF monitor.
        
        Args:
            monitor_path: Directory to monitor for PDF files
            pdf_processor: PDFProcessor instance
            embedding_manager: EmbeddingManager instance
        """
        self.monitor_path = Path(monitor_path)
        self.pdf_processor = pdf_processor
        self.embedding_manager = embedding_manager
        
        # Ensure monitor directory exists
        self.monitor_path.mkdir(parents=True, exist_ok=True)
        
        # File system observer
        self.observer = Observer()
        self.handler = PDFHandler(pdf_processor, embedding_manager, self)
        
        # Monitoring state
        self._is_monitoring = False
        self._monitoring_thread = None
        self._monitoring_lock = threading.RLock()
        
        # Event callbacks
        self.ingestion_callbacks = []
        self.error_callbacks = []
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_chunks_created': 0,
            'errors_encountered': 0,
            'monitoring_start_time': None,
            'last_processed_file': None
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"Initialized PDFMonitor for directory: {self.monitor_path}")
    
    def add_ingestion_callback(self, callback: callable):
        """
        Add callback for successful ingestion events.
        
        Args:
            callback: Function to call when files are ingested
        """
        self.ingestion_callbacks.append(callback)
    
    def add_error_callback(self, callback: callable):
        """
        Add callback for ingestion error events.
        
        Args:
            callback: Function to call when ingestion errors occur
        """
        self.error_callbacks.append(callback)
    
    def start_monitoring(self, recursive: bool = True):
        """
        Start monitoring the directory for PDF files.
        
        Args:
            recursive: Whether to monitor subdirectories recursively
        """
        with self._monitoring_lock:
            if self._is_monitoring:
                logger.warning("PDF monitoring is already active")
                return
            
            # Schedule the directory for monitoring
            self.observer.schedule(self.handler, str(self.monitor_path), recursive=recursive)
            
            # Start the observer
            self.observer.start()
            self._is_monitoring = True
            self._monitoring_thread = threading.current_thread()
            
            # Update statistics
            with self.stats_lock:
                self.stats['monitoring_start_time'] = datetime.now()
            
            logger.info(f"Started PDF monitoring on {self.monitor_path}")
    
    def stop_monitoring(self):
        """Stop monitoring the directory."""
        with self._monitoring_lock:
            if not self._is_monitoring:
                logger.warning("PDF monitoring is not active")
                return
            
            # Stop the observer
            self.observer.stop()
            self.observer.join()
            
            self._is_monitoring = False
            self._monitoring_thread = None
            
            logger.info("Stopped PDF monitoring")
    
    def process_existing_files(self):
        """Process all existing PDF files in the monitoring directory."""
        logger.info("Processing existing PDF files")
        
        pdf_files = list(self.monitor_path.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.info("No PDF files found in monitoring directory")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self._process_single_file(pdf_file)
                time.sleep(0.1)  # Small delay to avoid overwhelming the system
            except Exception as e:
                logger.error(f"Failed to process existing file {pdf_file.name}: {e}")
                self._notify_ingestion_error(pdf_file, str(e))
    
    def _process_single_file(self, file_path: Path):
        """Process a single PDF file (non-event based)."""
        try:
            # Extract text from PDF
            extracted_data = self.pdf_processor.extract_text_from_pdf(file_path)
            
            # Process document chunks and generate embeddings
            chunk_ids = self.embedding_manager.process_document(extracted_data)
            
            # Update statistics
            with self.stats_lock:
                self.stats['files_processed'] += 1
                self.stats['total_chunks_created'] += len(chunk_ids)
                self.stats['last_processed_file'] = str(file_path)
            
            # Notify callbacks
            self._notify_ingestion_success(file_path, extracted_data, chunk_ids)
            
            logger.info(f"Processed existing file {file_path.name}: {len(chunk_ids)} chunks")
            
        except Exception as e:
            logger.error(f"Existing file processing failed for {file_path.name}: {e}")
            with self.stats_lock:
                self.stats['errors_encountered'] += 1
            raise
    
    def scan_and_process(self, scan_subdirectories: bool = True):
        """
        Perform a one-time scan and process all PDF files.
        
        Args:
            scan_subdirectories: Whether to scan subdirectories
        """
        logger.info("Starting one-time scan and process")
        
        try:
            # Process existing files first
            self.process_existing_files()
            
            # If monitoring is not active, start it temporarily
            was_monitoring = self._is_monitoring
            if not was_monitoring:
                self.start_monitoring(recursive=scan_subdirectories)
                # Let it run for a short time to catch any files that might be written
                time.sleep(2)
                self.stop_monitoring()
            
            logger.info("One-time scan and process completed")
            
        except Exception as e:
            logger.error(f"One-time scan and process failed: {e}")
            raise
    
    def _notify_ingestion_success(self, file_path: Path, extracted_data: Dict[str, Any], chunk_ids: List[str]):
        """Notify callbacks of successful ingestion."""
        ingestion_event = {
            'event_type': 'ingestion_success',
            'file_path': str(file_path),
            'filename': file_path.name,
            'extracted_data': extracted_data,
            'chunk_ids': chunk_ids,
            'timestamp': datetime.now(),
            'chunks_count': len(chunk_ids),
            'text_length': len(extracted_data.get('text', ''))
        }
        
        for callback in self.ingestion_callbacks:
            try:
                callback(ingestion_event)
            except Exception as e:
                logger.error(f"Ingestion callback failed: {e}")
    
    def _notify_ingestion_error(self, file_path: Path, error_message: str):
        """Notify callbacks of ingestion errors."""
        error_event = {
            'event_type': 'ingestion_error',
            'file_path': str(file_path),
            'filename': file_path.name,
            'error': error_message,
            'timestamp': datetime.now()
        }
        
        for callback in self.error_callbacks:
            try:
                callback(error_event)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary containing monitoring statistics
        """
        with self.stats_lock:
            stats = self.stats.copy()
            
            # Add derived statistics
            if stats['monitoring_start_time']:
                monitoring_duration = datetime.now() - stats['monitoring_start_time']
                stats['monitoring_duration_seconds'] = monitoring_duration.total_seconds()
                stats['files_per_hour'] = (stats['files_processed'] / 
                                         max(1, monitoring_duration.total_seconds() / 3600))
            else:
                stats['monitoring_duration_seconds'] = 0
                stats['files_per_hour'] = 0
            
            stats['is_monitoring'] = self._is_monitoring
            stats['monitoring_path'] = str(self.monitor_path)
            
            return stats
    
    def get_monitored_files(self) -> List[Dict[str, Any]]:
        """
        Get list of files currently being monitored.
        
        Returns:
            List of monitored file information
        """
        monitored_files = []
        
        try:
            pdf_files = list(self.monitor_path.glob("**/*.pdf"))
            
            for pdf_file in pdf_files:
                file_stat = pdf_file.stat()
                monitored_files.append({
                    'path': str(pdf_file),
                    'name': pdf_file.name,
                    'size': file_stat.st_size,
                    'modified_time': datetime.fromtimestamp(file_stat.st_mtime),
                    'created_time': datetime.fromtimestamp(file_stat.st_ctime)
                })
        except Exception as e:
            logger.error(f"Failed to get monitored files: {e}")
        
        return monitored_files
    
    def remove_file_from_monitoring(self, file_path: Path):
        """
        Remove a file from the monitoring system.
        
        Args:
            file_path: Path to the file to remove
        """
        try:
            # Update handler state
            with self.handler._file_states_lock:
                self.handler._file_states.pop(str(file_path), None)
            
            # Remove from processed files
            with self.handler._processed_files_lock:
                processed_files_to_remove = [
                    key for key in self.handler._processed_files 
                    if file_path.name in key
                ]
                for key in processed_files_to_remove:
                    self.handler._processed_files.discard(key)
            
            logger.info(f"Removed {file_path.name} from monitoring")
            
        except Exception as e:
            logger.error(f"Failed to remove file from monitoring: {e}")
    
    def clear_monitoring_data(self):
        """Clear all monitoring data and reset statistics."""
        try:
            # Clear handler state
            with self.handler._file_states_lock:
                self.handler._file_states.clear()
            
            with self.handler._processed_files_lock:
                self.handler._processed_files.clear()
            
            # Reset statistics
            with self.stats_lock:
                self.stats = {
                    'files_processed': 0,
                    'total_chunks_created': 0,
                    'errors_encountered': 0,
                    'monitoring_start_time': None,
                    'last_processed_file': None
                }
            
            logger.info("Cleared all monitoring data")
            
        except Exception as e:
            logger.error(f"Failed to clear monitoring data: {e}")
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        return self._is_monitoring
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
    
    def shutdown(self):
        """Shutdown the monitor and clean up resources."""
        logger.info("Shutting down PDF monitor")
        
        # Stop monitoring if active
        if self._is_monitoring:
            self.stop_monitoring()
        
        # Stop the observer
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        # Clear callbacks
        self.ingestion_callbacks.clear()
        self.error_callbacks.clear()
        
        logger.info("PDF monitor shutdown completed")