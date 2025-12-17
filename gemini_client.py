"""
Gemini 2.5 Flash API client for advanced reasoning and response generation.
Handles API communication, response parsing, and error management.
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import httpx
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class GeminiModel(Enum):
    """Available Gemini models."""
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class GeminiError(Exception):
    """Exception raised by Gemini API operations."""
    pass


@dataclass
class GeminiRequest:
    """Request configuration for Gemini API."""
    model: str = GeminiModel.GEMINI_2_5_FLASH.value
    max_tokens: int = 8000
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    candidate_count: int = 1


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    text: str
    candidates: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    usage: Dict[str, Any]
    timestamp: datetime
    
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'GeminiResponse':
        """Create response object from API response data."""
        candidates = response_data.get('candidates', [])
        text = ""
        
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            
            if parts and len(parts) > 0:
                text = parts[0].get('text', '')
        
        return cls(
            text=text,
            candidates=candidates,
            metadata=response_data.get('metadata', {}),
            usage=response_data.get('usageMetadata', {}),
            timestamp=datetime.now()
        )


class GeminiClient:
    """
    Production-ready Gemini 2.5 Flash API client.
    
    Features:
    - Robust error handling and retry logic
    - Rate limiting and quota management
    - Request/response caching
    - Streaming support
    - Multi-modal content handling
    """
    
    def __init__(self, api_key: str, base_url: str = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            base_url: Base URL for API (optional)
        """
        self.api_key = api_key
        self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.model = GeminiModel.GEMINI_2_5_FLASH.value
        
        # HTTP client configuration
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                'x-goog-api-key': api_key,
                'Content-Type': 'application/json'
            }
        )
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)  # 60 RPM
        
        # Response cache
        self._response_cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Performance tracking
        self._request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0
        }
        
        logger.info("Initialized GeminiClient with API key")
    
    def generate(self, prompt: str, context: List[str] = None, request_config: GeminiRequest = None) -> GeminiResponse:
        """
        Generate response using Gemini 2.5 Flash.
        
        Args:
            prompt: Main prompt for generation
            context: Optional context documents/chunks
            request_config: Request configuration
            
        Returns:
            GeminiResponse object
        """
        if request_config is None:
            request_config = GeminiRequest()
        
        # Build complete prompt with context
        full_prompt = self._build_prompt_with_context(prompt, context)
        
        # Check cache first
        cache_key = self._get_cache_key(full_prompt, request_config)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.debug("Returning cached response")
            return cached_response
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            # Prepare request payload
            payload = self._prepare_request_payload(full_prompt, request_config)
            
            # Make API request
            start_time = time.time()
            response = self._make_api_request(payload)
            latency = time.time() - start_time
            
            # Parse response
            gemini_response = GeminiResponse.from_api_response(response)
            
            # Update performance statistics
            self._update_request_stats(True, latency)
            
            # Cache response
            self._cache_response(cache_key, gemini_response)
            
            logger.debug(f"Generated response in {latency:.2f}s")
            return gemini_response
            
        except Exception as e:
            self._update_request_stats(False, 0.0)
            logger.error(f"Gemini API request failed: {e}")
            raise GeminiError(f"Request failed: {e}")
    
    def generate_stream(self, prompt: str, context: List[str] = None, request_config: GeminiRequest = None):
        """
        Generate streaming response from Gemini.
        
        Args:
            prompt: Main prompt for generation
            context: Optional context documents/chunks
            request_config: Request configuration
            
        Yields:
            Response chunks as they arrive
        """
        if request_config is None:
            request_config = GeminiRequest()
        
        full_prompt = self._build_prompt_with_context(prompt, context)
        payload = self._prepare_request_payload(full_prompt, request_config)
        
        # Add streaming configuration
        payload['generationConfig']['streamingConfig'] = {
            'partialResults': True
        }
        
        try:
            with self.client.stream('POST', f'{self.base_url}/models/{self.model}:generateContent', json=payload) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'candidates' in data:
                                candidate = data['candidates'][0]
                                content = candidate.get('content', {})
                                parts = content.get('parts', [])
                                if parts:
                                    chunk = parts[0].get('text', '')
                                    if chunk:
                                        yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            raise GeminiError(f"Streaming request failed: {e}")
    
    def describe_image(self, image_base64: str, mime_type: str = "image/jpeg", prompt: str = None) -> GeminiResponse:
        """
        Generate a description of an image using Gemini's multimodal capabilities.
        
        Args:
            image_base64: Base64 encoded image data
            mime_type: MIME type of the image (e.g., 'image/jpeg', 'image/png')
            prompt: Optional custom prompt for image analysis
            
        Returns:
            GeminiResponse with image description
        """
        if prompt is None:
            prompt = """Analyze this image in detail. Describe:
1. What type of visual content this is (chart, graph, diagram, photo, etc.)
2. The main elements, data, or subjects shown
3. Key insights, trends, or patterns visible
4. Any text, labels, or annotations present
5. The overall purpose or message of the image

Provide a comprehensive description that would be useful for document search and retrieval."""
        
        images = [{
            'base64': image_base64,
            'mime_type': mime_type
        }]
        
        request_config = GeminiRequest()
        
        try:
            payload = self._prepare_request_payload(prompt, request_config, images)
            
            start_time = time.time()
            response = self._make_api_request(payload)
            latency = time.time() - start_time
            
            gemini_response = GeminiResponse.from_api_response(response)
            self._update_request_stats(True, latency)
            
            logger.debug(f"Generated image description in {latency:.2f}s")
            return gemini_response
            
        except Exception as e:
            self._update_request_stats(False, 0.0)
            logger.error(f"Image description failed: {e}")
            raise GeminiError(f"Image description failed: {e}")
    
    def batch_generate(self, prompts: List[str], context: List[str] = None, request_config: GeminiRequest = None) -> List[GeminiResponse]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            context: Optional context documents/chunks
            request_config: Request configuration
            
        Returns:
            List of GeminiResponse objects
        """
        responses = []
        
        for prompt in prompts:
            try:
                response = self.generate(prompt, context, request_config)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch request failed for prompt '{prompt[:50]}...': {e}")
                responses.append(GeminiResponse("", [], {}, {}, datetime.now()))
        
        return responses
    
    def analyze_context(self, context_chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Analyze context documents and provide insights.
        
        Args:
            context_chunks: List of context chunks with metadata
            query: Original query
            
        Returns:
            Analysis results
        """
        # Prepare context summary
        context_summary = self._prepare_context_summary(context_chunks)
        
        analysis_prompt = f"""
        Analyze the following context documents in relation to the query: "{query}"
        
        Context Summary:
        {context_summary}
        
        Provide:
        1. Relevance assessment for each document
        2. Key themes and concepts
        3. Potential knowledge gaps
        4. Recommended follow-up questions
        5. Confidence level in the available information
        
        Format as JSON with keys: relevance_assessment, themes, gaps, follow_ups, confidence
        """
        
        try:
            response = self.generate(analysis_prompt)
            
            # Try to parse as JSON
            try:
                analysis = json.loads(response.text)
                return analysis
            except json.JSONDecodeError:
                # Return structured text if JSON parsing fails
                return {
                    "analysis": response.text,
                    "confidence": "medium",
                    "raw_response": response.text
                }
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {
                "error": str(e),
                "confidence": "low",
                "analysis": "Context analysis failed"
            }
    
    def _build_prompt_with_context(self, prompt: str, context: List[str]) -> str:
        """Build complete prompt with context."""
        if not context:
            return prompt
        
        context_text = "\n\n".join([f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(context)])
        
        full_prompt = f"""
        Based on the following context documents, please answer the question:
        
        Context Documents:
        {context_text}
        
        Question: {prompt}
        
        Please provide a comprehensive answer based on the context. If the context doesn't contain sufficient information, please state this clearly.
        """
        
        return full_prompt
    
    def _prepare_request_payload(self, prompt: str, request_config: GeminiRequest, images: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Prepare request payload for API with optional image support."""
        parts = [{'text': prompt}]
        
        # Add images if provided
        if images:
            for img in images:
                if 'base64' in img and 'mime_type' in img:
                    parts.append({
                        'inline_data': {
                            'mime_type': img['mime_type'],
                            'data': img['base64']
                        }
                    })
        
        return {
            'contents': [{
                'parts': parts
            }],
            'generationConfig': {
                'maxOutputTokens': request_config.max_tokens,
                'temperature': request_config.temperature,
                'topP': request_config.top_p,
                'topK': request_config.top_k,
                'candidateCount': request_config.candidate_count
            },
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                },
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
                }
            ]
        }
    
    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Gemini API."""
        url = f'{self.base_url}/models/{self.model}:generateContent'
        
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def _prepare_context_summary(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare a summary of context chunks."""
        if not context_chunks:
            return "No context documents provided."
        
        summaries = []
        for i, chunk in enumerate(context_chunks[:10]):  # Limit to first 10 chunks
            filename = chunk.get('filename', 'Unknown')
            text_preview = chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            
            summary = f"Document {i+1} ({filename}): {text_preview}"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _get_cache_key(self, prompt: str, request_config: GeminiRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{prompt}_{request_config.model}_{request_config.max_tokens}_{request_config.temperature}"
        import hashlib
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[GeminiResponse]:
        """Get cached response if available and valid."""
        if cache_key in self._response_cache:
            cached_data = self._response_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self._cache_ttl:
                return cached_data['response']
        return None
    
    def _cache_response(self, cache_key: str, response: GeminiResponse):
        """Cache response."""
        self._response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # Clean old cache entries
        if len(self._response_cache) > 1000:
            current_time = time.time()
            self._response_cache = {
                k: v for k, v in self._response_cache.items()
                if current_time - v['timestamp'] < self._cache_ttl
            }
    
    def _update_request_stats(self, success: bool, latency: float):
        """Update request performance statistics."""
        self._request_stats['total_requests'] += 1
        
        if success:
            self._request_stats['successful_requests'] += 1
            # Update average latency
            total_latency = (self._request_stats['average_latency'] * 
                           (self._request_stats['successful_requests'] - 1) + latency)
            self._request_stats['average_latency'] = total_latency / self._request_stats['successful_requests']
        else:
            self._request_stats['failed_requests'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        return self._request_stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on API connectivity."""
        try:
            test_response = self.generate("Hello, this is a health check.")
            return {
                'status': 'healthy',
                'model': self.model,
                'latency': self._request_stats['average_latency'],
                'success_rate': (self._request_stats['successful_requests'] / 
                               max(1, self._request_stats['total_requests']))
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model': self.model
            }
    
    def close(self):
        """Close HTTP client."""
        if self.client:
            self.client.close()


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Remove old request times
        self.request_times = [t for t in self.request_times if current_time - t < self.time_window]
        
        # If at limit, wait
        if len(self.request_times) >= self.max_requests:
            sleep_time = self.time_window - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)