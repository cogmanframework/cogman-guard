"""
OpenAI Embedding Provider
"""

import numpy as np
from typing import List, Optional
from .base import EmbeddingProvider, ProviderConfig

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class OpenAIProvider(EmbeddingProvider):
    """
    Provider for OpenAI embedding models
    Supports text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large, etc.
    """
    
    def __init__(self, config: ProviderConfig):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI requires api_key in config")
        self._embedding_dim = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,  # For Azure OpenAI or custom endpoints
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI"""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0]
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from OpenAI (batch)
        
        OpenAI supports batch processing natively
        """
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = []
            for item in response.data:
                embedding = np.array(item.embedding)
                embeddings.append(embedding)
            
            if not embeddings:
                raise ValueError(f"No embeddings returned from OpenAI model {self.model_name}")
            
            # Cache dimension
            if self._embedding_dim is None:
                self._embedding_dim = len(embeddings[0])
            
            return np.array(embeddings)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self._embedding_dim is None:
            # Get dimension by making a test call
            test_embedding = self.get_embedding("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

