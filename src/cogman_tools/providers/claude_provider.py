"""
Anthropic Claude Embedding Provider
"""

import numpy as np
from typing import List, Optional
from .base import EmbeddingProvider, ProviderConfig

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None


class ClaudeProvider(EmbeddingProvider):
    """
    Provider for Anthropic Claude embedding models
    Note: Claude API may not have dedicated embedding endpoints,
    this provider uses message API with embedding extraction
    """
    
    def __init__(self, config: ProviderConfig):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Claude requires api_key in config")
        self._embedding_dim = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        self._client = Anthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Claude"""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0]
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from Claude
        
        Note: As of current Anthropic API, Claude may not have dedicated embedding endpoints.
        This implementation checks for embedding capabilities and provides a fallback.
        """
        embeddings = []
        
        for text in texts:
            try:
                # Check if Anthropic has embedding API
                # This may vary based on API version and availability
                
                # Option 1: If Anthropic adds embedding endpoint (future)
                # try:
                #     response = self._client.embeddings.create(
                #         model=self.model_name,
                #         input=text
                #     )
                #     embedding = np.array(response.embedding)
                # except AttributeError:
                #     # Fallback if not available
                #     pass
                
                # Option 2: Use message API with embedding extraction (if available)
                # This is a placeholder - adjust based on actual API
                
                # For now, raise informative error
                raise NotImplementedError(
                    f"Claude embedding API for model '{self.model_name}' is not yet implemented. "
                    f"Please check Anthropic documentation for embedding endpoints. "
                    f"If embeddings are available, update this provider accordingly."
                )
                
            except NotImplementedError:
                raise
            except Exception as e:
                raise RuntimeError(f"Claude API error: {e}")
        
        if not embeddings:
            raise ValueError("No embeddings generated")
        
        # Cache dimension
        if self._embedding_dim is None and embeddings:
            self._embedding_dim = len(embeddings[0])
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self._embedding_dim is None:
            # Get dimension by making a test call
            test_embedding = self.get_embedding("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

