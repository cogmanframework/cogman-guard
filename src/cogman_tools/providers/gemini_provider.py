"""
Google Gemini Embedding Provider
"""

import numpy as np
from typing import List, Optional
from .base import EmbeddingProvider, ProviderConfig

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None


class GeminiProvider(EmbeddingProvider):
    """
    Provider for Google Gemini embedding models
    Supports text-embedding-004, embedding-001, etc.
    """
    
    def __init__(self, config: ProviderConfig):
        if not HAS_GEMINI:
            raise ImportError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            )
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Gemini requires api_key in config")
        self._embedding_dim = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        genai.configure(api_key=self.config.api_key)
        self._client = genai
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Gemini"""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0]
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from Gemini (batch)
        
        Note: Gemini may process sequentially depending on API
        """
        embeddings = []
        
        for text in texts:
            try:
                # Use the embedding model
                result = self._client.embed_content(
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type="retrieval_document"  # or "retrieval_query", "semantic_similarity", etc.
                )
                
                embedding = np.array(result['embedding'])
                
                if len(embedding) == 0:
                    raise ValueError(f"Empty embedding from Gemini model {self.model_name}")
                
                embeddings.append(embedding)
                
            except Exception as e:
                raise RuntimeError(f"Gemini API error: {e}")
        
        if not embeddings:
            raise ValueError("No embeddings generated")
        
        # Cache dimension
        if self._embedding_dim is None:
            self._embedding_dim = len(embeddings[0])
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self._embedding_dim is None:
            # Get dimension by making a test call
            test_embedding = self.get_embedding("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

