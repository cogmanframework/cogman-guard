"""
Ollama Embedding Provider
"""

import numpy as np
from typing import List, Optional
import requests
from .base import EmbeddingProvider, ProviderConfig


class OllamaProvider(EmbeddingProvider):
    """
    Provider for Ollama embedding models
    Supports local and remote Ollama instances
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self._embedding_dim = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama connection"""
        # Ollama uses REST API, no special client needed
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama"""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0]
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from Ollama (batch)
        
        Note: Ollama may not support true batch, so we process sequentially
        """
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()
                embedding = np.array(data.get("embedding", []))
                
                if len(embedding) == 0:
                    raise ValueError(f"Empty embedding from Ollama model {self.model_name}")
                
                embeddings.append(embedding)
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama API error: {e}")
        
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

