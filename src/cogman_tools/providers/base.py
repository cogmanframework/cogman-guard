"""
Base Provider Interface for Embedding Models
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np
import torch


class ProviderConfig:
    """Configuration for embedding provider"""
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 32,
        **kwargs  # Additional provider-specific config
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class EmbeddingProvider(ABC):
    """
    Base class for embedding providers
    All providers must implement this interface
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize provider with configuration
        
        Args:
            config: ProviderConfig object with provider settings
        """
        self.config = config
        self.model_name = config.model_name
        self._client = None
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider's client/API connection"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embedding vector
        """
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of input text strings
            
        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this provider
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
    
    def get_embedding_tensor(self, text: str) -> torch.Tensor:
        """
        Get embedding as PyTorch tensor
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor of embedding vector
        """
        embedding = self.get_embedding(text)
        return torch.from_numpy(embedding).float()
    
    def get_embeddings_batch_tensor(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings as PyTorch tensors (batch)
        
        Args:
            texts: List of input text strings
            
        Returns:
            torch.Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = self.get_embeddings_batch(texts)
        return torch.from_numpy(embeddings).float()
    
    def validate_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Validate embedding quality
        
        Returns:
            Dictionary with validation results
        """
        if embedding is None or len(embedding) == 0:
            return {
                'valid': False,
                'error': 'Empty embedding',
                'dimension': 0
            }
        
        dimension = len(embedding)
        has_nan = np.isnan(embedding).any()
        has_inf = np.isinf(embedding).any()
        is_zero = np.allclose(embedding, 0)
        
        return {
            'valid': not (has_nan or has_inf or is_zero),
            'dimension': dimension,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_zero': is_zero,
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding))
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"

