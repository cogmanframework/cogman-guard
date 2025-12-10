"""
Cogman Tools - Operational Analysis Tools for Embeddings & AI Systems
"""

from .embedding_inspector import EmbeddingQualityInspector, EmbeddingPhysicsInspector
from .behavioral_analyzer import BehavioralAnalyzer, OperationalStatus
from .eimas_analyzer import EIMASAnalyzer, Alert, EmbeddingLineage
from .providers import (
    EmbeddingProvider,
    ProviderConfig,
    OllamaProvider,
    OpenAIProvider,
    GeminiProvider,
    ClaudeProvider,
    get_provider,
    list_providers
)

__version__ = "0.1.0"

__all__ = [
    'EmbeddingQualityInspector',
    'EmbeddingPhysicsInspector',  # Backward compatibility
    'BehavioralAnalyzer',
    'OperationalStatus',
    'EIMASAnalyzer',
    'Alert',
    'EmbeddingLineage',
    # Providers
    'EmbeddingProvider',
    'ProviderConfig',
    'OllamaProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'ClaudeProvider',
    'get_provider',
    'list_providers'
]

