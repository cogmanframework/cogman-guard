"""
Embedding Provider System
Support for multiple embedding providers: Ollama, OpenAI, Gemini, Claude
"""

from .base import EmbeddingProvider, ProviderConfig
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .claude_provider import ClaudeProvider
from .provider_factory import get_provider, list_providers

__all__ = [
    'EmbeddingProvider',
    'ProviderConfig',
    'OllamaProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'ClaudeProvider',
    'get_provider',
    'list_providers'
]

