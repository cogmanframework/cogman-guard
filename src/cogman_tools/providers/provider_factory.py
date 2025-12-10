"""
Provider Factory for creating embedding providers
"""

from typing import Optional, Dict, Any
from .base import EmbeddingProvider, ProviderConfig
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .claude_provider import ClaudeProvider


# Registry of available providers
PROVIDER_REGISTRY = {
    'ollama': OllamaProvider,
    'openai': OpenAIProvider,
    'gemini': GeminiProvider,
    'claude': ClaudeProvider,
}


def get_provider(
    provider_name: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider
    
    Args:
        provider_name: Name of provider ('ollama', 'openai', 'gemini', 'claude')
        model_name: Name of the embedding model
        api_key: API key (required for openai, gemini, claude)
        base_url: Base URL for API (optional, for custom endpoints)
        **kwargs: Additional provider-specific configuration
        
    Returns:
        EmbeddingProvider instance
        
    Examples:
        >>> # Ollama (local)
        >>> provider = get_provider('ollama', 'nomic-embed-text')
        
        >>> # OpenAI
        >>> provider = get_provider('openai', 'text-embedding-3-small', api_key='sk-...')
        
        >>> # Gemini
        >>> provider = get_provider('gemini', 'text-embedding-004', api_key='AIza...')
        
        >>> # Claude
        >>> provider = get_provider('claude', 'claude-embedding', api_key='sk-ant-...')
    """
    provider_name = provider_name.lower()
    
    if provider_name not in PROVIDER_REGISTRY:
        available = ', '.join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {available}"
        )
    
    ProviderClass = PROVIDER_REGISTRY[provider_name]
    
    # Create config
    config = ProviderConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    # Create and return provider
    return ProviderClass(config)


def list_providers() -> Dict[str, Any]:
    """
    List all available providers and their information
    
    Returns:
        Dictionary mapping provider names to their classes and info
    """
    return {
        name: {
            'class': provider_class,
            'name': name,
            'description': provider_class.__doc__ or f"{name} embedding provider"
        }
        for name, provider_class in PROVIDER_REGISTRY.items()
    }


def register_provider(name: str, provider_class: type):
    """
    Register a custom provider
    
    Args:
        name: Provider name
        provider_class: Provider class (must inherit from EmbeddingProvider)
    """
    if not issubclass(provider_class, EmbeddingProvider):
        raise TypeError(f"Provider class must inherit from EmbeddingProvider")
    
    PROVIDER_REGISTRY[name.lower()] = provider_class

