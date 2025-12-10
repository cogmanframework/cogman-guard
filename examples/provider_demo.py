"""
Demo: Using Embedding Providers
Examples for Ollama, OpenAI, Gemini, and Claude
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cogman_tools import get_provider, EmbeddingQualityInspector


def demo_ollama():
    """Demo: Ollama Provider"""
    print("=" * 60)
    print("Ollama Provider Demo")
    print("=" * 60)
    
    try:
        # Create Ollama provider (local instance)
        provider = get_provider(
            provider_name='ollama',
            model_name='nomic-embed-text',  # or 'all-minilm', etc.
            base_url='http://localhost:11434'  # Default Ollama URL
        )
        
        print(f"✅ Provider created: {provider}")
        print(f"   Model: {provider.model_name}")
        print(f"   Dimension: {provider.get_embedding_dimension()}")
        
        # Get embedding
        text = "Hello, this is a test embedding"
        embedding = provider.get_embedding(text)
        print(f"\n✅ Generated embedding:")
        print(f"   Shape: {embedding.shape}")
        print(f"   Mean: {embedding.mean():.4f}")
        print(f"   Std: {embedding.std():.4f}")
        
        # Analyze with EmbeddingQualityInspector
        inspector = EmbeddingQualityInspector()
        result = inspector.analyze_embedding(embedding)
        print(f"\n📊 Quality Analysis:")
        print(f"   EQI: {result.get('embedding_quality_index', 'N/A'):.3f}")
        print(f"   Information Strength: {result.get('information_strength', 'N/A'):.0f}")
        
        return provider, embedding
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return None, None


def demo_openai():
    """Demo: OpenAI Provider"""
    print("\n" + "=" * 60)
    print("OpenAI Provider Demo")
    print("=" * 60)
    
    try:
        # Get API key from environment or config
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI demo.")
            return None, None
        
        # Create OpenAI provider
        provider = get_provider(
            provider_name='openai',
            model_name='text-embedding-3-small',  # or 'text-embedding-ada-002', etc.
            api_key=api_key
        )
        
        print(f"✅ Provider created: {provider}")
        print(f"   Model: {provider.model_name}")
        print(f"   Dimension: {provider.get_embedding_dimension()}")
        
        # Get embeddings (batch)
        texts = [
            "This is the first text",
            "This is the second text",
            "This is the third text"
        ]
        embeddings = provider.get_embeddings_batch(texts)
        print(f"\n✅ Generated embeddings (batch):")
        print(f"   Shape: {embeddings.shape}")
        
        # Analyze first embedding
        inspector = EmbeddingQualityInspector()
        result = inspector.analyze_embedding(embeddings[0])
        print(f"\n📊 Quality Analysis (first embedding):")
        print(f"   EQI: {result.get('embedding_quality_index', 'N/A'):.3f}")
        
        return provider, embeddings
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install openai")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def demo_gemini():
    """Demo: Gemini Provider"""
    print("\n" + "=" * 60)
    print("Gemini Provider Demo")
    print("=" * 60)
    
    try:
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not set. Skipping Gemini demo.")
            return None, None
        
        # Create Gemini provider
        provider = get_provider(
            provider_name='gemini',
            model_name='text-embedding-004',  # or 'embedding-001', etc.
            api_key=api_key
        )
        
        print(f"✅ Provider created: {provider}")
        print(f"   Model: {provider.model_name}")
        print(f"   Dimension: {provider.get_embedding_dimension()}")
        
        # Get embedding
        text = "This is a test for Gemini embeddings"
        embedding = provider.get_embedding(text)
        print(f"\n✅ Generated embedding:")
        print(f"   Shape: {embedding.shape}")
        
        # Validate
        validation = provider.validate_embedding(embedding)
        print(f"\n✅ Validation:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Dimension: {validation['dimension']}")
        
        return provider, embedding
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install google-generativeai")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def demo_claude():
    """Demo: Claude Provider"""
    print("\n" + "=" * 60)
    print("Claude Provider Demo")
    print("=" * 60)
    
    try:
        # Get API key from environment
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set. Skipping Claude demo.")
            return None, None
        
        # Create Claude provider
        provider = get_provider(
            provider_name='claude',
            model_name='claude-embedding',  # Adjust based on actual model name
            api_key=api_key
        )
        
        print(f"✅ Provider created: {provider}")
        print(f"   Model: {provider.model_name}")
        
        # Note: Claude embedding may need special implementation
        print("⚠️  Note: Claude embedding API may require special implementation")
        print("   Check Anthropic documentation for embedding endpoints")
        
        return provider, None
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install anthropic")
        return None, None
    except NotImplementedError as e:
        print(f"⚠️  {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def main():
    """Run all provider demos"""
    print("\n" + "=" * 60)
    print("Embedding Provider System Demo")
    print("=" * 60)
    
    # List available providers
    from cogman_tools import list_providers
    providers = list_providers()
    print(f"\n📦 Available providers: {', '.join(providers.keys())}")
    
    # Run demos
    demo_ollama()
    demo_openai()
    demo_gemini()
    demo_claude()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

