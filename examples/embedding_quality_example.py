"""
Example: Embedding Quality Inspector
"""
import torch
from cogman_tools import EmbeddingQualityInspector


def main():
    inspector = EmbeddingQualityInspector()
    good_emb = torch.randn(1, 768) * 0.5
    bad_emb = torch.zeros(1, 768)
    bad_emb[:, :100] = 1.0

    print("-- Good Embedding --")
    res_good = inspector.analyze_embedding(good_emb)
    print(f"EQI: {res_good.get('embedding_quality_index', 0):.2f}")
    print(f"Signal Quality: {res_good.get('signal_quality', 0):.3f}")

    print("\n-- Bad Embedding --")
    res_bad = inspector.analyze_embedding(bad_emb)
    print(f"EQI: {res_bad.get('embedding_quality_index', 0):.2f}")
    print(f"Signal Quality: {res_bad.get('signal_quality', 0):.3f}")


if __name__ == "__main__":
    main()
