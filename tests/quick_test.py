from cogman_tools import EmbeddingQualityInspector
import torch
import numpy as np


def quick_test():
    """ทดลองใช้งานเร็วๆ"""
    
    print("=" * 60)
    print("Embedding Quality Inspector - Quick Test")
    print("=" * 60)
    
    # 1. สร้าง embedding ตัวอย่าง
    print("\nCreating test embeddings...")
    
    # Good embedding (ปกติ - random distribution)
    good_emb = torch.randn(1, 768) * 0.5  # BERT-like dimension
    
    # Bad embedding (มีปัญหา - mostly zeros with some ones)
    bad_emb = torch.zeros(1, 768)
    bad_emb[:, :100] = 1.0  # Only 13% non-zero, poor distribution
    
    # 2. ตรวจสอบ
    inspector = EmbeddingQualityInspector()
    
    print("\n" + "-" * 40)
    print("Good Embedding (normal random distribution)")
    print("-" * 40)
    good_result = inspector.analyze_embedding(good_emb)
    good_eqi = good_result.get('embedding_quality_index', 0)
    good_h = good_result.get('distribution_entropy', 0)
    good_s = good_result.get('signal_quality', 0)
    good_i = good_result.get('information_strength', 0)
    
    print(f"  EQI (Embedding Quality Index): {good_eqi:.1f}/100")
    print(f"  Distribution Entropy (H):      {good_h:.3f} (0-1, optimal ~0.7)")
    print(f"  Signal Quality (S):            {good_s:.3f} (0-1)")
    print(f"  Information Strength (I):      {good_i:.0f}")
    
    print("\n" + "-" * 40)
    print("Bad Embedding (sparse, poor distribution)")
    print("-" * 40)
    bad_result = inspector.analyze_embedding(bad_emb)
    bad_eqi = bad_result.get('embedding_quality_index', 0)
    bad_h = bad_result.get('distribution_entropy', 0)
    bad_s = bad_result.get('signal_quality', 0)
    bad_i = bad_result.get('information_strength', 0)
    
    print(f"  EQI (Embedding Quality Index): {bad_eqi:.1f}/100")
    print(f"  Distribution Entropy (H):      {bad_h:.3f} (0-1, optimal ~0.7)")
    print(f"  Signal Quality (S):            {bad_s:.3f} (0-1)")
    print(f"  Information Strength (I):      {bad_i:.0f}")
    
    # 3. เปรียบเทียบ
    print("\n" + "-" * 40)
    print("Comparison Summary")
    print("-" * 40)
    comparison = inspector.compare_embeddings([good_emb, bad_emb], ['Good', 'Bad'])
    
    # Verify Good > Bad
    if good_eqi > bad_eqi:
        print(f"  ✅ Good EQI ({good_eqi:.1f}) > Bad EQI ({bad_eqi:.1f}) - CORRECT!")
    else:
        print(f"  ⚠️  Good EQI ({good_eqi:.1f}) <= Bad EQI ({bad_eqi:.1f}) - UNEXPECTED")
    
    # Quality interpretation
    print(f"\n  Quality Interpretation:")
    print(f"    Good Embedding: {'Excellent' if good_eqi >= 70 else 'Normal' if good_eqi >= 50 else 'Warning' if good_eqi >= 30 else 'Bad'}")
    print(f"    Bad Embedding:  {'Excellent' if bad_eqi >= 70 else 'Normal' if bad_eqi >= 50 else 'Warning' if bad_eqi >= 30 else 'Bad'}")
    
    # 4. Visualization
    print("\nGenerating visualization for good embedding...")
    inspector.visualize(good_result)
    
    print("\n" + "=" * 60)
    print("Quick Test Completed!")
    print("=" * 60)
    
    return good_result, bad_result


if __name__ == "__main__":
    good, bad = quick_test()

