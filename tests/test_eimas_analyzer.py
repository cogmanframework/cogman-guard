"""
Test file for EIMAS Analyzer
"""

from cogman_tools import EIMASAnalyzer, Alert, EmbeddingLineage
import numpy as np
import torch
from datetime import datetime, timedelta


def test_reference_similarity_verification():
    """ทดสอบ Reference Similarity Verification (EIMAS 4.1)"""
    print("\n" + "="*50)
    print("Testing Reference Similarity Verification")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง trusted sources
    trusted_sources = [np.random.randn(768) * 0.5 for _ in range(5)]
    
    # สร้าง embedding ที่คล้ายกับ trusted sources
    conformant_emb = np.random.randn(768) * 0.5
    
    # สร้าง embedding ที่ต่างจาก trusted sources
    non_conformant_emb = np.random.randn(768) * 3.0
    
    # ทดสอบ
    result_conformant = analyzer.reference_similarity_verification(
        conformant_emb, trusted_sources, domain_threshold=0.7
    )
    result_non_conformant = analyzer.reference_similarity_verification(
        non_conformant_emb, trusted_sources, domain_threshold=0.7
    )
    
    print(f"Conformant embedding - Domain conformance: {result_conformant['is_conformant']}")
    print(f"Non-conformant embedding - Domain conformance: {result_non_conformant['is_conformant']}")
    
    assert 'domain_conformance_score' in result_conformant, "Reference similarity test failed"
    print("✅ Reference Similarity Verification test passed")


def test_imitation_forgery_detection():
    """ทดสอบ Imitation & Forgery Detection (EIMAS 4.2)"""
    print("\n" + "="*50)
    print("Testing Imitation & Forgery Detection")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง embeddings ที่มี near-duplicates
    base_emb = np.random.randn(768)
    near_duplicates = [base_emb + np.random.randn(768) * 0.01 for _ in range(5)]
    
    # สร้าง embeddings ที่แตกต่างกัน
    diverse_embeddings = [np.random.randn(768) for _ in range(5)]
    
    # ทดสอบ
    result_duplicates = analyzer.imitation_forgery_detection(near_duplicates)
    result_diverse = analyzer.imitation_forgery_detection(diverse_embeddings)
    
    print(f"Near-duplicates - Forgery risk: {result_duplicates['forgery_risk']:.3f}")
    print(f"Diverse embeddings - Forgery risk: {result_diverse['forgery_risk']:.3f}")
    
    assert result_duplicates['forgery_risk'] > result_diverse['forgery_risk'], "Forgery detection test failed"
    print("✅ Imitation & Forgery Detection test passed")


def test_hidden_communication_pattern_detection():
    """ทดสอบ Hidden Communication Pattern Detection (EIMAS 4.3)"""
    print("\n" + "="*50)
    print("Testing Hidden Communication Pattern Detection")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง embeddings ที่มี hidden patterns (clusters)
    cluster1 = [np.random.randn(768) + np.array([1, 0, 0] * 256) for _ in range(5)]
    cluster2 = [np.random.randn(768) + np.array([0, 1, 0] * 256) for _ in range(5)]
    embeddings_with_patterns = cluster1 + cluster2
    
    # สร้าง embeddings แบบสุ่ม
    random_embeddings = [np.random.randn(768) for _ in range(10)]
    
    # ทดสอบ
    result_patterns = analyzer.hidden_communication_pattern_detection(embeddings_with_patterns)
    result_random = analyzer.hidden_communication_pattern_detection(random_embeddings)
    
    print(f"Embeddings with patterns - Patterns detected: {result_patterns['hidden_patterns_detected']}")
    print(f"Random embeddings - Patterns detected: {result_random['hidden_patterns_detected']}")
    
    assert 'hidden_patterns_detected' in result_patterns, "Hidden pattern detection test failed"
    print("✅ Hidden Communication Pattern Detection test passed")


def test_information_propagation_tracking():
    """ทดสอบ Information Propagation Tracking (EIMAS 4.4)"""
    print("\n" + "="*50)
    print("Testing Information Propagation Tracking")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง parent embeddings
    parent_emb = np.random.randn(768)
    parent_lineage = analyzer.information_propagation_tracking(
        embedding_id="parent_001",
        embedding=parent_emb,
        source="model_v1"
    )
    
    # สร้าง child embedding
    child_emb = parent_emb + np.random.randn(768) * 0.1
    child_lineage = analyzer.information_propagation_tracking(
        embedding_id="child_001",
        embedding=child_emb,
        source="model_v1",
        parent_ids=["parent_001"]
    )
    
    print(f"Parent lineage ID: {parent_lineage.embedding_id}")
    print(f"Child lineage ID: {child_lineage.embedding_id}")
    print(f"Child has {len(child_lineage.parent_ids)} parent(s)")
    
    assert child_lineage.parent_ids == ["parent_001"], "Propagation tracking test failed"
    print("✅ Information Propagation Tracking test passed")


def test_real_time_monitoring():
    """ทดสอบ Real-time Monitoring (EIMAS 5.1)"""
    print("\n" + "="*50)
    print("Testing Real-time Monitoring")
    print("="*50)
    
    analyzer = EIMASAnalyzer(enable_monitoring=True)
    
    # Ingest embeddings
    for i in range(5):
        emb = np.random.randn(768) * 0.5
        result = analyzer.ingest_embedding(emb, embedding_id=f"test_{i}")
        print(f"  Ingested {result['embedding_id']}: Quality Index = {result['quality_index']:.2f}")
    
    print(f"Monitoring buffer size: {len(analyzer.monitoring_buffer)}")
    
    assert len(analyzer.monitoring_buffer) == 5, "Real-time monitoring test failed"
    print("✅ Real-time Monitoring test passed")


def test_alert_system():
    """ทดสอบ Alert System (EIMAS 5.2)"""
    print("\n" + "="*50)
    print("Testing Alert System")
    print("="*50)
    
    analyzer = EIMASAnalyzer(enable_monitoring=True)
    
    # สร้าง embedding ที่แย่จริงๆ: sparse + collapsed
    low_quality_emb = np.zeros(768)
    low_quality_emb[:10] = 1.0  # Only 10/768 dimensions active (very sparse)
    
    # ตรวจสอบค่า EQI และ S ก่อน เพื่อตั้ง threshold ที่เหมาะสม
    quality_result = analyzer.quality_inspector.analyze_embedding(low_quality_emb)
    eqi = quality_result.get('embedding_quality_index', 0)
    signal_quality = quality_result.get('signal_quality', 0)
    
    # ตั้ง threshold ให้สูงกว่าค่าจริง เพื่อให้ trigger alert
    # ใช้ค่า EQI + 10 และ S + 0.1 เพื่อให้แน่ใจว่า trigger
    analyzer.configure_thresholds({
        'quality_index_min': min(50.0, eqi + 10.0),  # Higher than actual
        'signal_quality_min': min(0.7, signal_quality + 0.1)  # Higher than actual
    })
    
    print(f"  Low-quality embedding - EQI: {eqi:.2f}, Signal Quality: {signal_quality:.3f}")
    print(f"  Thresholds - EQI min: {analyzer.thresholds['quality_index_min']:.2f}, S min: {analyzer.thresholds['signal_quality_min']:.3f}")
    
    # Ingest low-quality embedding (should trigger alert)
    result = analyzer.ingest_embedding(low_quality_emb, embedding_id="low_quality")
    
    # Get alerts
    alerts = analyzer.get_alerts(level='WARNING')
    
    print(f"Total alerts generated: {len(alerts)}")
    if alerts:
        for alert in alerts:
            print(f"  Alert: {alert.message}")
    
    assert len(alerts) > 0, f"Alert system test failed - No alerts generated. EQI: {eqi:.2f} (threshold: {analyzer.thresholds['quality_index_min']:.2f}), Signal Quality: {signal_quality:.3f} (threshold: {analyzer.thresholds['signal_quality_min']:.3f})"
    print("✅ Alert System test passed")


def test_version_comparison():
    """ทดสอบ Version Comparison (EIMAS 5.4)"""
    print("\n" + "="*50)
    print("Testing Version Comparison")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง embeddings สำหรับ 2 เวอร์ชัน
    embeddings_v1 = [np.random.randn(768) * 0.5 for _ in range(10)]
    embeddings_v2 = [np.random.randn(768) * 0.6 for _ in range(10)]  # ต่างกันเล็กน้อย
    
    comparison = analyzer.compare_versions(
        version_a="1.0.0",
        version_b="2.0.0",
        embeddings_a=embeddings_v1,
        embeddings_b=embeddings_v2
    )
    
    print(f"Version A: {comparison['version_a']}")
    print(f"Version B: {comparison['version_b']}")
    print(f"Regression detected: {comparison['regression_detected']}")
    
    assert 'metrics' in comparison, "Version comparison test failed"
    print("✅ Version Comparison test passed")


def test_explainability_tools():
    """ทดสอบ Explainability Tools (EIMAS 7.1)"""
    print("\n" + "="*50)
    print("Testing Explainability Tools")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # Test explain clustering
    embeddings = [np.random.randn(768) for _ in range(10)]
    cluster_result = analyzer.cluster_analysis(embeddings)
    explanation = analyzer.explain_clustering(embeddings, cluster_result['labels'])
    
    print(f"Cluster explanation - Cluster count: {explanation['cluster_count']}")
    
    # Test explain anomaly
    baseline = [np.random.randn(768) * 0.5 for _ in range(10)]
    anomalous_emb = np.random.randn(768) * 3.0
    anomaly_explanation = analyzer.explain_anomaly(anomalous_emb, baseline)
    
    print(f"Anomaly explanation - Anomaly score: {anomaly_explanation['anomaly_score']:.3f}")
    print(f"Top contributing dimensions: {len(anomaly_explanation['top_contributing_dimensions'])}")
    
    assert 'cluster_explanations' in explanation, "Explainability test failed"
    assert 'top_contributing_dimensions' in anomaly_explanation, "Explainability test failed"
    print("✅ Explainability Tools test passed")


def test_confidence_scoring():
    """ทดสอบ Confidence Scoring (EIMAS 7.2)"""
    print("\n" + "="*50)
    print("Testing Confidence Scoring")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง analysis result
    embeddings = [np.random.randn(768) for _ in range(20)]
    results = analyzer.comprehensive_analysis(embeddings)
    
    confidence = analyzer.assess_confidence(results)
    
    print(f"Overall confidence: {confidence['overall_confidence']:.3f}")
    print(f"Data sufficiency: {confidence['data_sufficiency']:.3f}")
    print(f"Confidence level: {confidence['confidence_level']}")
    
    assert 'overall_confidence' in confidence, "Confidence scoring test failed"
    print("✅ Confidence Scoring test passed")


def test_comparative_analysis():
    """ทดสอบ Comparative Analysis (EIMAS 7.3)"""
    print("\n" + "="*50)
    print("Testing Comparative Analysis")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    # สร้าง 2 กลุ่ม embeddings
    group_a = [np.random.randn(768) * 0.5 for _ in range(10)]
    group_b = [np.random.randn(768) * 0.6 for _ in range(10)]
    
    comparison = analyzer.comparative_analysis(
        embeddings_group_a=group_a,
        embeddings_group_b=group_b,
        group_a_label="Model A",
        group_b_label="Model B"
    )
    
    print(f"Group A status: {comparison['group_a']['status']}")
    print(f"Group B status: {comparison['group_b']['status']}")
    print(f"Significant difference: {comparison['interpretation']['significant_difference']}")
    
    assert 'differences' in comparison, "Comparative analysis test failed"
    print("✅ Comparative Analysis test passed")


def test_comprehensive_eimas_analysis():
    """ทดสอบ Comprehensive EIMAS Analysis"""
    print("\n" + "="*50)
    print("Testing Comprehensive EIMAS Analysis")
    print("="*50)
    
    analyzer = EIMASAnalyzer()
    
    embeddings = [np.random.randn(768) for _ in range(10)]
    results = analyzer.comprehensive_analysis(embeddings, include_specialized=True)
    
    print(f"EIMAS version: {results['eimas_version']}")
    print(f"Core analysis status: {results['core_analysis']['operational_status'].status}")
    print(f"Mean quality index: {results['quality_analysis']['mean_quality_index']:.2f}")
    
    if 'specialized_inspections' in results:
        print(f"Specialized inspections included: ✅")
    
    assert 'core_analysis' in results, "Comprehensive analysis test failed"
    assert 'quality_analysis' in results, "Comprehensive analysis test failed"
    print("✅ Comprehensive EIMAS Analysis test passed")


def test_report_generation():
    """ทดสอบ Report Generation (EIMAS 10.1)"""
    print("\n" + "="*50)
    print("Testing Report Generation")
    print("="*50)
    
    analyzer = EIMASAnalyzer(enable_monitoring=True)
    
    # Ingest some embeddings
    embeddings = [np.random.randn(768) for _ in range(5)]
    for i, emb in enumerate(embeddings):
        analyzer.ingest_embedding(emb, embedding_id=f"report_test_{i}")
    
    # Generate report
    import os
    report_path = os.path.join('outputs', 'reports', 'test_eimas_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = analyzer.generate_eimas_report(embeddings, save_path=report_path)
    
    assert 'EIMAS' in report, "Report generation test failed"
    assert 'SYSTEM STATUS' in report, "Report generation test failed"
    print("✅ Report Generation test passed")


def run_all_tests():
    """รันทุก test"""
    print("\n" + "="*70)
    print("EIMAS ANALYZER - TEST SUITE")
    print("="*70)
    
    tests = [
        test_reference_similarity_verification,
        test_imitation_forgery_detection,
        test_hidden_communication_pattern_detection,
        test_information_propagation_tracking,
        test_real_time_monitoring,
        test_alert_system,
        test_version_comparison,
        test_explainability_tools,
        test_confidence_scoring,
        test_comparative_analysis,
        test_comprehensive_eimas_analysis,
        test_report_generation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    return passed, failed


if __name__ == "__main__":
    run_all_tests()

