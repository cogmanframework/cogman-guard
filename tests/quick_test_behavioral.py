"""
Quick test for Baseline Behavioral Analyzer
"""

from cogman_tools import BehavioralAnalyzer
import numpy as np
from datetime import datetime, timedelta


def quick_test():
    """ทดลองใช้งานเร็วๆ"""
    
    print("🔍 Baseline Behavioral Analyzer - Quick Test")
    print("=" * 50)
    
    # 1. สร้าง baseline embeddings
    print("\n📊 Creating baseline embeddings...")
    baseline_embeddings = [np.random.randn(768) * 0.5 for _ in range(20)]
    
    # 2. สร้าง analyzer
    analyzer = BehavioralAnalyzer(
        baseline_embeddings=baseline_embeddings,
        similarity_threshold=0.7,
        anomaly_threshold=3.0
    )
    
    # 3. สร้าง test embeddings
    print("\n🔬 Creating test embeddings...")
    
    # คำนวณ baseline statistics
    baseline_mean = np.mean(baseline_embeddings, axis=0)
    baseline_std = np.std(baseline_embeddings, axis=0)
    
    # Good embeddings (คล้าย baseline - ควรได้ NORMAL)
    # สร้างจาก baseline mean + small noise
    good_embeddings = [
        baseline_mean + np.random.randn(768) * baseline_std * 0.3
        for _ in range(5)
    ]
    
    # Bad embeddings (ต่างจาก baseline มาก - ควรได้ WARNING/DEGRADED/UNSAFE)
    # สร้างจากค่าที่ห่างจาก baseline มาก
    bad_embeddings = [
        baseline_mean + np.random.randn(768) * baseline_std * 5.0 + 3.0
        for _ in range(5)
    ]
    
    # 4. Similarity Analysis
    print("\n🔗 Similarity Analysis:")
    sim_result = analyzer.similarity_analysis(good_embeddings[0], good_embeddings[1])
    print(f"  Similarity: {sim_result['similarity']:.3f}")
    print(f"  Distance: {sim_result['distance']:.3f}")
    
    # 5. Anomaly Detection
    print("\n🚨 Anomaly Detection:")
    
    print("\n  Good Embeddings:")
    good_anomaly = analyzer.anomaly_detection(good_embeddings)
    print(f"    Anomaly Density: {good_anomaly['anomaly_density']:.2%}")
    print(f"    Stress Index: {good_anomaly['stress_index']:.3f}")
    
    print("\n  Bad Embeddings:")
    bad_anomaly = analyzer.anomaly_detection(bad_embeddings)
    print(f"    Anomaly Density: {bad_anomaly['anomaly_density']:.2%}")
    print(f"    Stress Index: {bad_anomaly['stress_index']:.3f}")
    
    # 6. Cluster Analysis
    print("\n📊 Cluster Analysis:")
    all_embeddings = good_embeddings + bad_embeddings
    cluster_result = analyzer.cluster_analysis(all_embeddings)
    print(f"  Cluster Count: {cluster_result['cluster_count']}")
    print(f"  Distribution Shift: {cluster_result['distribution_shift']:.3f}")
    
    # 7. Operational Status
    print("\n⚡ Operational Status:")
    
    print("\n  Good Embeddings:")
    good_status = analyzer.assess_operational_status(good_embeddings, include_trends=False)
    print(f"    Status: {good_status.status}")
    print(f"    Confidence: {good_status.confidence:.2%}")
    
    print("\n  Bad Embeddings:")
    bad_status = analyzer.assess_operational_status(bad_embeddings, include_trends=False)
    print(f"    Status: {bad_status.status}")
    print(f"    Confidence: {bad_status.confidence:.2%}")
    print(f"    Reasons: {', '.join(bad_status.reasons[:2])}")
    
    # 8. Comprehensive Analysis
    print("\n📋 Comprehensive Analysis:")
    comprehensive = analyzer.comprehensive_analysis(all_embeddings)
    print(f"  Operational Status: {comprehensive['operational_status'].status}")
    print(f"  Anomaly Density: {comprehensive['anomaly_detection']['anomaly_density']:.2%}")
    print(f"  Cluster Count: {comprehensive['cluster_analysis']['cluster_count']}")
    
    # 9. Trend Analysis (จำลอง)
    print("\n📈 Trend Analysis:")
    base_time = datetime.now()
    for i in range(10):
        timestamp = base_time + timedelta(seconds=i*10)
        value = 0.1 + i * 0.02
        analyzer.record_metric('test_metric', value, timestamp)
    
    trend_result = analyzer.trend_analysis('test_metric')
    print(f"  Drift Slope: {trend_result['drift_slope']:.6f}")
    print(f"  Pattern Persistence: {trend_result['pattern_persistence']:.3f}")
    
    # 10. Generate Report
    print("\n📝 Generating report...")
    import os
    report_path = os.path.join('outputs', 'reports', 'behavioral_quick_test_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = analyzer.generate_report(save_path=report_path)
    
    print("\n✅ Quick test completed!")
    
    return analyzer, comprehensive


if __name__ == "__main__":
    analyzer, results = quick_test()

