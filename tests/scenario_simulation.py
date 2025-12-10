"""
Cogman Tools - Real-world Scenario Simulation
จำลองสถานการณ์จริงเพื่อให้ผู้ใช้ประเมินประสิทธิภาพเครื่องมือได้ชัดเจน
"""

import numpy as np
import time
import sys
from cogman_tools import EIMASAnalyzer


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"🎬 SCENARIO: {title}")
    print(f"{'=' * 60}")


def simulate_scenarios():
    print("🚀 Starting Cogman Tools - Production Simulation...")

    # Setup Analyzer
    # สร้าง Baseline จากข้อมูลปกติ (Random Normal distribution คืออุดมคติของ High Entropy Embedding)
    baseline_data = [np.random.randn(768) for _ in range(50)]
    analyzer = EIMASAnalyzer(baseline_embeddings=baseline_data, enable_monitoring=True)

    # ---------------------------------------------------------
    # Scenario 1: The "Silent Death" (Model Collapse)
    # สถานการณ์: โมเดลพัง เงียบๆ เริ่มส่งค่า Vector ที่เป็น 0 หรือค่าซ้ำๆ ออกมา
    # ---------------------------------------------------------
    print_header("1. Model Collapse (Silent Failure)")
    print("📝 Context: Model เริ่มรวน ส่งค่า Zero Vectors ออกมาปนกับข้อมูลจริง")

    mixed_batch = []
    # ข้อมูลดี 5 ตัว
    mixed_batch.extend([np.random.randn(768) for _ in range(5)])
    # ข้อมูลพัง (Zero/Sparse Vectors) 3 ตัว
    sparse_vector = np.zeros(768);
    sparse_vector[:5] = 1.0  # มีค่าแค่ 5 dimension
    mixed_batch.extend([sparse_vector for _ in range(3)])

    print("\n[Monitoring Log]")
    for i, emb in enumerate(mixed_batch):
        # Ingest ทีละตัวเหมือน Real-time
        result = analyzer.ingest_embedding(emb, embedding_id=f"stream_s1_{i}")

        # ดึงค่า S (Signal Quality) มาโชว์
        s_score = result['quality_analysis']['signal_quality']
        status = "✅ OK" if s_score > 0.8 else "❌ COLLAPSED"

        print(f"  Time {i}: Signal Quality = {s_score:.4f} | {status}")

    alerts = analyzer.get_alerts(level='WARNING')
    print(f"\n🚨 Alerts Triggered: {len(alerts)} (Expected: ~3)")
    if len(alerts) > 0:
        print(f"  Latest Alert: {alerts[-1].message}")

    # ---------------------------------------------------------
    # Scenario 2: The "Slow Drift" (Concept Drift)
    # สถานการณ์: ข้อมูลค่อยๆ เปลี่ยนหน้าตาไปเรื่อยๆ (Drift) จนหลุด Baseline
    # ---------------------------------------------------------
    print_header("2. Gradual Concept Drift")
    print("📝 Context: ข้อมูล User เปลี่ยนพฤติกรรมทีละนิด (Shift Mean)")

    # Clear monitoring buffer (keep deque type)
    analyzer.monitoring_buffer.clear()

    # สร้างข้อมูลที่ค่อยๆ "เลื่อน" หนีจาก Baseline (Mean=0)
    print("\n[Monitoring Drift]")
    baseline_mean = np.mean(baseline_data, axis=0)

    for t in range(1, 6):
        # Shift เพิ่มทีละ 0.2 (ลดความแรงลงนิดนึงเพื่อให้เห็นกราฟค่อยๆ ขึ้น)
        drift_factor = t * 0.2
        
        # FIX: สร้าง Batch ข้อมูล 10 ตัว (แทนที่จะส่งไปตัวเดียว)
        # เพื่อให้คำนวณ Distribution ได้โดยไม่เกิด Divide by Zero
        drifted_batch = [
            np.random.randn(768) + (np.ones(768) * drift_factor)
            for _ in range(10)
        ]

        # วิเคราะห์เทียบกับ Baseline เดิม
        result = analyzer.comprehensive_analysis(drifted_batch)
        dist_shift = result['core_analysis']['cluster_analysis']['distribution_shift']

        # Check Status
        status = result['core_analysis']['operational_status'].status

        # Visualization
        bar_length = int(dist_shift * 2)  # Scale bar
        bar = "█" * bar_length
        print(f"  Month {t}: Shift = {dist_shift:.2f} {bar:<15} | Status: {status}")

    # ---------------------------------------------------------
    # Scenario 3: The "Attack" (Anomaly Spike)
    # สถานการณ์: มีข้อมูลขยะ (Random Noise ที่ distribution ผิดเพี้ยน) ถล่มเข้ามา
    # ---------------------------------------------------------
    print_header("3. Anomaly Spike (Potential Attack)")
    print("📝 Context: เจอข้อมูล Outlier ที่มีความเครียด (Stress) สูงผิดปกติ")

    # สร้างข้อมูลที่มี Variance สูงผิดปกติ (High Energy Noise)
    attack_embeddings = [np.random.randn(768) * 5.0 for _ in range(5)]

    anomaly_result = analyzer.anomaly_detection(attack_embeddings)
    stress_index = anomaly_result['stress_index']

    print(f"\n📊 Batch Analysis Result:")
    print(f"  Anomaly Density: {anomaly_result['anomaly_density']:.1%}")
    print(f"  Stress Index:    {stress_index:.2f} (Normal < 1.0)")

    if stress_index > 2.0:
        print("  ⚠️  CRITICAL: High Stress Detected! System might be under attack or broken.")
    else:
        print("  ✅  System Stable")

    print("\n" + "=" * 60)
    print("✅ Simulation Completed. Ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    simulate_scenarios()