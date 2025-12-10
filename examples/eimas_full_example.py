"""
Example: EIMAS Analyzer
"""
import numpy as np
from cogman_tools import EIMASAnalyzer


def main():
    baseline = [np.random.randn(768) for _ in range(20)]
    analyzer = EIMASAnalyzer(baseline_embeddings=baseline, enable_monitoring=True)

    embeddings = [np.random.randn(768) for _ in range(10)]
    results = analyzer.comprehensive_analysis(embeddings)

    op_status = results['core_analysis']['operational_status']
    print("Operational Status:", op_status.status)
    print("Reasons:", "; ".join(op_status.reasons[:3]))

    # Monitoring ingest
    ingest = analyzer.ingest_embedding(embeddings[0], embedding_id="demo_0")
    alerts = analyzer.get_alerts(level='WARNING')
    print("Alerts count:", len(alerts))


if __name__ == "__main__":
    main()
