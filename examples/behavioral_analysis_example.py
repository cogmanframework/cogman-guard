"""
Example: Baseline Behavioral Analyzer
"""
import numpy as np
from cogman_tools import BehavioralAnalyzer


def main():
    baseline = [np.random.randn(768) for _ in range(20)]
    analyzer = BehavioralAnalyzer(baseline_embeddings=baseline)

    good_embeddings = [np.random.randn(768) * 0.5 for _ in range(5)]
    bad_embeddings = [np.random.randn(768) * 3.0 + 3.0 for _ in range(5)]

    good_status = analyzer.assess_operational_status(good_embeddings, include_trends=False)
    bad_status = analyzer.assess_operational_status(bad_embeddings, include_trends=False)

    print("Good Status :", good_status.status, "confidence", f"{good_status.confidence:.2%}")
    print("Bad Status  :", bad_status.status, "confidence", f"{bad_status.confidence:.2%}")
    print("Bad Reasons :", "; ".join(bad_status.reasons[:3]))


if __name__ == "__main__":
    main()
