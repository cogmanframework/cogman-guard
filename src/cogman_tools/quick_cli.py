import argparse
import numpy as np
import torch

from cogman_tools import (
    EmbeddingQualityInspector,
    BehavioralAnalyzer,
    EIMASAnalyzer,
)
from cogman_tools.eimas_analyzer import demo_eimas


def run_embedding():
    print("=== Embedding Quality Quick Test ===")
    inspector = EmbeddingQualityInspector()

    good_emb = torch.randn(1, 768) * 0.5
    bad_emb = torch.zeros(1, 768)
    bad_emb[:, :100] = 1.0

    good_result = inspector.analyze_embedding(good_emb)
    bad_result = inspector.analyze_embedding(bad_emb)

    print("Good -> EQI: {:.2f}, S: {:.3f}, H: {:.3f}".format(
        good_result.get('embedding_quality_index', 0),
        good_result.get('signal_quality', 0),
        good_result.get('distribution_entropy', 0)))
    print("Bad  -> EQI: {:.2f}, S: {:.3f}, H: {:.3f}".format(
        bad_result.get('embedding_quality_index', 0),
        bad_result.get('signal_quality', 0),
        bad_result.get('distribution_entropy', 0)))


def run_behavioral():
    print("=== Baseline Behavioral Quick Test ===")
    baseline_embeddings = [np.random.randn(768) * 0.5 for _ in range(20)]
    analyzer = BehavioralAnalyzer(
        baseline_embeddings=baseline_embeddings,
        similarity_threshold=0.7,
        anomaly_threshold=3.0,
    )

    good_embeddings = [np.random.randn(768) * 0.5 for _ in range(5)]
    bad_embeddings = [np.random.randn(768) * 3.0 + 3.0 for _ in range(5)]

    good_status = analyzer.assess_operational_status(good_embeddings, include_trends=False)
    bad_status = analyzer.assess_operational_status(bad_embeddings, include_trends=False)

    print(f"Good Status : {good_status.status} (confidence {good_status.confidence:.2%})")
    print(f"Bad Status  : {bad_status.status} (confidence {bad_status.confidence:.2%})")


def run_eimas():
    print("=== EIMAS Analyzer Demo ===")
    analyzer, results = demo_eimas()
    op = results['core_analysis']['operational_status'].status
    print(f"Operational Status: {op}")


def main():
    parser = argparse.ArgumentParser(description="Cogman Tools quick CLI")
    parser.add_argument(
        "command",
        choices=["embedding", "behavioral", "eimas"],
        help="Which quick demo to run",
    )
    args = parser.parse_args()

    if args.command == "embedding":
        run_embedding()
    elif args.command == "behavioral":
        run_behavioral()
    elif args.command == "eimas":
        run_eimas()


if __name__ == "__main__":
    main()
