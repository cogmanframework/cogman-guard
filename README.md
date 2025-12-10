# 🔧 Cogman Tools

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://github.com/)
[![EIMAS](https://img.shields.io/badge/Standard-EIMAS%20v1.0-orange)](docs/EIMAS_MAPPING.md)

**Operational Analysis Tools for Embeddings & AI Systems**

A comprehensive toolkit for **monitoring health, stability, and operational risks** of Embedding and AI systems in production environments.

---

## 🎯 Project Goal

Cogman Tools is designed to answer one critical question for MLOps:

> **"Is this AI system still safe and reliable for production use?"**

- ✅ **Focuses on Operational Health:** Detects drift, collapse, and anomalies.
- ✅ **Agnostic:** Does NOT judge the "correctness" of the answer (Truthfulness), but validates the **integrity of the signal**.
- ✅ **Complementary:** Works alongside standard evaluation benchmarks (RAGAS, etc.).

---

## 🏗️ System Architecture

```
Input: Raw Embeddings
        |
        v
Layer 1: Quality Inspector
  - Physics & Signal Analysis
  - Calculates: Entropy (H), Signal Quality (S)
  - Output: EQI Score
        |
        v
Layer 2: Behavioral Analyzer
  - Baseline Comparison
  - Checks: Concept Drift, Anomaly Detection
  - Output: Operational Status
        |
        v
Layer 3: EIMAS Core
  - Intelligence Monitoring
  - Actions: Alerts, Forgery/Pattern Detection
        |
        v
Output: Report & Dashboard
```

---

## 📦 Core Components

### 1. Embedding Quality Inspector 🔍
Analyzes the **physics and structure** of embedding vectors.
- **Key Metric:** `Signal Quality (S)` — Measures signal integrity (1.0 = Perfect, 0.0 = Noise/Collapsed).

### 2. Baseline Behavioral Analyzer 📈
Detects behavioral changes relative to a known baseline.
- **Key Metric:** `Drift Slope` & `Operational Status` (Normal → Unsafe).

### 3. EIMAS Analyzer 🔬
The core engine following the **EIMAS v1.0 Specification**.
- **Features:** Real-time monitoring, alerting, specialized inspections (forgery detection).

---

## 🌍 Real-world Simulation Results

We simulated 3 common production scenarios to demonstrate Cogman Tools' capabilities.  
Run it yourself: `python tests/scenario_simulation.py`

### Scenario 1: Model Collapse (The "Silent Death")
*Context: Model starts outputting corrupted/zero vectors.*
```text
[Log Output]
Time 0: Signal Quality = 0.9588 | ✅ OK
Time 4: Signal Quality = 0.9686 | ✅ OK
Time 5: Signal Quality = 0.0080 | ❌ COLLAPSED  <-- Detected immediately
Time 6: Signal Quality = 0.0080 | ❌ COLLAPSED
```

### Scenario 2: Gradual Concept Drift
*Context: User behavior slowly changes over months.*
```text
[Log Output]
Month 1: Shift = 2.94 █████           | Status: DEGRADED (Early Warning)
Month 2: Shift = 3.70 ███████         | Status: DEGRADED
Month 3: Shift = 4.78 █████████       | Status: DEGRADED
Month 4: Shift = 6.03 ████████████    | Status: UNSAFE   <-- Critical Threshold
Month 5: Shift = 7.46 ██████████████  | Status: UNSAFE
```

### Scenario 3: Anomaly Spike (Potential Attack)
*Context: Injection of high-variance noise.*
```text
[Log Output]
Anomaly Density: 100.0%
Stress Index:    2.59 (Normal < 1.0)
⚠️ CRITICAL: High Stress Detected! System might be under attack.
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cogman_tools.git
cd cogman_tools

# Install package
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Quick Start

### 1. Run Diagnostics
```bash
# Run the full test suite to verify installation
python tests/run_all_tests.py
```

### 2. Python Usage Example
```python
from cogman_tools import EIMASAnalyzer
import numpy as np

# 1. Setup Analyzer with Baseline Data (Normal behavior)
baseline = [np.random.randn(768) for _ in range(50)]
analyzer = EIMASAnalyzer(baseline_embeddings=baseline, enable_monitoring=True)

# 2. Ingest New Data (Real-time)
new_embedding = np.random.randn(768)
result = analyzer.ingest_embedding(new_embedding, embedding_id="req_123")

# 3. Check Status
if result['operational_status'].status == "UNSAFE":
    print("🚨 BLOCK THIS REQUEST")
    print(f"Reason: {result['operational_status'].reasons}")
else:
    print(f"✅ Pass. Quality Score: {result['quality_index']:.2f}")
```

---

## 📊 Metrics Explained

| Metric | Range | Good Direction | Description |
|--------|-------|----------------|-------------|
| **Signal Quality (S)** | 0.0 - 1.0 | Higher (→ 1.0) | Smoothness and integrity of the vector signal. Low values indicate noise or collapse. |
| **Entropy (H)** | 0.0 - 1.0 | ~0.7 (Optimal) | Information distribution. Too low = sparse/collapsed. Too high = white noise. |
| **EQI Score** | 0 - 100 | Higher | **Embedding Quality Index**. The composite health score. |
| **Stress Index** | 0 - 10+ | Lower | Structural tension in the vector space. High values mean "Out of Domain". |

---

## 🔌 Supported Providers

Cogman Tools is compatible with any vector source, with built-in wrappers for:
- ✅ **OpenAI** (`text-embedding-3`, `ada-002`)
- ✅ **Ollama** (Local models: `nomic-embed`, `llama3`)
- ✅ **Google Gemini**
- ✅ **HuggingFace** (via Transformers)

---

## 📂 Examples & CLI

- Examples:
  - `examples/embedding_quality_example.py`
  - `examples/behavioral_analysis_example.py`
  - `examples/eimas_full_example.py`
- CLI:
  - `cogman-tools-quick-test embedding`
  - `cogman-tools-quick-test behavioral`
  - `cogman-tools-quick-test eimas`

---

## 📚 Docs

- `docs/BASELINE_BEHAVIORAL_ANALYSIS_SPEC.md`
- `docs/EIMAS_MAPPING.md`

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Built for the AI Engineering community to make production systems safer and more observable.*
