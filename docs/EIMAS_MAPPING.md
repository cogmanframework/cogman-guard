# Mapping: Cogman Tools ↔ EIMAS Specification

**Document Type:** Reference Mapping  
**Purpose:** แสดงความสอดคล้องเชิงหน้าที่ (Functional Compliance)  
**Note:** Cogman Tools เป็น reference implementation ไม่ใช่มาตรฐานบังคับ

---

## 1. System Architecture Components

### 1.1 Embedding Models Layer

**EIMAS Requirement:**
- Multilingual / domain / multimodal embeddings

**Cogman Tools:**
- ✅ รองรับ embedding ใด ๆ ที่เป็น numerical vector
- ✅ ใช้ได้กับ HuggingFace, custom PyTorch, API-based models

**Component:**
- `EmbeddingQualityInspector`
- `BaselineBehavioralAnalyzer`
- `EIMASAnalyzer`

---

### 1.2 Vector Data Management Layer

**EIMAS Requirement:**
- Real-time / versioned vector handling

**Cogman Tools:**
- ✅ รองรับ ingestion แบบ list / batch / stream
- ✅ รองรับ baseline vs current comparison (implicit versioning)
- ✅ Embedding lineage tracking

**Component:**
- `BehavioralAnalyzer.baseline_embeddings`
- `BehavioralAnalyzer.record_metric()` + historical buffer
- `EIMASAnalyzer.embedding_lineage`
- `EIMASAnalyzer.ingest_embedding()`

---

### 1.3 Visualization & Dashboard

**EIMAS Requirement:**
- Interactive, time-based visualization

**Cogman Tools:**
- ✅ Multi-panel plots
- ✅ 3D embedding visualization
- ✅ Comparative visualization

**Component:**
- `EmbeddingQualityInspector.visualize()`
- `EmbeddingQualityInspector.compare_embeddings()`
- `EIMASAnalyzer.comprehensive_analysis()`

---

### 1.4 Integration API

**EIMAS Requirement:**
- API / integration-ready

**Cogman Tools:**
- ✅ Python SDK (callable interface)
- ✅ JSON-serializable results
- ⚠️ REST API (planned / external wrapper)

**Component:**
- Public-facing Python classes
- `EIMASAnalyzer.export_json()`

---

## 2. Core Analysis Capabilities

### Similarity Analysis (EIMAS 3.1)

**EIMAS:**
- Contextual similarity
- Reference similarity

**Cogman:**
- ✅ Cosine-based similarity
- ✅ Distance-based out-of-domain signal
- ✅ Reference similarity verification

**Method:**
- `BehavioralAnalyzer.similarity_analysis(emb1, emb2)`
- `EIMASAnalyzer.similarity_analysis()` (with reference support)

---

### Cluster Analysis (EIMAS 3.2)

**EIMAS:**
- Behavior regime grouping
- Distribution shift detection

**Cogman:**
- ✅ Dynamic cluster count
- ✅ Cluster density
- ✅ Distribution shift metric
- ✅ Cluster explanation

**Method:**
- `BehavioralAnalyzer.cluster_analysis(embeddings)`
- `EIMASAnalyzer.explain_clustering()`

---

### Anomaly Detection (EIMAS 3.3)

**EIMAS:**
- Instance & distribution anomaly

**Cogman:**
- ✅ Anomaly score
- ✅ Anomaly density
- ✅ Stress index
- ✅ Anomaly explanation

**Method:**
- `BehavioralAnalyzer.anomaly_detection(embeddings)`
- `EIMASAnalyzer.explain_anomaly()`

---

### Trend Analysis (EIMAS 3.4)

**EIMAS:**
- Temporal drift
- Stability detection

**Cogman:**
- ✅ Drift slope
- ✅ Stability variance
- ✅ Silent failure detection

**Method:**
- `BehavioralAnalyzer.record_metric()`
- `BehavioralAnalyzer.trend_analysis(metric_name)`

---

### Cross-modal Analysis (EIMAS 3.5)

**EIMAS:**
- Modal alignment / integrity check

**Cogman:**
- ✅ Cross-modal alignment score
- ✅ Modality divergence

**Method:**
- `BehavioralAnalyzer.cross_modal_analysis(modal_embeddings)`

---

## 3. Specialized Inspection Capabilities

### Reference Similarity Verification (EIMAS 4.1)

**EIMAS:**
- Similarity to trusted sources

**Cogman:**
- ✅ Baseline embedding comparison
- ✅ Out-of-domain detection
- ✅ Domain conformance scoring

**Method:**
- `EIMASAnalyzer.reference_similarity_verification(embedding, trusted_sources)`

---

### Imitation / Forgery Detection (EIMAS 4.2)

**EIMAS:**
- Near-duplicate pattern detection

**Cogman:**
- ✅ High similarity + anomaly conflict detection
- ✅ Cluster saturation signal
- ✅ Stylometric similarity analysis

**Method:**
- `EIMASAnalyzer.imitation_forgery_detection(embeddings)`

---

### Hidden Communication Pattern Detection (EIMAS 4.3)

**EIMAS:**
- Latent non-obvious structure

**Cogman:**
- ✅ Unexpected cluster emergence
- ✅ Recurrent anomaly signatures
- ✅ Latent correlation analysis

**Method:**
- `EIMASAnalyzer.hidden_communication_pattern_detection(embeddings)`

---

### Information Propagation Tracking (EIMAS 4.4)

**EIMAS:**
- Diffusion / spread analysis

**Cogman:**
- ✅ Trend-based propagation analysis
- ✅ Time-indexed embedding behavior
- ✅ Embedding lineage tracking

**Method:**
- `EIMASAnalyzer.information_propagation_tracking()`
- `EIMASAnalyzer.embedding_lineage`

---

## 4. Monitoring & Surveillance Functions

### Real-time Monitoring (EIMAS 5.1)

**EIMAS:**
- Stream-based health signals

**Cogman:**
- ✅ Batch + rolling window support
- ✅ Real-time compatible architecture
- ✅ Streaming embedding ingestion

**Method:**
- `EIMASAnalyzer.ingest_embedding()`
- `EIMASAnalyzer.monitoring_buffer`

---

### Alert System (EIMAS 5.2)

**EIMAS:**
- Metric-based alerting

**Cogman:**
- ✅ Operational status mapping
- ✅ Reason-based alerts
- ✅ Multi-level alerts (INFO, WARNING, CRITICAL)

**Method:**
- `EIMASAnalyzer.assess_operational_status()`
- `EIMASAnalyzer.get_alerts()`
- `EIMASAnalyzer.alert_history`

---

### Threshold Configuration (EIMAS 5.3)

**EIMAS:**
- Configurable thresholds

**Cogman:**
- ✅ similarity / anomaly thresholds
- ✅ Per-metric thresholds
- ✅ Extendable configuration design

**Method:**
- `EIMASAnalyzer.configure_thresholds()`
- Constructor parameters

---

### Historical Tracking & Versioning (EIMAS 5.4)

**EIMAS:**
- Version comparison

**Cogman:**
- ✅ Baseline vs current
- ✅ Metric history buffers
- ✅ Version comparison

**Method:**
- `EIMASAnalyzer.compare_versions()`
- Stored baselines + trend metrics

---

## 5. Security, Privacy & Governance

### Local Processing (EIMAS 6.1)

**EIMAS:**
- Sensitive data on-prem

**Cogman:**
- ✅ Fully local Python execution
- ✅ No external calls required

---

### Audit Logging (EIMAS 6.4)

**EIMAS:**
- Traceable decisions

**Cogman:**
- ✅ Deterministic metric outputs
- ✅ Explainable reason lists
- ✅ Alert history
- ✅ Embedding lineage

**Artifact:**
- Generated reports (.txt, JSON-ready data)
- `EIMASAnalyzer.alert_history`
- `EIMASAnalyzer.embedding_lineage`

---

## 6. Decision Support & Interpretability

### Explainability Tools (EIMAS 7.1)

**EIMAS:**
- Explainable detection

**Cogman:**
- ✅ Reason breakdown per status
- ✅ Metric contribution traceable
- ✅ Clustering explanation
- ✅ Anomaly explanation

**Method:**
- `EIMASAnalyzer.explain_clustering()`
- `EIMASAnalyzer.explain_anomaly()`
- `EIMASAnalyzer.explain_metric_contribution()`
- `status.reasons`

---

### Confidence Scoring (EIMAS 7.2)

**EIMAS:**
- Confidence level of analysis

**Cogman:**
- ✅ Confidence percentage
- ✅ Multi-signal agreement logic
- ✅ Data sufficiency assessment

**Method:**
- `EIMASAnalyzer.assess_confidence()`
- `status.confidence`

---

### Comparative Analysis (EIMAS 7.3)

**EIMAS:**
- Compare time / source / version

**Cogman:**
- ✅ Group comparison
- ✅ Stress index comparison
- ✅ Version comparison

**Method:**
- `EIMASAnalyzer.comparative_analysis()`
- `EIMASAnalyzer.compare_versions()`

---

## 7. Evaluation & Improvement

### Performance Metrics (EIMAS 8.1)

**EIMAS:**
- Latency / coverage

**Cogman:**
- ✅ Lightweight computation
- ✅ Batch scaling friendly

---

### Human Feedback Loop (EIMAS 8.2)

**EIMAS:**
- Human-in-the-loop

**Cogman:**
- ⚠️ External integration recommended
- ✅ Supports threshold reconfiguration

---

### Bias Detection (EIMAS 8.3)

**EIMAS:**
- Bias awareness

**Cogman:**
- ✅ Cluster imbalance signals
- ✅ Subgroup drift via comparison

---

## 8. Reporting & Export

### Automated Reporting (EIMAS 10.1)

**EIMAS:**
- Systematic reports

**Cogman:**
- ✅ Auto-generated text reports
- ✅ EIMAS compliance report

**Method:**
- `EIMASAnalyzer.generate_eimas_report()`
- `EmbeddingQualityInspector.generate_report()`

---

### External Export / API (EIMAS 10.2)

**EIMAS:**
- System integration

**Cogman:**
- ✅ Structured dict outputs
- ✅ JSON-serializable
- ✅ JSON export

**Method:**
- `EIMASAnalyzer.export_json()`

---

## 9. Compliance Summary

| EIMAS Area | Cogman Coverage | Status |
|------------|----------------|--------|
| Core Analysis | ✅ Full | Complete |
| Specialized Inspections | ✅ Full | Complete |
| Monitoring | ✅ Partial | Extensible |
| Alerting | ✅ | Complete |
| Decision Support | ✅ | Complete |
| Security / Locality | ✅ | Complete |
| Governance | ✅ | Complete |
| Enterprise API | ⚠️ | External wrapper needed |

---

## 10. Positioning Statement

**Cogman Tools provide a reference implementation that demonstrates conformity with the Embedding Intelligence Monitoring & Analysis Specification (EIMAS).**

They do not define the standard, but show how the standard can be implemented in practice.

The tools are designed to be:
- **Extensible** - Easy to add new capabilities
- **Composable** - Can be used together or separately
- **Production-ready** - Suitable for real-world deployment
- **Transparent** - All decisions are explainable

---

## Implementation Notes

1. **Base Analyzers**: EIMASAnalyzer builds on top of `EmbeddingQualityInspector` and `BehavioralAnalyzer`, providing a unified interface.

2. **Monitoring**: Real-time monitoring is implemented via a rolling buffer that can be extended to support streaming.

3. **Specialized Inspections**: Advanced capabilities like forgery detection and hidden pattern detection are implemented as separate methods.

4. **Explainability**: All analysis results include interpretation and explanation fields.

5. **Extensibility**: The architecture allows for easy addition of new analysis capabilities without breaking existing functionality.

---

## Metrics Reference

### Embedding Quality Index (EQI)

**สูตร:**
```
EQI = (0.4 × S + 0.35 × H_score + 0.25 × info_ratio) × 100
```

**ตัวแปร:**
- **S** (Signal Quality): ความเรียบ/เสถียรของสัญญาณ (0-1)
- **H_score**: Entropy Score ที่ปรับแล้ว (optimal ~ 0.7)
- **info_ratio**: effective_dims / total_dims

**การตีความ:**
| EQI Range | ความหมาย |
|-----------|----------|
| 70-100 | ดีมาก (Good) |
| 50-70 | ปกติ (Normal) |
| 30-50 | ควรตรวจสอบ (Warning) |
| 0-30 | มีปัญหา (Bad) |

---

### Distribution Entropy (H)

**สูตร:**
```
H = Shannon_Entropy / Max_Entropy
```

**คุณสมบัติ:**
- ค่าอยู่ในช่วง 0-1 เสมอ
- H = 0: ค่าไม่กระจาย (collapsed)
- H = 1: กระจายสูงสุด
- Optimal: 0.6-0.8

---

### Signal Quality (S) - Signal Integrity

**สูตร:**
```
S = 0.30 × S_uniformity + 0.25 × S_active + 0.25 × S_shape + 0.20 × S_variance
```

**Components:**
| Component | สูตร | ความหมาย |
|-----------|------|----------|
| **S_uniformity** | unique_values / total_values | ความหลากหลายของค่า (ไม่ collapse) |
| **S_active** | active_dims / total_dims | สัดส่วน dimensions ที่มีค่า (ไม่ sparse) |
| **S_shape** | 1 / (1 + \|kurtosis\| / 3) | ความเป็น normal (ไม่มี outlier) |
| **S_variance** | exp(-\|CV - 1\| / 2) | ความสม่ำเสมอของ variance |

**การตีความ:**
- Good Embedding (random gaussian): S ≈ 0.7-0.9
- Bad Embedding (sparse/collapsed): S ≈ 0.1-0.4

---

### Anomaly Detection

**สูตร:**
```
z_score = |embedding - baseline_mean| / baseline_std
p95_z = percentile(z_scores, 95)
is_anomaly = (p95_z > threshold) OR (mean_z > threshold × 0.5)
```

**Stress Index:**
```
stress = (0.6 × mean(p95_z) + 0.4 × mean(mean_z)) / threshold
```

---

### Operational Status Thresholds

| Metric | WARNING | DEGRADED | UNSAFE |
|--------|---------|----------|--------|
| Anomaly Density | > 15% | > 30% | > 50% |
| Stress Index | > 1.5 | > 2.0 | > 3.0 |
| Distribution Shift | > 1.0 | > 2.0 | > 5.0 |

