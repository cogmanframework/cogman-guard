# Cogman Tools 🔧

**Operational Analysis Tools for Embeddings & AI Systems**

ชุดเครื่องมือสำหรับ **ตรวจสอบสุขภาพ ความเสถียร และความเสี่ยงในการใช้งาน**
ของ Embedding และระบบ AI เชิงปฏิบัติ (production-oriented)

---

## 🎯 เป้าหมายของโปรเจกต์

Cogman Tools ถูกออกแบบมาเพื่อตอบคำถามเดียว:

**"ระบบ AI นี้ยังปลอดภัยและเชื่อถือได้สำหรับการใช้งานอยู่หรือไม่?"**

✅ ไม่ตัดสินความถูกต้องของคำตอบ  
✅ ไม่แทนที่ benchmark หรือ evaluation dataset  
✅ มุ่งเน้น operational health, behavior, และ risk signals

---

## 📦 โปรเจกต์ที่รวมอยู่

### 1. Embedding Quality Inspector 🔍

*(เดิม: Embedding Physics Inspector)*

เครื่องมือวิเคราะห์ **คุณภาพและโครงสร้างของ embedding vectors**
โดยใช้สถิติ สัญญาณ และการกระจายตัวของข้อมูล

---

### 2. Baseline Behavioral Analyzer 🔍

ระบบวิเคราะห์พฤติกรรมของ AI ตาม
**Baseline Behavioral Analysis Specification**

ใช้สำหรับ:
- ตรวจว่าโมเดล "เปลี่ยนพฤติกรรม" หรือไม่
- ตรวจการ drift / degradation
- ประเมินสถานะการใช้งานจริง (Operational Status)

---

### 3. EIMAS Analyzer 🔬

ระบบวิเคราะห์ Embedding เชิงลึกตาม
**Embedding Intelligence Monitoring & Analysis Specification (EIMAS v1.0)**

รวมความสามารถจาก Embedding Quality Inspector และ Behavioral Analyzer
พร้อมความสามารถเฉพาะทาง:

- ✅ **Specialized Inspections** - Reference verification, Forgery detection, Hidden patterns
- ✅ **Real-time Monitoring** - Streaming ingestion, Alert system
- ✅ **Decision Support** - Explainability, Confidence scoring, Comparative analysis
- ✅ **Version Tracking** - Model version comparison, Historical analysis

---

## ✅ คุณสมบัติหลัก

### 1. ตรวจสอบสุขภาพของ Embedding

- ✅ **Information Strength** – ความหนาแน่นของข้อมูลที่ใช้จริง
- ✅ **Distribution Entropy** – ความกระจัดกระจายของ embedding
- ✅ **Signal Quality Score** – ความเรียบ/เสถียรของสัญญาณ
- ✅ **Embedding Quality Index (EQI)** – คะแนนคุณภาพโดยรวม

**ใช้ตรวจ:**
- embedding เสีย
- embedding เสื่อม
- embedding ไม่เหมาะกับงาน

---

### 2. ตรวจจับปัญหาเชิงระบบ

- 🚨 Distribution ผิดปกติ
- 🚨 Noise สูงผิดปกติ
- 🚨 Embedding collapse
- 🚨 Overfitting / Underfitting indicators
- 🚨 Out-of-domain representations

---

### 3. Visualization ที่ครอบคลุม

- 📊 Multi-panel plots แสดง metric สำคัญ
- 🔍 Interactive 3D embedding space
- 📈 เปรียบเทียบหลายโมเดล / หลายเวอร์ชัน

---

### 4. รายงานอัตโนมัติ

- 📝 สรุปสถานะการใช้งาน
- 💡 คำแนะนำเชิงปฏิบัติ
- ⚠️ Warning และ Risk indicator

---

## 🚀 การติดตั้ง

```bash
# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

---

## 🧪 การใช้งานพื้นฐาน

### 1. ทดสอบเร็ว (Quick Test)

```bash
# Embedding Quality Inspector (เก่าที่สุด)
python tests/quick_test.py

# Baseline Behavioral Analyzer (กลาง)
python tests/quick_test_behavioral.py

# EIMAS Analyzer (ใหม่ที่สุด)
python -m cogman_tools.eimas_analyzer
```

### ทดสอบทั้งหมด (เรียงลำดับจากเก่าไปใหม่)

```bash
# รันทุก test suite เรียงลำดับจากเก่าไปใหม่
python tests/run_all_tests.py
```

**ลำดับการทดสอบ:**
1. **Embedding Quality Inspector** - ระบบเก่าที่สุด
2. **Baseline Behavioral Analyzer** - ระบบกลาง
3. **EIMAS Analyzer** - ระบบใหม่ที่สุด

### Test Suites

```bash
# Test suite สำหรับแต่ละระบบ
python tests/test_behavioral_analyzer.py  # Baseline Behavioral Analyzer
python tests/test_eimas_analyzer.py       # EIMAS Analyzer
```

---

### 2. วิเคราะห์ Embedding ของคุณเอง

```python
from cogman_tools import EmbeddingQualityInspector
import torch

inspector = EmbeddingQualityInspector()

embedding = torch.randn(10, 256)  # torch.Tensor หรือ numpy array
result = inspector.analyze_embedding(embedding)

inspector.visualize(result)
report = inspector.generate_report(
    [embedding],
    save_path="outputs/reports/embedding_quality_report.txt"
)
```

---

### 3. เปรียบเทียบหลาย Embeddings / โมเดล

```python
embeddings = [emb1, emb2, emb3]
labels = ["Model A", "Model B", "Model C"]

comparison = inspector.compare_embeddings(embeddings, labels)
```

---

### 4. ใช้งานกับ HuggingFace Models

```python
from transformers import AutoModel, AutoTokenizer
from cogman_tools.embedding_inspector import inspect_model_embeddings

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

results = inspect_model_embeddings(
    model,
    tokenizer,
    sample_texts=["Hello world", "Testing embeddings"]
)
```

---

## 🔌 Embedding Providers

ระบบรองรับหลาย embedding providers:

- ✅ **Ollama** - สำหรับโมเดล local (nomic-embed-text, all-minilm, etc.)
- ✅ **OpenAI** - text-embedding-3-small, text-embedding-ada-002, etc.
- ✅ **Gemini** - text-embedding-004, embedding-001, etc.
- ⚠️ **Claude** - ต้องตรวจสอบ API availability

### การใช้งาน Provider

```python
from cogman_tools import get_provider, EmbeddingQualityInspector

# 1. สร้าง provider
# Ollama (local)
provider = get_provider('ollama', 'nomic-embed-text')

# OpenAI
provider = get_provider('openai', 'text-embedding-3-small', api_key='sk-...')

# Gemini
provider = get_provider('gemini', 'text-embedding-004', api_key='AIza...')

# 2. สร้าง embeddings
text = "Hello, world"
embedding = provider.get_embedding(text)  # numpy array
embedding_tensor = provider.get_embedding_tensor(text)  # torch.Tensor

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = provider.get_embeddings_batch(texts)

# 3. วิเคราะห์คุณภาพ
inspector = EmbeddingQualityInspector()
result = inspector.analyze_embedding(embedding)
print(f"EQI: {result['embedding_quality_index']:.3f}")

# 4. Validate embedding
validation = provider.validate_embedding(embedding)
print(f"Valid: {validation['valid']}")
```

### Provider Demo

```bash
# รัน demo สำหรับทุก providers
python examples/provider_demo.py
```

---

## 🏗️ System Architecture (High-Level)

```text
Input (Raw Embeddings from OpenAI / Ollama / Gemini / Claude)
        ↓
Layer 1: Quality Inspector
        - Signal Integrity (S), Entropy (H), EQI
        - Circuit + statistical checks
        ↓
Layer 2: Behavioral Analyzer
        - Baseline comparison (Drift / Anomaly / Cluster)
        - Operational Status (NORMAL / WARNING / DEGRADED / UNSAFE)
        ↓
Layer 3: EIMAS Core
        - Specialized inspections (forgery, hidden patterns)
        - Monitoring, Alerts, Thresholds
        ↓
Output
        - Status + Alerts + Reports
```

---

## 📊 Metrics ที่ใช้ (Non-physics)

| Metric | ช่วงค่า | ความหมาย |
|--------|---------|----------|
| **Information Strength (I)** | 1 to N | จำนวน dimensions ที่มีข้อมูลจริง (effective rank, N = total dimensions) |
| **Distribution Entropy (H)** | 0 - 1 | ความกระจายของค่า embedding (0=ไม่กระจาย, 1=กระจายมาก) |
| **Signal Quality (S)** | 0 - 1 | ระดับความสมบูรณ์ของสัญญาณ (Signal Integrity, 1=ดีมาก) |
| **Embedding Quality Index (EQI)** | 0 - 100 | คะแนนคุณภาพโดยรวม (สูง=ดี) |
| **Anomaly Density** | 0 - 1 | สัดส่วนของ embeddings ที่ผิดปกติ |
| **Stress Index** | 0 - 10+ | ระดับความเสี่ยงเชิงโครงสร้าง (ต่ำ=ดี) |
| **Drift Slope** | -∞ to +∞ | ความเร็วของการเปลี่ยนพฤติกรรม (0=เสถียร) |

### 📐 สูตรคำนวณ EQI (Embedding Quality Index)

```
EQI = (0.4 × S + 0.35 × H_score + 0.25 × info_ratio) × 100
```

โดย:
- **S** = Signal Quality Score (0-1)
- **H_score** = Entropy Score ที่ปรับแล้ว (optimal ~ 0.7)
- **info_ratio** = effective_dims / total_dims

**การตีความ EQI:**
- 70-100: ดีมาก (Good)
- 50-70: ปกติ (Normal)
- 30-50: ควรตรวจสอบ (Warning)
- 0-30: มีปัญหา (Bad)

---

## 🌍 Real-world Simulation Results

รัน `python tests/scenario_simulation.py`

```text
Scenario 1: Model Collapse (Silent Failure)
  Signal Quality: ปกติ ~0.96, พัง ~0.008 (แยกชัดเจน)
  Alerts Triggered: 8

Scenario 2: Gradual Concept Drift
  Month 1: Shift = 2.94 █████           | Status: DEGRADED (early warning)
  Month 4: Shift = 6.03 ████████████    | Status: UNSAFE
  Month 5: Shift = 7.46 ██████████████  | Status: UNSAFE

Scenario 3: Anomaly Spike (Potential Attack)
  Anomaly Density: 100%
  Stress Index: 2.54 (⚠️ CRITICAL)
```

หมายเหตุ: Drift ใช้ batch ขนาด 10 ต่อรอบ เพื่อให้การวัด distribution เสถียรและไม่เกิด divide-by-zero

---

## 🧠 Baseline Behavioral Analyzer

ระบบวิเคราะห์พฤติกรรมตาม **Baseline Behavioral Analysis Specification**

### โมดูลที่รองรับ

- ✅ **Similarity Analysis**
- ✅ **Cluster Analysis**
- ✅ **Anomaly Detection**
- ✅ **Trend Analysis**
- ✅ **Cross-modal Analysis**
- ✅ **Operational Status Assessment**

---

### Operational Status

ระบบจะประเมินสถานะแบบตรงไปตรงมา:

| Status | เงื่อนไข | คำแนะนำ |
|--------|----------|---------|
| **NORMAL** | anomaly_density < 15%, stress < 1.5, shift < 1.0 | ใช้งานได้ปกติ |
| **WARNING** | anomaly_density 15-30%, stress 1.5-2.0, shift 1.0-2.0 | ตรวจสอบเพิ่มเติม |
| **DEGRADED** | anomaly_density 30-50%, stress 2.0-3.0, shift 2.0-5.0 | ควรลดการใช้งาน |
| **UNSAFE** | anomaly_density > 50%, stress > 3.0, shift > 5.0 | ไม่ควรใช้งาน |

**หมายเหตุ:**
- Status เหล่านี้ **ไม่ใช่การตัดสินความถูกต้องของผลลัพธ์**
- แต่เป็นสัญญาณเชิงปฏิบัติสำหรับผู้ใช้งาน
- ค่า threshold สามารถปรับได้ตามความเหมาะสมของแต่ละ use case

---

### ตัวอย่างการใช้งาน

```python
from cogman_tools import BehavioralAnalyzer
import numpy as np

baseline = [np.random.randn(768) for _ in range(20)]
analyzer = BehavioralAnalyzer(baseline_embeddings=baseline)

test_embeddings = [np.random.randn(768) for _ in range(10)]
status = analyzer.assess_operational_status(test_embeddings)

print(status.status)
print(status.reasons)
```

---

## 🔬 EIMAS Analyzer

ระบบวิเคราะห์ Embedding เชิงลึกตาม **Embedding Intelligence Monitoring & Analysis Specification (EIMAS v1.0)**

### คุณสมบัติหลัก

- ✅ **Core Analysis** - Similarity, Cluster, Anomaly, Trend, Cross-modal
- ✅ **Specialized Inspections** - Reference verification, Forgery detection, Hidden patterns, Propagation tracking
- ✅ **Real-time Monitoring** - Streaming ingestion, Alert system, Threshold configuration
- ✅ **Decision Support** - Explainability tools, Confidence scoring, Comparative analysis
- ✅ **Version Tracking** - Model version comparison, Historical analysis

### การใช้งานพื้นฐาน

```python
from cogman_tools import EIMASAnalyzer
import numpy as np

# สร้าง analyzer
baseline = [np.random.randn(768) for _ in range(20)]
analyzer = EIMASAnalyzer(
    baseline_embeddings=baseline,
    enable_monitoring=True
)

# Comprehensive analysis
embeddings = [np.random.randn(768) for _ in range(10)]
results = analyzer.comprehensive_analysis(embeddings)

# Specialized inspections
forgery_result = analyzer.imitation_forgery_detection(embeddings)
hidden_patterns = analyzer.hidden_communication_pattern_detection(embeddings)

# Real-time monitoring
for i, emb in enumerate(embeddings):
    ingest_result = analyzer.ingest_embedding(emb, embedding_id=f"emb_{i}")
    alerts = analyzer.get_alerts(level='WARNING')

# Explainability
explanation = analyzer.explain_anomaly(embeddings[0], baseline)
confidence = analyzer.assess_confidence(results)

# Generate report
report = analyzer.generate_eimas_report(embeddings, save_path='outputs/reports/eimas_report.txt')
```

### EIMAS Compliance

Cogman Tools เป็น **reference implementation** ที่สอดคล้องกับ EIMAS Specification

ดูรายละเอียดการ mapping ใน `docs/EIMAS_MAPPING.md`

---

## 🚨 What This Tool Is NOT

- ❌ ไม่บอกว่าโมเดล "ถูกหรือผิด"
- ❌ ไม่แทน benchmark / eval metrics
- ❌ ไม่ตรวจ internals ของโมเดล

**Cogman Tools วิเคราะห์ พฤติกรรมและความเสี่ยงในการใช้งานจริง**

---

## 👥 Who Should Use This

- ML Engineers
- MLOps / Platform Teams
- AI QA & Safety Teams
- Researchers ที่ต้องดู behavior ของ embedding

---

## 📄 Spec

- `docs/BASELINE_BEHAVIORAL_ANALYSIS_SPEC.md` - Baseline Behavioral Analysis Specification
- `docs/EIMAS_MAPPING.md` - Mapping ระหว่าง Cogman Tools กับ EIMAS Specification

---

## 📁 Project Structure

ดูโครงสร้างโปรเจกต์ใน `PROJECT_STRUCTURE.md`

---

## 📜 License

MIT License

---

## ✅ สรุปตรง ๆ

เวอร์ชันนี้:

- ✅ ไม่มีศัพท์ฟิสิกส์ = ไม่งง
- ✅ ภาษา production / operational
- ✅ ไม่อวด ไม่ขายฝัน
- ✅ วางตัวเป็น "มาตรฐานตรวจสอบ" ได้ชัด

---

## 🌐 Language Versions

- 🇹🇭 [README.md](README.md) - Thai version (this file)
- 🇬🇧 [README_EN.md](README_EN.md) - English version
