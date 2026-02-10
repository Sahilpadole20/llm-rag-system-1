# LLM-RAG System for Edge-Fog-Cloud Deployment

A **Retrieval-Augmented Generation (RAG)** system for intelligent network deployment decisions using ML + Threshold algorithm with Gradient Boosting.

## Overview

This system combines:
- **FAISS Vector Store** for semantic document retrieval
- **Gradient Boosting ML Model** for deployment predictions
- **Groq LLM** (llama-3.3-70b-versatile) for natural language responses

Based on the paper: *"Adaptive Task Scheduling in Edge-Fog-Cloud with Network Failure Resilience"*

## Project Structure

```
llm-rag-system-1/
├── src/
│   ├── rag_pipeline_simple.py      # Main RAG pipeline (7 steps)
│   ├── train_paper_algorithm.py    # ML model training
│   ├── test_10_paper_algorithm.py  # Test on 10 scenarios
│   ├── main.py                     # Entry point
│   ├── network_data_loader.py      # Data loading utilities
│   ├── embeddings/                 # Embedding utilities
│   ├── llm/                        # Groq LLM integration
│   ├── rag/                        # RAG pipeline components
│   ├── retrieval/                  # Document & vector store
│   └── utils/                      # Config & logging
├── config/
│   ├── llm_config.yaml             # LLM settings
│   └── embedding_config.yaml       # Embedding settings
├── network_faiss_store/            # Trained models & vectors
│   ├── faiss.index                 # Vector embeddings
│   ├── metadata.pkl                # Document metadata
│   ├── tfidf_embedder.pkl          # TF-IDF vectorizer
│   ├── gradient_boosting_model.pkl # ML model
│   └── label_encoder.pkl           # Label encoder
├── results/                        # Test results
├── requirements.txt
└── README.md
```

## RAG Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING    → Load CSV (1000 network scenarios)        │
│  STEP 2: DOCUMENTS       → Create knowledge documents (102 docs)    │
│  STEP 3: CHUNKING        → Split into chunks (500 chars, 100 overlap)│
│  STEP 4: EMBEDDINGS      → TF-IDF + SVD vectorization               │
│  STEP 5: VECTOR STORE    → FAISS IndexFlatL2 for similarity search  │
│  STEP 6: ML MODEL        → Gradient Boosting (100% accuracy)        │
│  STEP 7: GROQ LLM        → Natural language response generation     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train RAG System
```bash
cd src
python rag_pipeline_simple.py
```

### Train ML Model Only
```bash
python train_paper_algorithm.py
```

### Test on 10 Scenarios
```bash
python test_10_paper_algorithm.py
```

## Dataset

**File**: `edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv`

| Feature | Description | Range |
|---------|-------------|-------|
| datarate_mbps | Network data rate | 0.25 - 45.6 Mbps |
| sinr | Signal-to-Noise Ratio | 1.79 - 21.89 dB |
| latency_ms | End-to-end latency | 5.56 - 204.26 ms |
| rsrp_dbm | Signal power | -130.71 to -97.21 dBm |
| cpu_demand | CPU usage | 1 - 45% |
| memory_demand | Memory usage | 100 - 1013 MB |

**Class Distribution**:
- Cloud: 46.4%
- Edge: 34.4%
- Fog: 19.2%

## Deployment Algorithm

### Threshold Rules (from Paper)

| Layer | Latency | Data Rate | CPU | Use Case |
|-------|---------|-----------|-----|----------|
| **Edge** | < 50 ms | > 30 Mbps | < 50% | Real-time, AR/VR |
| **Fog** | 50-200 ms | > 15 Mbps | - | Smart city, IoT |
| **Cloud** | > 200 ms | - | > 70% | Big data, ML training |

### Signal Quality Override
- RSRP < -120 dBm → Cloud (poor signal)
- SINR < 5 dB → Fog (interference)

## Performance

| Metric | Value |
|--------|-------|
| ML Model Accuracy | 100% |
| Feature Importance | latency_ms = 1.0 |
| Vector Store Size | 103 vectors |
| Embedding Dimension | 102 (TF-IDF) |

## API Configuration

Set Groq API key in environment:
```bash
export GROQ_API_KEY="your_api_key_here"  # Linux/Mac
set GROQ_API_KEY=your_api_key_here       # Windows
```

Or in code:
```python
os.environ["GROQ_API_KEY"] = "your_api_key_here"
```

## Requirements

- Python 3.10+
- scikit-learn
- faiss-cpu
- groq
- pandas
- numpy

## Files Description

| File | Purpose |
|------|---------|
| `rag_pipeline_simple.py` | Main 7-step RAG pipeline |
| `train_paper_algorithm.py` | Train Gradient Boosting model |
| `test_10_paper_algorithm.py` | Test on 10 sample scenarios |
| `faiss.index` | FAISS vector store (103 vectors) |
| `gradient_boosting_model.pkl` | Trained ML model |

## Results

Latest test results in `results/` folder:
- **Accuracy**: 90-100% on test scenarios
- **Latency prediction**: Most important feature
- **Response time**: 2-5 seconds per query with LLM
