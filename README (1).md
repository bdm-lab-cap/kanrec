# KAN-REC

**Continuous Numerical Feature Encoding and Symbolic Scoring Extraction for Recommendation Systems**

> TFM · Máster en Big Data & Data Engineering · Universidad Complutense de Madrid  
> Autor: Pedro Antonio Martínez Sánchez  
> Tutores: Jorge Centeno · Alberto González · Septiembre 2026

[![CI](https://github.com/bdm-lab-cap/kanrec/actions/workflows/ci.yml/badge.svg)](https://github.com/bdm-lab-cap/kanrec/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Microsoft Fabric](https://img.shields.io/badge/Microsoft-Fabric-blue.svg)](https://app.fabric.microsoft.com)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://cloud.mongodb.com)
[![Confluent Cloud](https://img.shields.io/badge/Confluent-Cloud-red.svg)](https://confluent.cloud)

---

## What is KAN-REC?

KAN-REC replaces the discretisation step of **AutoDis** (KDD 2021) with a **Kolmogorov-Arnold Network (KAN) encoder** that maps each raw numerical feature to an embedding via a learnable B-spline — no buckets, no step discontinuities. After training, the spline curves are pruned and symbolically fitted, producing a **closed-form scoring formula** auditable by non-technical stakeholders:

```
ŷ ≈ f(I3, I11) where:
  φ_I3(x)  ≈ −0.0735·exp(x) + 0.0378   [R²=0.931, stability 3/3 seeds]
  φ_I11(x) ≈  0.1720·exp(x) − 0.1034   [R²=0.918, stability 3/3 seeds]
```

**AI Act relevance:** KAN-REC produces the intrinsic explanations that e-commerce recommendation systems must provide under EU Regulation 2024/1689 (Art. 13).

---

## Results

| Encoder | Dataset | Test AUC | Log-loss | Seeds |
|---------|---------|---------|---------|-------|
| Raw normalisation | 100K bal. | 0.885 | 0.532 | 1 |
| AutoDis (KDD 2021) | 100K bal. | 0.817 | 0.694 | 1 |
| **KAN-REC (ours)** | **100K bal.** | **0.897 ± 0.002** | **0.366** | **3** |
| KAN-REC (Colab GPU) | 8M natural | 0.8026 ± 0.0005 | 0.444 | 3 |

KAN-REC supera a AutoDis en **+0.066 AUC** y **-0.33 log-loss** en comparativa controlada.

Symbolic extraction: **10/13 fields** accept an operator with R² > 0.90. Fields I3 and I11 are **100% stable** across seeds. Exponential operator dominates all numerical fields.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SOURCES                                                        │
│  Criteo CSV (10M) · Confluent Cloud (Kafka) · Item Metadata API │
└──────────┬──────────────┬──────────────────┬───────────────────┘
           ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  MICROSOFT FABRIC — workspace KAN-REC                           │
│  01_spark_ingest_mlllib  → Delta Tables (8M/998K/1M rows)      │
│  02_streaming_kafka      → Confluent → Eventstream → KQL+Delta  │
│  03_api_ingest           → item_metadata (1,396 items)          │
│  04_training_kanrec      → KAN model + MLflow experiments       │
│  05_symbolic_extraction  → symbolic_results.json                │
│  06_stream_processing    → streaming_processed + CTR metrics    │
│  07_autodis_baseline     → AutoDis real (AUC=0.817)            │
│  08_comparativa_encoders → Raw vs AutoDis vs KAN-REC           │
│  09_mongodb_vector_search → embeddings KAN → Atlas $vectorSearch│
│                                                                 │
│  Real-Time Dashboard (KQL) · Power BI (3 pages)               │
│  Data Activator (4 rules) · Data Pipelines (orchestration)     │
└─────────────────────────────────────────────────────────────────┘
         ▼                              ▼
┌──────────────────┐       ┌───────────────────────────────────┐
│  Google Colab    │       │  MongoDB Atlas (ClusterKanrec)    │
│  GPU T4 14.6 GB  │       │  kanrec.symbolic_results          │
│  8M rows         │       │  kanrec.model_alerts              │
│  AUC=0.8026      │       │  kanrec.item_embeddings (VS 13D)  │
└──────────────────┘       └───────────────────────────────────┘
```

---

## Curricula coverage (12/13 subjects)

| Component | Technology | Subject |
|-----------|-----------|---------|
| KAN encoder + symbolic extraction | PyTorch + EfficientKAN | Deep Learning |
| Distributed normalisation | Spark MLlib in Fabric | Spark |
| Real-time click stream | Confluent Cloud + Fabric Eventstream | Kafka |
| Feature store | OneLake + Delta Lake | Data ingest design |
| Multi-source architecture | Microsoft Fabric | Data architectures |
| Cloud orchestration | Fabric Data Pipelines | Cloud pipelines |
| Symbolic store + Vector Search | MongoDB Atlas | NoSQL |
| Delta Table modelling | Delta Table schemas | Data modelling |
| Experiment tracking | Fabric ML Experiments | Machine Learning |
| CI/CD + pip packaging | GitHub Actions + setup.py | Productivisation |
| Executive dashboard + Real-Time | Power BI + KQL Dashboard | — |
| Containers + DevOps | Docker + GitHub Actions | Containers |

---

## Quick start (local)

```bash
# 1. Clone and setup (Intel Mac)
git clone https://github.com/bdm-lab-cap/kanrec.git && cd kanrec

# Secrets: never hardcoded. See SECURITY.md
cp .env.example .env      # fill in ATLAS_URI, CONFLUENT_API_KEY, CONFLUENT_API_SECRET

~/.pyenv/versions/3.11.9/bin/python -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2
pip install git+https://github.com/Blealtan/efficient-kan.git@7b6ce1c --no-deps
pip install numpy==1.26.4
pip install -e ".[dev,streaming,api]"

# 2. Start local services
docker compose -f infra/docker-compose.yml up -d     # Kafka
mongod --dbpath /usr/local/var/mongodb &              # MongoDB (optional)
nohup uvicorn data.mock_api_server:app --port 8000 & # Mock API

# 3. Run tests
pytest tests/test_encoder.py tests/test_symbolic.py tests/test_data.py -v

# 4. Send to Confluent Cloud
python infra/kafka_producer_confluent.py --max-rows 500 --delay-ms 50
```

## Microsoft Fabric setup

1. Create workspace `KAN-REC` at https://app.fabric.microsoft.com
2. Create Lakehouse `kanrec_lakehouse`
3. Upload `data/criteo_10m.tsv` to `Files/raw/`
4. Run notebooks in order: `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09`
5. Create Power BI report from `kanrec_lakehouse` SQL endpoint

## Google Colab (8M rows training)

1. Upload `data/criteo_10m.tsv` to Google Drive `/kanrec/`
2. Open `notebooks/kanrec_8M_training.ipynb` in Colab
3. Activate GPU (T4 or A100)
4. Run all cells

---

## Repository structure

```
kanrec/
├── .github/workflows/ci.yml
├── data/
│   ├── download_criteo.sh
│   ├── feature_selection.json
│   └── mock_api_server.py
├── infra/
│   ├── docker-compose.yml
│   ├── kafka_producer.py          # Docker Kafka producer
│   └── kafka_producer_confluent.py # Confluent Cloud producer
├── fabric/                        # Microsoft Fabric notebooks
│   ├── 01_spark_ingest_mlllib.py
│   ├── 02_streaming_kafka.py
│   ├── 03_api_ingest.py
│   ├── 04_training_kanrec.py
│   ├── 05_symbolic_extraction.py
│   ├── 06_stream_processing.py
│   ├── 07_autodis_baseline.py
│   ├── 08_comparativa_encoders.py
│   └── 09_mongodb_vector_search.py
├── kanrec/                        # Installable Python package
│   ├── encoder.py                 # KANNumericalEncoder
│   ├── model.py                   # KANRecModel
│   ├── data.py                    # KANRecDataModule
│   ├── symbolic.py                # SymbolicExtractor
│   ├── faithfulness.py            # FaithfulnessEvaluator
│   └── mongo_store.py             # MongoSymbolicStore (Atlas)
├── experiments/
│   ├── train.py
│   ├── symbolic_extraction.py
│   ├── mongodb_import.py          # Import results to MongoDB Atlas
│   └── run_all.sh
├── notebooks/
│   └── kanrec_8M_training.ipynb  # Google Colab GPU training
├── tests/
│   ├── test_encoder.py   (5/5 ✅)
│   ├── test_symbolic.py  (6/6 ✅)
│   └── test_data.py      (4/4 ✅)
├── setup.py
├── requirements.txt
└── README.md
```

---

## References

- Guo et al. (2021). **AutoDis**. KDD 2021. [arXiv:2012.08986](https://arxiv.org/abs/2012.08986)
- Liu et al. (2024). **KAN**. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Liu et al. (2024). **KAN 2.0**. [arXiv:2408.10205](https://arxiv.org/abs/2408.10205)
- Luo et al. (2024). **EfficientKAN**. [GitHub](https://github.com/blealtan/efficient-kan)
- Yang et al. (2025). **ShapKAN**. [arXiv:2510.01663](https://arxiv.org/abs/2510.01663)
- European Parliament (2024). **AI Act** — Regulation EU 2024/1689.

---

## License

MIT
