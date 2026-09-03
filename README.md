# KAN-REC

**Continuous Numerical Feature Encoding and Symbolic Scoring Extraction for Recommendation Systems**

> TFM · Máster en Big Data & Data Engineering · Universidad Complutense de Madrid  
> Autor: Pedro Antonio Martínez Sánchez  
> Tutores: Jorge Centeno · Alberto González

[![CI](https://github.com/TU_USUARIO/kanrec/actions/workflows/ci.yml/badge.svg)](https://github.com/TU_USUARIO/kanrec/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)

---

## What is KAN-REC?

KAN-REC replaces the discretisation step of AutoDis (KDD 2021) with a **Kolmogorov-Arnold Network (KAN) encoder** that maps each raw numerical feature to an embedding via a learnable spline — no buckets, no step discontinuities. After training, the spline curves are pruned and symbolically fitted, producing a **closed-form scoring formula** that can be audited by non-technical stakeholders:

```
ŷ ≈ f(I1, I3, I7) where:
  φ_I1(x) ≈ 0.72·log(|x|+1) − 0.04   [stability 5/5  R²=0.982]
  φ_I3(x) ≈ 0.31·x + 0.01             [stability 4/5  R²=0.961]
  φ_I7(x) ≈ 0.55·√|x| + 0.08          [stability 5/5  R²=0.974]
```

This is relevant under **AI Act (EU 2024/1689)**: KAN-REC produces the intrinsic explanations that e-commerce recommendation systems must provide.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Sources: Criteo/Avazu CSV · Kafka · Item Metadata API  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Databricks + Spark MLlib                               │
│  StandardScaler · ChiSqSelector · Structured Streaming  │
│  Delta Lake feature store (train / val / test)          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  PyTorch · EfficientKAN                                 │
│  KAN numerical encoder → KAN interaction → DeepFM      │
│  MLflow tracking                                        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Symbolic extraction + MongoDB                          │
│  ShapKAN prune → scipy fit → kanrec.symbolic_results    │
│  Faithfulness eval: gap RMSE · stability · monotonicity │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Streamlit dashboard · pip package · CI/CD              │
└─────────────────────────────────────────────────────────┘
```

---

## Quick start (10 commands)

```bash
# 1. Clone and install
git clone https://github.com/TU_USUARIO/kanrec.git && cd kanrec
pip install -e ".[dev,dashboard,streaming,api]"

# 2. Start Kafka (Docker)
docker compose -f infra/docker-compose.yml up -d

# 3. Start mock API server (new terminal)
uvicorn data.mock_api_server:app --port 8000

# 4. Start MongoDB (if not already running)
mongod --dbpath /tmp/mongodb &

# 5. Download Criteo (requires Kaggle CLI configured)
bash data/download_criteo.sh

# 6. Run Databricks notebooks (see databricks/ directory)
#    Or use the pre-processed Parquet files if you have them

# 7. Train the model
python experiments/train.py --encoder kan-bspline --dataset criteo --seed 42

# 8. Extract symbolic formula and persist to MongoDB
python experiments/symbolic_extraction.py \
    --checkpoint checkpoints/best_kan-bspline_criteo_s42.pt \
    --dataset criteo --seed 42

# 9. Launch dashboard
streamlit run dashboard/app.py

# 10. Run full experiment suite (all baselines + symbolic)
bash experiments/run_all.sh
```

---

## Repository structure

```
kanrec/
├── .github/workflows/ci.yml     # CI: pytest + MongoDB service
├── data/
│   ├── download_criteo.sh       # Dataset download with MD5 check
│   └── mock_api_server.py       # FastAPI mock for item metadata
├── infra/
│   ├── docker-compose.yml       # Kafka + Zookeeper + Kafka UI
│   └── kafka_producer.py        # Criteo CSV → Kafka stream replay
├── databricks/
│   ├── 01_spark_ingest_mlllib.py  # MLlib normalisation + feature importance
│   ├── 02_streaming_kafka.py      # Spark Structured Streaming consumer
│   ├── 03_api_ingest.py           # API pull + Delta Lake merge
│   └── 04_feature_store.py        # Consolidation + Parquet export
├── kanrec/                       # Installable Python package
│   ├── encoder.py                # KANNumericalEncoder (core contribution)
│   ├── model.py                  # KANRecModel (encoder + interaction + head)
│   ├── data.py                   # KANRecDataModule (Parquet → DataLoader)
│   ├── symbolic.py               # SymbolicExtractor (prune → fit → persist)
│   ├── faithfulness.py           # FaithfulnessEvaluator (gap, stability, mono)
│   └── mongo_store.py            # MongoSymbolicStore (persistence + queries)
├── experiments/
│   ├── train.py                  # Main training script
│   ├── symbolic_extraction.py    # Extraction + MongoDB persistence
│   └── run_all.sh                # Reproduces full results table
├── dashboard/
│   └── app.py                    # Streamlit interactive explorer
├── tests/
│   ├── conftest.py
│   ├── test_encoder.py
│   ├── test_symbolic.py
│   ├── test_mongo_store.py
│   └── test_data.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Size | Source | Role |
|---------|------|--------|------|
| Criteo Display Advertising | 45M rows · 4.3 GB | [Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge) | Primary training |
| Avazu CTR | 40M rows · 1.2 GB | [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction) | External validation |

Both are public and anonymised. No confidentiality agreements required.

---

## Key results (expected)

| Encoder | AUC ↑ | Log-loss ↓ | Latency (ms/batch) |
|---------|-------|-----------|-------------------|
| Raw normalisation | baseline | baseline | baseline |
| AutoDis (KDD 2021) | +0.002 | −0.001 | +2 ms |
| **KAN-REC (ours)** | **±0.001 vs AutoDis** | **comparable** | **≤ AutoDis** |

Symbolic extraction results:
- R² > 0.95 for accepted operators
- Gap RMSE < 0.01 on test set
- Stability ≥ 4/5 seeds for dominant operators

---

## Curricula alignment (UCM MSc Big Data & Data Engineering)

| Component | Asignatura |
|-----------|-----------|
| Spark MLlib (normalisation + feature importance) | Spark — Pablo Villacorta |
| Spark Structured Streaming (Kafka consumer) | Kafka y procesamiento en tiempo real — Jorge Centeno |
| Delta Lake, ingestion design | Diseño de ingestas y lagos de datos — Jorge Centeno |
| Databricks, cloud pipelines | Pipelines de datos en Cloud — Alberto González |
| MongoDB symbolic store | Bases de datos NoSQL — Marlon Cárdenas |
| PyTorch, KAN, symbolic regression | Machine Learning · Deep Learning |
| CI/CD, pip packaging, Docker | Productivización — Pablo Hidalgo |
| Streamlit dashboard | Python para desarrolladores |

---

## Running tests

```bash
# Unit tests (no MongoDB required)
pytest tests/test_encoder.py tests/test_symbolic.py tests/test_data.py -v

# All tests including MongoDB (requires mongod on localhost:27017)
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=kanrec --cov-report=term-missing
```

---

## Databricks setup (Community Edition)

1. Go to [community.cloud.databricks.com](https://community.cloud.databricks.com) — free account
2. Create cluster: Runtime **14.3 LTS ML** (Spark 3.5, Python 3.11, MLflow 2.11)
3. Install libraries on cluster: `efficient-kan`, `fuxictr`, `confluent-kafka`
4. Connect Repos to this GitHub repository
5. Upload Criteo CSV to DBFS: `dbfs:/FileStore/kanrec/criteo_10m.tsv`
6. Run notebooks in order: `01 → 02 → 03 → 04`
7. Download `delta/parquet/` to local `data/delta_parquet/`

---

## References

- Guo et al. (2021). **AutoDis**. KDD 2021. [arXiv:2012.08986](https://arxiv.org/abs/2012.08986)
- Liu et al. (2024). **KAN: Kolmogorov-Arnold Networks**. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Liu et al. (2024). **KAN 2.0**. [arXiv:2408.10205](https://arxiv.org/abs/2408.10205)
- Luo et al. (2024). **EfficientKAN**. [GitHub](https://github.com/blealtan/efficient-kan)
- Yang et al. (2025). **ShapKAN**. [arXiv:2510.01663](https://arxiv.org/abs/2510.01663)
- Zhou et al. (2025). **CoxKAN**. PMC 2025.
- European Parliament (2024). **AI Act** — Regulation EU 2024/1689.

---

## License

MIT — see [LICENSE](LICENSE) for details.
