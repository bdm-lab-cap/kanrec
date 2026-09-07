# KAN-REC

**Continuous Numerical Feature Encoding and Symbolic Scoring Extraction for Recommendation Systems**

> TFM · Máster en Big Data & Data Engineering · Universidad Complutense de Madrid  
> Autor: Pedro Antonio Martínez Sánchez  
> Tutores: Jorge Centeno · Alberto González

[![CI](https://github.com/TU_USUARIO/kanrec/actions/workflows/ci.yml/badge.svg)](https://github.com/TU_USUARIO/kanrec/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Microsoft Fabric](https://img.shields.io/badge/Microsoft-Fabric-blue.svg)](https://app.fabric.microsoft.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│  FUENTES DE DATOS                                               │
│  Criteo/Avazu CSV · Kafka (Docker) · Item Metadata API          │
└──────────────┬───────────────┬──────────────────┬──────────────┘
               ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  MICROSOFT FABRIC                                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Real-Time Intelligence                                 │   │
│  │  Eventstream (Kafka) → KQL Database → RT Dashboard      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Data Engineering                                       │   │
│  │  Spark MLlib (normalización + feature importance)       │   │
│  │  Data Pipelines (orquestación)                          │   │
│  │  OneLake / Delta Lake (feature store)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Data Science                                           │   │
│  │  ML Experiments (MLflow integrado)                      │   │
│  │  Checkpoints en OneLake                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Data Activator                                         │   │
│  │  Alerta automática si AUC < baseline − 0.005            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Power BI (integrado en Fabric)                         │   │
│  │  Curvas φ · Fórmula scoring · AUC · CTR Real-Time       │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
┌─────────────────────┐               ┌──────────────────────┐
│  Google Colab       │               │  Local (Mac)         │
│  KAN encoder        │               │  MongoDB             │
│  Symbolic extract.  │               │  Kafka Docker        │
│  → OneLake          │               │  GitHub Actions CI   │
└─────────────────────┘               └──────────────────────┘
```

---

## Cobertura del máster (12/13 asignaturas)

| Componente | Tecnología | Asignatura |
|-----------|-----------|-----------|
| KAN encoder + extracción simbólica | PyTorch + EfficientKAN | Deep Learning — Eduardo Fernández |
| Feature importance + normalización | Spark MLlib en Fabric | Spark — Pablo Villacorta |
| Streaming de clics | Kafka (Docker) + Fabric Eventstream | Kafka — Jorge Centeno |
| Feature store | OneLake + Delta Lake | Diseño de ingestas — Jorge Centeno |
| Arquitectura de datos | Microsoft Fabric | Arquitecturas de datos — Jorge Centeno |
| Orquestación cloud | Fabric Data Pipelines | Pipelines en Cloud — Alberto González |
| NoSQL symbolic store | MongoDB | NoSQL — Marlon Cárdenas |
| Modelado de resultados | Esquema Delta Tables | Modelado de datos — Gabriel Marín |
| Tracking experimentos | Fabric ML Experiments | Machine Learning — Elena Gavilán |
| Alertas MLOps | Data Activator (Reflex) | Productivización — Pablo Hidalgo |
| Dashboard ejecutivo | Power BI integrado en Fabric | — |
| CI/CD + packaging | GitHub Actions + pip | Aplicaciones en contenedores — Luis Piñón |

---

## Quick start

### 1. Entorno local

```bash
git clone https://github.com/TU_USUARIO/kanrec.git && cd kanrec
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install --no-deps /tmp/efficient-kan   # ver instalación en docs/
pip install -e ".[dev,streaming,api]"
```

### 2. Servicios locales

```bash
# Kafka
docker compose -f infra/docker-compose.yml up -d

# MongoDB
mongod --dbpath /usr/local/var/mongodb &

# Mock API
nohup uvicorn data.mock_api_server:app --port 8000 > /tmp/api.log 2>&1 &
```

### 3. Microsoft Fabric

1. Crear workspace `KAN-REC` en https://app.fabric.microsoft.com
2. Crear Lakehouse `kanrec_lakehouse`
3. Subir Criteo a `Files/raw/criteo_10m.tsv`
4. Ejecutar notebooks en orden: `01 → 02 → 03 → 04`
5. Descargar `Files/parquet/` a `data/delta_parquet/criteo/`

### 4. Entrenamiento (Google Colab)

Abre `notebooks/colab_training.ipynb` en Google Colab con GPU T4.
Sigue las celdas en orden. Los checkpoints se guardan en Google Drive
y se suben a OneLake con `fabric/upload_to_fabric.py`.

### 5. Tests

```bash
pytest tests/ -v --tb=short
```

---

## Estructura del repositorio

```
kanrec/
├── .github/workflows/ci.yml      # CI: pytest + MongoDB
├── data/
│   ├── download_criteo.sh         # Descarga dataset
│   └── mock_api_server.py         # FastAPI mock para metadata
├── infra/
│   ├── docker-compose.yml         # Kafka + Zookeeper
│   └── kafka_producer.py          # Replay Criteo → Kafka
├── fabric/                        # Notebooks Microsoft Fabric
│   ├── 01_spark_ingest_mlllib.py  # MLlib normalización + feature importance
│   ├── 02_eventstream_kafka.py    # Structured Streaming desde Eventstream
│   ├── 03_api_ingest.py           # Pull API → OneLake
│   ├── 04_feature_store.py        # Consolidación + exportación Parquet
│   ├── 05_ml_experiments.py       # Registro en Fabric ML Experiments
│   ├── 06_data_activator_setup.py # Configuración alertas AUC
│   └── upload_to_fabric.py        # SDK para subir resultados de Colab
├── powerbi/
│   └── README.md                  # Setup Power BI + medidas DAX
├── kanrec/                        # Paquete Python instalable
│   ├── encoder.py                 # KANNumericalEncoder
│   ├── model.py                   # KANRecModel
│   ├── data.py                    # KANRecDataModule
│   ├── symbolic.py                # SymbolicExtractor
│   ├── faithfulness.py            # FaithfulnessEvaluator
│   └── mongo_store.py             # MongoSymbolicStore
├── experiments/
│   ├── train.py                   # Entrenamiento principal
│   ├── symbolic_extraction.py     # Extracción + MongoDB
│   └── run_all.sh                 # Reproducción completa
├── notebooks/
│   └── colab_training.ipynb       # Notebook Google Colab listo para usar
├── tests/
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

| Dataset | Filas | Fuente |
|---------|-------|--------|
| Criteo Display Advertising | 45M | [Kaggle](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) |
| Avazu CTR | 40M | [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction) |

---

## Referencias

- Guo et al. (2021). **AutoDis**. KDD 2021. [arXiv:2012.08986](https://arxiv.org/abs/2012.08986)
- Liu et al. (2024). **KAN**. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Luo et al. (2024). **EfficientKAN**. [GitHub](https://github.com/blealtan/efficient-kan)
- Yang et al. (2025). **ShapKAN**. [arXiv:2510.01663](https://arxiv.org/abs/2510.01663)
- European Parliament (2024). **AI Act** — Regulation EU 2024/1689.

---

## Licencia

MIT
