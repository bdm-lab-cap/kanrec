# Anexos — KAN-REC (v2)

**Trabajo Fin de Máster · Máster en Big Data & Data Engineering**
Pedro Antonio Martínez Sánchez · Tutores: Jorge Centeno y Alberto González

---

# Anexo A. Cobertura de asignaturas del máster

| Asignatura | Aplicación en el proyecto | Artefacto |
|---|---|---|
| **Fundamentos de Big Data** | Dataset de 10 M de filas y 2,26 GB que excede el procesamiento en una sola máquina; decisiones de particionado y materialización. | `fabric/01_spark_ingest_mlllib.py` |
| **Procesamiento distribuido (Spark)** | Ingesta y normalización distribuidas: esquema explícito, `randomSplit`, agregaciones para estadísticos de escalado, ventanas y `join` en difusión para el indexado de 26 variables categóricas (hasta 409.063 categorías). Materialización por bloques para acotar el plan de ejecución. | `fabric/01`, `kanrec/spark_utils.py` |
| **Arquitecturas de datos / Lakehouse** | Lakehouse en OneLake con tablas Delta gestionadas, transaccionalidad ACID, versionado e historial de escrituras. Separación `Files` (artefactos) / `Tables` (datos consultables). | Lakehouse `kanrec_lakehouse` |
| **Streaming y tiempo real** | Productor Kafka a Confluent Cloud (GCP), ingesta con Eventstream de Fabric, escritura simultánea a Delta y base de datos KQL, métricas agregadas y alertas con Data Activator. Consistencia batch/stream mediante estadísticos persistidos. | `data/kafka_producer_confluent.py`, Eventstream, `fabric/06` |
| **Bases de datos NoSQL** | MongoDB Atlas para resultados simbólicos (documentos de esquema variable), *aggregation pipelines* para el informe de estabilidad, e índice vectorial para búsqueda por similitud de embeddings. | `kanrec/mongo_store.py`, `fabric/09` |
| **Integración de datos / APIs** | Servicio FastAPI de metadatos, ingesta paginada y enriquecimiento por unión con la tabla principal. | `fabric/03`, `data/mock_api_server.py` |
| **Machine Learning** | Encoder KAN sobre splines B-spline, baselines `raw` y AutoDis, protocolo de evaluación con tres semillas y análisis de significancia, ablación de hiperparámetros. | `kanrec/encoder.py`, `kanrec/baselines.py` |
| **MLOps** | Seguimiento con MLflow, versionado de checkpoints, CI con GitHub Actions (escaneo de secretos y 86 tests), paquete instalable, gestión de secretos con Key Vault y variables de entorno. | `.github/workflows/ci.yml`, `kanrec/config.py` |
| **Visualización y BI** | Panel en Power BI sobre Delta con Direct Lake: comparativa de encoders, curvas φ y CTR del flujo de streaming. | Informe Power BI |
| **Ingeniería de software** | Paquete estructurado, 86 tests unitarios y de integración, tipado, documentación y trazabilidad de cada corrección a su test de regresión. | `tests/`, `README.md` |

---

# Anexo B. Diagrama de arquitectura detallado

## B.1. Flujo de datos completo

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌────────────────────┐
│ FUENTES             │     │ INGESTA Y PROCESO        │     │ ALMACENAMIENTO     │
├─────────────────────┤     ├──────────────────────────┤     ├────────────────────┤
│ Criteo TSV          │────►│ 01_spark_ingest          │────►│ Tables/train       │
│ 10.000.001 filas    │     │  · esquema explícito     │     │ Tables/val         │
│ 2,26 GB             │     │  · nulos → 0, abs()      │     │ Tables/test        │
│ 13 num + 26 cat     │     │  · split 80/10/10        │     │ Files/config/      │
│                     │     │  · log1p  I1–I5          │     │  scaler_stats.json │
│                     │     │  · StdScaler I6–I13      │     │ Tables/            │
│                     │     │  · indexado categórico   │     │  cat_index_maps    │
├─────────────────────┤     ├──────────────────────────┤     ├────────────────────┤
│ Confluent Cloud     │────►│ Eventstream              │────►│ Tables/            │
│ tópico              │     │  ↓                       │     │  streaming_kfk     │
│ ad_impressions      │     │ 06_stream_processing     │────►│  streaming_        │
│ (Kafka, GCP)        │     │  (mismos estadísticos)   │     │   processed        │
│                     │     │                          │     │ KQL Database       │
├─────────────────────┤     ├──────────────────────────┤     ├────────────────────┤
│ Mock API REST       │────►│ 03_api_ingest            │────►│ Tables/            │
│ FastAPI             │     │  (paginado + join)       │     │  train_enriched    │
└─────────────────────┘     └──────────────────────────┘     └────────────────────┘
                                        │
                                        ▼
                            ┌──────────────────────────┐
                            │ 04_model_comparison      │
                            │  raw │ AutoDis │ KAN-REC │
                            │  3 semillas · MLflow     │
                            └──────────────────────────┘
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
                ┌──────────────────────┐  ┌──────────────────────┐
                │ 05_symbolic          │  │ Power BI             │
                │  · poda L1           │  │  (Direct Lake)       │
                │  · ajuste 16 dims    │  │  · comparativa       │
                │  · fidelidad         │  │  · curvas φ          │
                │  · monotonía         │  │  · CTR streaming     │
                └──────────┬───────────┘  └──────────────────────┘
                           ▼
                ┌──────────────────────┐
                │ MongoDB Atlas        │
                │  symbolic_results    │
                │  item_embeddings     │
                │  (+ vector search)   │
                └──────────────────────┘
```

## B.2. Arquitectura del modelo

```
x_num [B, 13]                              x_cat [B, 26]
     │                                          │
     ▼                                          ▼
┌─────────────────────────┐          ┌─────────────────────┐
│ ENCODER NUMÉRICO        │          │ 26 × nn.Embedding   │
│ (la variable del        │          │ (card+1, 16)        │
│  experimento)           │          └─────────┬───────────┘
│                         │                    │
│ raw:     Linear(1→16)   │                    │
│ AutoDis: meta-emb + att │                    │
│ KAN:     φⱼ spline      │                    │
│   · clamp ±10σ          │                    │
│   · SiLU(x)·w_base      │                    │
│   · B-spline(G=10,k=3)  │                    │
└───────────┬─────────────┘                    │
            │ [B, 13, 16]                      │ [B, 26, 16]
            └──────────────┬───────────────────┘
                           ▼
                    concat [B, 39, 16] → flatten [B, 624]
                           │
                           ▼
              ┌────────────────────────────┐
              │ INTERACCIÓN (compartida)   │
              │ Linear(624→256) → ReLU     │
              │ Dropout(0,1)               │
              │ Linear(256→64)  → ReLU     │
              └────────────┬───────────────┘
                           ▼
                    Linear(64→1) → logits [B, 1]
                           │
                    BCEWithLogitsLoss (entrenamiento)
                    sigmoid (inferencia)
```

## B.3. Vectorización del encoder

```
ANTES (bucle: 13 lanzamientos de kernel)     DESPUÉS (1 kernel)
─────────────────────────────────────        ──────────────────────────────
for j in range(13):                          bases = b_splines(x)
    xj = x[:, j:j+1]                             → [B, 13, n_coef]
    ej = field_kans[j](xj)                   out = bmm(bases.T, W)
    out.append(ej)                               W: [13, n_coef, 16]
                                                 → [13, B, 16] → [B, 13, 16]
8,83 ms/lote (84,5 % del total)              0,95 ms/lote (35,5 %)
                                             Equivalencia en float32
```

---

# Anexo C. Configuración experimental detallada

## C.1. Hiperparámetros del modelo

| Parámetro | Valor | Justificación |
|---|---|---|
| `embedding_dim` | 16 | Compartido por encoders numéricos y categóricos. |
| `grid_size` | 10 | Óptimo medido; 5 pierde 0,0078 de AUC, 20 diverge. |
| `spline_order` | 3 | B-splines cúbicas, estándar en la literatura KAN. |
| `input_clip` | 10 σ | Winsorizado; preserva el 99,99 % de los datos. |
| Interacción | 624 → 256 → 64 | Capa compartida por los tres encoders. |
| `dropout` | 0,1 | En la capa de interacción. |
| AutoDis `n_buckets` | 20 | Meta-embeddings por campo. |
| AutoDis `temperature` | aprendida (softplus) | Parametrizada para garantizar positividad. |

## C.2. Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | Adam, `weight_decay` = 1e-5 |
| Learning rate base | 1e-3 (`raw`, KAN) · 1e-2 (AutoDis) |
| Multiplicador lr del spline | 25× |
| `entropy_reg_weight` | 1e-5 |
| Función de pérdida | `BCEWithLogitsLoss` |
| Recorte de gradiente | norma máxima 1,0 |
| Batch size | 2048 (entrenamiento) · 4096 (latencia) |
| Épocas máximas | 30 |
| Parada temprana | paciencia 3 sobre AUC de validación |
| Planificador | `ReduceLROnPlateau`, factor 0,5 |
| Calibración del grid | 50.000 filas, en CPU |
| Semillas | 42, 123, 256 |

## C.3. Configuración de datos

| Parámetro | Valor |
|---|---|
| Filas totales | 10.000.001 |
| Partición | 80 / 10 / 10 con semilla 42 |
| Tamaños resultantes | 7.998.378 / 1.000.109 / 1.001.514 |
| Muestra para la comparativa | 1.500.000 filas |
| Muestra para fidelidad | 300.000 filas |
| Transformación I1–I5 | `log1p(max(x, 0))` |
| Transformación I6–I13 | Estandarización (ajustada solo en train) |
| Indexado categórico | Frecuencia descendente, desempate alfabético |
| Categorías no vistas | Índice reservado `len(categorías)` |
| CTR del conjunto | ≈ 25 % |

## C.4. Entorno de ejecución

| Componente | Especificación |
|---|---|
| Fabric — pool Spark | Small (4 vCores), capacidad de prueba |
| Colab — GPU | NVIDIA T4 |
| Python | 3.11 |
| PyTorch | *(completar con la versión del entorno)* |
| Paquete `kanrec` | 0.3.2 |

> Las versiones exactas de las librerías se obtienen con `pip freeze` en el entorno de ejecución y se recogen en `requirements.lock` del repositorio.

## C.5. Reproducción de los experimentos

```bash
# Instalación
git clone https://github.com/bdm-lab-cap/kanrec.git && cd kanrec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"

# Tests
pytest tests/ -q                       # 86 tests

# Comparativa completa (3 encoders × 3 semillas)
bash experiments/run_all.sh

# Extracción simbólica
python experiments/symbolic_extraction.py

# Figura de validación con formas conocidas
PYTHONPATH=. python experiments/figura_validacion_simbolica.py
```

En Fabric, el orden lo impone el pipeline `kanrec_end_to_end`: `01` → (`03`, `04`, `06` en paralelo) → `05` → `09`. El streaming lo alimenta Eventstream de forma continua desde Confluent, al margen de esa secuencia.
En Colab: `kanrec_full_comparison.ipynb` → `kanrec_analisis_cierre.ipynb`.

---

# Anexo D. Evidencias de ejecución

## D.1. Verificación de la normalización (notebook 01, Fabric)

Salida sobre las 7.998.378 filas de entrenamiento:

| Columna | Media | Desviación | Mínimo | Máximo |
|---|---|---|---|---|
| I6 | 3,65·10⁻¹⁶ | 1,0000000000000009 | −0,272 | 700,20 |
| I7 | 8,18·10⁻¹⁷ | 0,9999999999999992 | −0,237 | 424,62 |
| I8 | −1,11·10⁻¹⁶ | 1,000000000000002 | −0,595 | 240,12 |
| I9 | −8,41·10⁻¹⁷ | 1,0000000000000004 | −0,466 | 113,23 |
| I10 | 9,82·10⁻¹⁵ | 0,999999999999999 | −0,570 | 14,74 |
| I11 | −2,14·10⁻¹⁶ | 0,9999999999999998 | −0,503 | 36,42 |
| I12 | 6,40·10⁻¹⁷ | 0,9999999999999994 | −0,085 | 701,35 |
| I13 | −1,89·10⁻¹⁶ | 1,0 | −0,327 | 353,02 |

Los máximos de 700 desviaciones típicas en I6 e I12 son los que motivan el winsorizado descrito en 5.2.

## D.2. Comparativa de encoders (Colab, 3 semillas)

| Encoder | Semilla | AUC test | Log-loss | Épocas | Tiempo |
|---|---|---|---|---|---|
| raw | 42 | 0,7830 | 0,4637 | 8 | 136 s |
| raw | 123 | 0,7848 | 0,4609 | 9 | 148 s |
| raw | 256 | 0,7846 | 0,4645 | 9 | 147 s |
| autodis | 42 | 0,7806 | 0,4684 | 9 | 207 s |
| autodis | 123 | 0,7829 | 0,4629 | 8 | 180 s |
| autodis | 256 | 0,7813 | 0,4641 | 9 | 198 s |
| kan-bspline | 42 | 0,7843 | 0,4620 | 9 | 257 s |
| kan-bspline | 123 | 0,7842 | 0,4646 | 9 | 259 s |
| kan-bspline | 256 | 0,7868 | 0,4607 | 9 | 257 s |

## D.3. Extracción simbólica (semilla 42)

| Campo | Operador | R² medio | Acuerdo entre dims | Fórmula |
|---|---|---|---|---|
| I2 | linear | 0,9973 ± 0,0018 | 100 % | 0,1269·x − 0,0085 |
| I3 | linear | 0,9973 ± 0,0018 | 100 % | −0,7210·x + 0,0319 |
| I4 | linear | 0,9828 ± 0,0329 | 94 % | 0,2898·x − 0,0483 |
| I5 | linear | 0,9794 ± 0,0403 | 94 % | −0,5175·x − 0,0223 |
| I6 | linear | 0,9193 ± 0,2131 | 88 % | 0,1307·x + 0,0720 |
| I7 | linear | 0,9971 ± 0,0021 | 100 % | 0,6246·x + 0,5612 |
| I8 | linear | 0,9977 ± 0,0000 | 100 % | 0,3635·x + 0,4371 |
| I9 | linear | 0,9972 ± 0,0005 | 100 % | 0,6891·x + 0,3927 |
| I11 | linear | 0,9965 ± 0,0043 | 100 % | 0,7315·x + 0,5387 |
| I13 | linear | 0,9978 ± 0,0006 | 100 % | −0,4651·x − 0,6063 |

## D.4. Auditoría de monotonía

| Campo | Dirección | Violación | Veredicto |
|---|---|---|---|
| I1 | creciente | 0,02 % ± 0,08 % | monótona |
| I2 | creciente | 2,74 % ± 10,61 % | monótona |
| I3 | decreciente | 0,88 % ± 3,40 % | monótona |
| I4 | decreciente | 2,95 % ± 11,41 % | monótona |
| I5 | decreciente | 1,55 % ± 4,96 % | monótona |
| I6 | decreciente | 4,54 % ± 2,40 % | monótona |
| I7 | creciente | 3,45 % ± 2,07 % | monótona |
| I8 | creciente | 2,88 % ± 1,33 % | monótona |
| I9 | creciente | 6,10 % ± 5,45 % | casi monótona |
| I10 | creciente | 1,05 % ± 3,20 % | monótona |
| I11 | decreciente | 5,18 % ± 2,45 % | casi monótona |
| I12 | decreciente | 1,94 % ± 1,16 % | monótona |
| I13 | decreciente | 3,60 % ± 0,78 % | monótona |

## D.5. Fidelidad por sustitución (3 semillas)

| Semilla | AUC original | AUC sustituido | Δ AUC | Error de curva | Máximo |
|---|---|---|---|---|---|
| 42 | 0,7249 | 0,6891 | +0,0358 | 0,0698 | 0,117 (I4) |
| 123 | 0,7211 | 0,6920 | +0,0291 | 0,0629 | 0,140 (I4) |
| 256 | 0,7259 | 0,6907 | +0,0353 | 0,0658 | 0,112 (I11) |
| **Media** | **0,7240** | **0,6906** | **+0,0334 ± 0,0037** | **0,0662 ± 0,0035** | — |

## D.6. Latencia antes y después de vectorizar (T4, lote 4096, 100 ejecuciones)

| Configuración | Modelo | Encoder | % encoder | Filas/s |
|---|---|---|---|---|
| raw | 2,58 ± 0,27 ms | 0,71 ms | 27,5 % | 1.589.091 |
| autodis | 4,44 ± 0,13 ms | 3,61 ms | 81,4 % | 923.550 |
| kan-bspline (original) | 11,08 ± 1,50 ms | 9,05 ms | 81,7 % | 369.805 |
| **kan-bspline (vectorizado)** | **2,68 ± 0,41 ms** | **0,95 ms** | **35,5 %** | **1.525.526** |

Equivalencia entre original y vectorizado: diferencia máxima **0,00** en la medición sobre GPU y **2,4·10⁻⁷** en CPU, en ambos casos dentro de la precisión de float32.

---

# Anexo E. Revisión de literatura

| Trabajo | Publicación | Relación con KAN-REC |
|---|---|---|
| **AutoDis** — Guo et al. | KDD 2021 | Baseline directo. Discretización suave mediante meta-embeddings y atención. KAN-REC lo sustituye por una spline continua. |
| **KAN** — Liu et al. | arXiv 2404.19756 (2024) | Formulación original de las Kolmogorov-Arnold Networks. Base teórica del encoder. |
| **KAN 2.0** — Liu et al. | arXiv 2408.10205 (2024) | API de ajuste simbólico y regularización por entropía. Referencia procedimental de la extracción. |
| **EfficientKAN** — Blealtan | GitHub (2024) | Implementación con evaluación vectorizada de B-splines. Vendorizada en el paquete por no estar publicada en PyPI. |
| **KarSein** — Shi et al. | arXiv 2408.08713 (2024) | Aplica KAN a la capa de *interacción* en CTR, no a la codificación. Ortogonal a este trabajo. |
| **CF-KAN** — Yang et al. | arXiv 2409.05633 (2024) | Autoencoder KAN para filtrado colaborativo. Sin variables numéricas ni extracción simbólica. |
| **TabKANet** — Liu et al. | ScienceDirect (2025) | Encoder KAN para datos tabulares genéricos. No aborda recomendación ni extracción. |
| **CoxKAN** — Zhou et al. | PMC (2025) | Extracción simbólica de KAN en análisis de supervivencia. Inspiración metodológica del pipeline de extracción. |
| **ShapKAN** — Yang et al. | arXiv 2510.01663 (2025) | Poda por importancia basada en valores de Shapley. Alternativa al criterio por percentil, propuesta como línea futura. |
| **DeepFM** — Guo et al. | IJCAI 2017 | Arquitectura de referencia para CTR; inspira el backbone empleado. |

**Posicionamiento.** La combinación de encoder numérico continuo basado en KAN con extracción simbólica de la codificación, aplicada a sistemas de recomendación, no aparece en la literatura consultada. Los trabajos con KAN en recomendación (KarSein, CF-KAN) actúan sobre la interacción o el filtrado colaborativo; los que aplican extracción simbólica (CoxKAN) lo hacen en otros dominios.

---

# Anexo F. Bibliografía

- Guo, H., Chen, B., Tang, R., Zhang, W., Li, Z. y He, X. (2021). *AutoDis: Automatic Discretization for Embedding Numerical Features in CTR Prediction.* KDD 2021. arXiv:2012.08986.
- Guo, H., Tang, R., Ye, Y., Li, Z. y He, X. (2017). *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.* IJCAI 2017. arXiv:1703.04247.
- Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y. y Tegmark, M. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756.
- Liu, Z., Ma, P., Wang, Y., Matusik, W. y Tegmark, M. (2024). *KAN 2.0: Kolmogorov-Arnold Networks Meet Science.* arXiv:2408.10205.
- Kolmogorov, A. N. (1957). *On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition.* Doklady Akademii Nauk SSSR, 114, 953–956.
- Blealtan (2024). *efficient-kan: An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network.* GitHub: blealtan/efficient-kan.
- Shi, X. et al. (2024). *KarSein: Kolmogorov-Arnold Regularized Self-Interaction Network for CTR Prediction.* arXiv:2408.08713.
- Yang, S. et al. (2024). *CF-KAN: Kolmogorov-Arnold Network-based Collaborative Filtering.* arXiv:2409.05633.
- Yang, M. et al. (2025). *ShapKAN: Shapley-value based Pruning for Stable Symbolic Extraction from KANs.* arXiv:2510.01663.
- Zhou, C. et al. (2025). *CoxKAN: Kolmogorov-Arnold Networks for Interpretable, High-Performance Survival Analysis.* PMC 2025.
- Liu, T. et al. (2025). *TabKANet: Tabular Data Modelling with Kolmogorov-Arnold Networks.* ScienceDirect.
- Parlamento Europeo y Consejo de la Unión Europea (2024). *Reglamento (UE) 2024/1689 por el que se establecen normas armonizadas en materia de inteligencia artificial (Reglamento de Inteligencia Artificial).* DOUE.
- Criteo Labs (2014). *Criteo Display Advertising Challenge Dataset.* Kaggle.
- Lundberg, S. M. y Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. (Referencia de contraste para explicabilidad post-hoc.)
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016. (Ídem.)
