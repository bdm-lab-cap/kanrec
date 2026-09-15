# KAN-REC

**Codificación de variables numéricas continuas y extracción de puntuación simbólica para sistemas de recomendación**

> TFM · Máster en Big Data & Data Engineering · Universidad Complutense de Madrid
> Autor: Pedro Antonio Martínez Sánchez
> Tutores: Jorge Centeno · Alberto González · Septiembre 2026

[![CI](https://github.com/bdm-lab-cap/kanrec/actions/workflows/ci.yml/badge.svg)](https://github.com/bdm-lab-cap/kanrec/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Microsoft Fabric](https://img.shields.io/badge/Microsoft-Fabric-blue.svg)](https://app.fabric.microsoft.com)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://cloud.mongodb.com)
[![Confluent Cloud](https://img.shields.io/badge/Confluent-Cloud-red.svg)](https://confluent.cloud)

---

## Qué es KAN-REC

KAN-REC sustituye la discretización de **AutoDis** (Guo et al., KDD 2021) por un
**encoder basado en Kolmogorov-Arnold Networks**: cada variable numérica se
transforma en un embedding mediante una B-spline aprendida, sin cubos y sin
discontinuidades en los límites.

Su aportación no es predictiva sino **descriptiva**. Tras el entrenamiento, la
curva aprendida de cada campo se ajusta a una librería de operadores
elementales, produciendo una fórmula cerrada por variable que puede auditarse
sin abrir el modelo:

```
φⱼ(x) ≈ aⱼ·op(x) + bⱼ        op ∈ {x, log(x+1), exp(x), x², √x, 1/x, sigmoid(x)}
```

El operador ganador, sus coeficientes, el R² del ajuste y el grado de acuerdo
entre las dieciséis dimensiones del embedding se calculan en
`05_symbolic_extraction`, se persisten en MongoDB Atlas y se consultan en la
página de fórmulas del informe de Power BI.

Esa descripción se persiste en MongoDB Atlas, se explota en Power BI y —desde la
versión 0.4.0— se **vigila en producción**: el rango calibrado de cada spline
define dónde la fórmula sigue siendo válida, y el sistema alerta cuando el
tráfico se sale de él.

**Contexto regulatorio.** Las obligaciones de transparencia sobre sistemas de
recomendación (Art. 27 del DSA, y el AI Act para los usos que caen bajo su
ámbito) exigen explicar los parámetros principales que determinan lo que se
muestra a cada usuario. Un modelo que produce una descripción funcional cerrada
de cada variable responde a ese requisito por construcción, no mediante
explicaciones post-hoc.

---

## Resultados

### Capacidad predictiva — paridad, no superioridad

Criteo, 1,5 M de filas, tres semillas, backbone idéntico para los tres encoders:

| Encoder | Test AUC | Log-loss | Entrenamiento |
|---|---|---|---|
| Normalización directa | 0,7841 ± 0,0010 | 0,4631 ± 0,0019 | 144 s |
| AutoDis (KDD 2021) | 0,7816 ± 0,0012 | 0,4651 ± 0,0029 | 195 s |
| **KAN-REC** | **0,7851 ± 0,0015** | **0,4624 ± 0,0020** | 258 s |

La diferencia frente a la normalización directa (0,0010) es **del mismo orden
que la desviación entre semillas**, por lo que no es estadísticamente
significativa: hay **paridad**. La normalización directa gana en una de las tres
semillas, lo que lo confirma. Frente a AutoDis la diferencia sí es consistente
(0,0035, unas 2,3 desviaciones, ganando en las tres semillas).

Una versión anterior de este README publicaba un AUC de 0,897. Ese resultado era
inválido: el muestreo usaba `.limit()` de Spark, que devuelve las primeras filas
en orden de fichero en lugar de una muestra aleatoria, de modo que positivos y
negativos provenían de ventanas temporales distintas y cualquier variable
correlacionada con la posición delataba la etiqueta. Se detectó en una auditoría
interna y se corrigió; la memoria documenta el episodio completo.

### Interpretabilidad — fórmulas auditadas y verificadas

La extracción converge al operador **lineal** en los diez campos retenidos, con
R² entre 0,919 y 0,998 y acuerdo del 88 % al 100 % entre las dieciséis
dimensiones del embedding. Estabilidad entre semillas: 90,9 %.

Que todo salga lineal no es una limitación del método. Entrenando el mismo
modelo sobre señales de forma conocida, el encoder recupera la curva sinusoidal
cuando la señal es sinusoidal y la recta cuando es lineal: la linealidad es un
hallazgo sobre Criteo. La fidelidad se verificó sustituyendo la fórmula dentro
del propio modelo, que reproduce la forma con un 6 % de error.

### Coste de la interpretabilidad

El perfilado mostró que el encoder consumía el 84,5 % del tiempo de inferencia,
no por el coste de evaluar las splines sino porque el forward recorría los trece
campos en un bucle de Python. Vectorizado en una única operación matricial:

| | Original | Vectorizado | Mejora |
|---|---|---|---|
| Encoder numérico | 8,83 ms | 0,95 ms | 9,27× |
| Modelo completo | 10,45 ms | 2,68 ms | 3,89× |
| Tiempo en el encoder | 84,5 % | 35,5 % | — |

`kanrec.vectorized` copia los pesos del encoder entrenado, de modo que la
optimización cambia la velocidad y nunca el modelo. La equivalencia se verifica
en los tests unitarios y se mide sobre el checkpoint real con
`experiments/latency_report.py`, que compara ambas baselines vectorizadas para
que el sobrecoste reportado sea el del método y no el del bucle.

---

## Arquitectura

```
┌───────────────────────────────────────────────────────────────────────┐
│  FUENTES                                                              │
│  Criteo TSV (10 M) · Confluent Cloud (Kafka) · API REST de metadatos  │
└─────────┬──────────────────┬───────────────────┬─────────────────────┘
          ▼                  ▼                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  MICROSOFT FABRIC — workspace KAN-REC                                 │
│                                                                       │
│  01_spark_ingest_mlllib   normalización distribuida → train/val/test  │
│                           + scaler_stats.json + cat_index_maps        │
│  03_api_ingest            enriquecimiento REST → item_metadata        │
│  04_model_comparison      raw vs AutoDis vs KAN, 3 semillas           │
│                           → checkpoints + manifiestos firmados        │
│  05_symbolic_extraction   fórmulas por campo → symbolic_results       │
│  06_stream_processing     Structured Streaming incremental:           │
│                             normaliza (misma función que 01)          │
│                             → puntúa con el checkpoint verificado     │
│                             → deriva por campo (PSI + cobertura)      │
│  09_mongodb_vector_search embeddings KAN → Atlas $vectorSearch        │
│                                                                       │
│  Eventstream · Lakehouse sobre OneLake · Data Pipelines               │
│  Power BI (Direct Lake, 4 páginas) · Activator (alertas)              │
└─────────┬─────────────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  Google Colab (GPU T4)   │   │  MongoDB Atlas                       │
│  comparativa a 1,5 M     │   │  symbolic_results · item_embeddings  │
└──────────────────────────┘   └──────────────────────────────────────┘
```

**Consistencia batch/stream.** El entrenamiento y el servicio aplican
literalmente la misma función de normalización
(`kanrec.spark_utils.normalise_like_train`) con los mismos estadísticos y los
mismos mapas de indexado. No es una convención: cada checkpoint lleva un
manifiesto con el hash SHA-256 de `scaler_stats.json`, y el notebook de
streaming **aborta** si intenta servir con estadísticos distintos de los del
entrenamiento.

---

## El modelo en servicio

Desde la v0.4.0 el circuito se cierra: el modelo no solo se entrena, se sirve y
se vigila.

```python
from kanrec.serving import load_model, check_manifest, score_spark

check_manifest(ckpt, scaler_stats_path=stats, numerical_cols=cols)  # o aborta
scored = score_spark(df, ckpt, numerical_cols, idx_cols, output_col="p_click")
```

`load_model` deduce el encoder y todos sus hiperparámetros de las formas del
`state_dict`, así que un checkpoint es autocontenido: no hace falta reconstruir
el contexto de entrenamiento para cargarlo.

**Detección de deriva** (`kanrec.drift`), por lote y por campo:

- **Cobertura del rango calibrado** — fracción del lote que cae dentro de los
  nudos de φⱼ. Fuera de ese rango la spline es nula y φⱼ degenera en su ruta
  base: el modelo responde, pero ya no con la curva auditada ni con la fórmula
  publicada. Es una señal que **solo existe porque el encoder es interpretable**;
  una capa densa no tiene rango válido que vigilar.
- **PSI por campo** frente a la distribución de entrenamiento, con los umbrales
  habituales de scoring (0,10 vigilar, 0,25 alerta).

---

## Cobertura de asignaturas

| Componente | Tecnología | Asignatura |
|---|---|---|
| Encoder KAN + extracción simbólica | PyTorch + EfficientKAN | Deep Learning |
| Normalización distribuida | Spark en Fabric | Spark |
| Stream de clics en tiempo real | Confluent Cloud + Eventstream | Kafka |
| Feature store | OneLake + Delta Lake | Diseño de ingesta |
| Arquitectura multi-fuente | Microsoft Fabric | Arquitecturas de datos |
| Orquestación cloud | Fabric Data Pipelines | Pipelines cloud |
| Store simbólico + Vector Search | MongoDB Atlas | NoSQL |
| Modelado de tablas Delta | Esquemas Delta | Modelado de datos |
| Seguimiento de experimentos | Fabric ML Experiments | Machine Learning |
| CI/CD + empaquetado | GitHub Actions + wheel | Productivización |
| Cuadro de mando y tiempo real | Power BI Direct Lake | — |
| Contenedores + DevOps | Docker + GitHub Actions | Contenedores |

---

## Puesta en marcha local

```bash
git clone https://github.com/bdm-lab-cap/kanrec.git && cd kanrec

# Credenciales: nunca en el código. Ver SECURITY.md
cp .env.example .env          # ATLAS_URI, CONFLUENT_API_KEY, CONFLUENT_API_SECRET

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,streaming,api]"
```

`efficient-kan` está vendorizado en `kanrec/vendor/` porque no se publica en
PyPI; no hace falta instalarlo aparte.

> **En notebooks de Fabric, instalar `kanrec` sin extras.** El extra `[train]`
> arrastra mlflow desde PyPI y rompe el mlflow propio de Fabric, integrado con
> su plugin `synapse.ml.mlflow`.

```bash
pytest tests/ -v                                        # suite completa
python infra/kafka_producer_confluent.py --max-rows 500 # productor a Confluent
```

### Artefacto

Cada etiqueta `v*` publica una wheel construida por CI, validada con `twine` e
instalada en un entorno limpio fuera del árbol de fuentes antes de publicarse.
Es la misma que se sube al entorno de Microsoft Fabric.

### Microsoft Fabric

1. Workspace `KAN-REC` y lakehouse `kanrec_lakehouse`
2. Subir `criteo_10m.tsv` a `Files/raw/`
3. Publicar la wheel como entorno **del workspace** (no solo del notebook:
   `score_spark` usa `mapInPandas` y necesita el paquete en los ejecutores)
4. Ejecutar `01 → 03 → 04 → 05 → 06 → 09`, o el pipeline completo
   (`fabric/pipeline_kanrec_end_to_end.json`)

---

## Estructura

```
kanrec/
├── .github/workflows/ci.yml        # escaneo de secretos · tests · wheel
├── kanrec/                         # paquete instalable
│   ├── encoder.py                  # KANNumericalEncoder (B-splines por campo)
│   ├── vectorized.py               # versiones vectorizadas (KAN y raw)
│   ├── model.py                    # backbone compartido
│   ├── baselines.py                # raw · AutoDis
│   ├── symbolic.py                 # SymbolicExtractor
│   ├── faithfulness.py             # verificación de fidelidad
│   ├── serving.py                  # carga, manifiesto y scoring en Spark
│   ├── drift.py                    # PSI + cobertura del rango calibrado
│   ├── spark_utils.py              # normalización compartida batch/stream
│   └── vendor/efficient_kan.py     # vendorizado (MIT)
├── fabric/                         # notebooks de Microsoft Fabric
├── experiments/                    # entrenamiento local e informe de latencia
├── infra/                          # Docker, productores Kafka
├── powerbi/                        # modelo semántico, DAX y reglas de alerta
├── tests/
├── setup.py · MANIFEST.in · SECURITY.md
```

---

## Calidad

- **145 tests** con pytest sobre Python 3.11 y un servicio MongoDB real, con un
  **75 % de cobertura** sobre las 1.226 líneas del paquete. Cada test de
  regresión está escrito para **fallar contra el código anterior al arreglo**,
  de modo que documenta el error que previene y no solo el comportamiento
  deseado.
- Escaneo de secretos con gitleaks sobre el historial completo, en cada push.
- Construcción y validación de la wheel en entorno limpio.

---

## Referencias

- Guo et al. (2021). **AutoDis**. KDD 2021. [arXiv:2012.08986](https://arxiv.org/abs/2012.08986)
- Liu et al. (2024). **KAN**. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Liu et al. (2024). **KAN 2.0**. [arXiv:2408.10205](https://arxiv.org/abs/2408.10205)
- Blealtan (2024). **efficient-kan**. [GitHub](https://github.com/blealtan/efficient-kan)
- Parlamento Europeo (2022). **DSA** — Reglamento UE 2022/2065, Art. 27.
- Parlamento Europeo (2024). **AI Act** — Reglamento UE 2024/1689.

---

## Licencia

MIT
