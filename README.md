# KAN-REC

**Codificación continua de variables numéricas y extracción simbólica para sistemas de recomendación**

> TFM · Máster en Big Data & Data Engineering · Universidad Complutense de Madrid
> Autor: Pedro Antonio Martínez Sánchez
> Tutores: Jorge Centeno · Alberto González · Septiembre 2026

[![CI](https://github.com/bdm-lab-cap/kanrec/actions/workflows/ci.yml/badge.svg)](https://github.com/bdm-lab-cap/kanrec/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Microsoft Fabric](https://img.shields.io/badge/Microsoft-Fabric-blue.svg)](https://app.fabric.microsoft.com)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://cloud.mongodb.com)
[![Confluent Cloud](https://img.shields.io/badge/Confluent-Cloud-red.svg)](https://confluent.cloud)

---

## Qué es

KAN-REC sustituye la discretización de **AutoDis** (KDD 2021) por un encoder basado en
**Kolmogorov-Arnold Networks**: cada variable numérica se mapea a su embedding mediante
una spline B-spline aprendible, sin cubos ni discontinuidades. Tras el entrenamiento, las
curvas se ajustan a operadores elementales, produciendo una descripción cerrada y
auditable de cómo el modelo transforma cada variable:

```
phi_I3(x)  ~ -0,7210*x + 0,0319     R2 = 0,9973 +- 0,0018   acuerdo 100 % de 16 dims
phi_I11(x) ~  0,7315*x + 0,5387     R2 = 0,9965 +- 0,0043   acuerdo 100 % de 16 dims
```

Todo el trabajo se sustenta sobre una arquitectura de datos completa en Microsoft Fabric:
ingesta distribuida con Spark de 10 M de impresiones, streaming desde Confluent Cloud,
enriquecimiento vía API REST, lakehouse en Delta, orquestación con Data Factory,
persistencia en MongoDB Atlas y explotación en Power BI.

---

## Resultados

Comparativa sobre **1,5 M de impresiones** de Criteo, 3 semillas, backbone idéntico
(el encoder numérico es la única variable):

| Encoder | Test AUC | Log-loss | Entrenamiento |
|---|---|---|---|
| Normalización directa | 0,7841 ± 0,0010 | 0,4631 ± 0,0019 | 144 s |
| AutoDis (KDD 2021) | 0,7816 ± 0,0012 | 0,4651 ± 0,0029 | 195 s |
| **KAN-REC** | **0,7851 ± 0,0015** | **0,4624 ± 0,0020** | 258 s |

**Lectura honesta.** La diferencia frente a la normalización directa (0,0010) es del mismo
orden que la desviación entre semillas, y la normalización gana en una de las tres: **hay
paridad, no superioridad**. Frente a AutoDis la diferencia sí es consistente (0,0035,
unas 2,3 desviaciones, ganando en las tres semillas).

### Interpretabilidad

- **Extracción simbólica:** operador `linear` en los 10 campos retenidos, R² medio
  0,919–0,998, con acuerdo del 88 %–100 % entre las 16 dimensiones del embedding.
- **Fidelidad:** sustituir phi_j por su fórmula dentro del modelo reproduce la forma con un
  error relativo de 0,066 ± 0,003.
- **Estabilidad:** el mismo operador en 8 de 11 campos sobre 3 semillas (90,9 %).
- **Monotonía:** 11 de 13 campos monótonos (violación media 2,84 %), propiedad emergente
  y verificable, no impuesta.
- **Validación del método:** sobre señales sintéticas de forma conocida, el encoder
  produce curvas cuando las hay (R² lineal 0,601 con `sin(1,5x)`) y rectas cuando no
  (0,937 con señal lineal). La linealidad detectada en Criteo es una propiedad del
  dataset, no una limitación del extractor.

### Optimización de la inferencia

El perfilado mostró que el encoder consumía el **84,5 %** del tiempo de inferencia, por
recorrer los 13 campos en un bucle de Python (13 lanzamientos de kernel por lote).
Vectorizado en una sola operación `bmm`:

| Configuración (lote 4096, T4) | Modelo | Encoder | Filas/s |
|---|---|---|---|
| Normalización directa | 2,58 ms | 0,71 ms | 1.589.091 |
| AutoDis | 4,44 ms | 3,61 ms | 923.550 |
| KAN-REC original | 10,45 ms | 8,83 ms | 391.862 |
| **KAN-REC vectorizado** | **2,68 ms** | **0,95 ms** | **1.525.526** |

El encoder acelera **9,27 veces** y el modelo completo **3,89 veces**. La salida es idéntica
hasta la precisión de float32 (diferencia máxima medida: 0,00 en GPU y 2,4e-7 en CPU), por
lo que todos los resultados anteriores siguen siendo válidos. El modelo vectorizado queda
**prácticamente a la par de la normalización directa** (2,68 frente a 2,58 ms) y 1,66 veces
por delante de AutoDis.

---

## Nota sobre el método

Los primeros resultados de este proyecto daban un AUC de 0,90, muy por encima del estado
del arte publicado en Criteo (0,80–0,815). Una revisión crítica en la semana 6 reveló que
el muestreo usaba `.limit()` de Spark, que no es aleatorio: devolvía las primeras filas en
orden de fichero, de modo que positivos y negativos procedían de ventanas temporales
distintas. Con muestreo aleatorio real el AUC bajó a 0,78.

Esa revisión destapó cuatro problemas más, entre ellos una baseline de AutoDis mal
implementada cuyo log-loss era peor que el de un predictor constante. Todos están
documentados en la sección 7.4 de la memoria, junto con las limitaciones vigentes.

---

## Arquitectura

```
FUENTES                   INGESTA / PROCESO           ALMACENAMIENTO      EXPLOTACION

Criteo TSV (2,26 GB)  -->  01 Spark ingest       -->  Delta / OneLake --> Power BI
10.000.001 filas           log1p, StandardScaler      train/val/test      (Direct Lake)
                           indexado categorico
                                   |
Confluent Cloud       -->  Eventstream           -->  streaming_kfk   --> Real-Time
(Kafka)                    06 Stream processing        + KQL DB            Dashboard
                           (mismos estadisticos)
                                   |
Mock API REST         -->  03 API ingest         -->  train_enriched
                                   |
                                   v
                           04 Comparativa             05 Extraccion   --> MongoDB Atlas
                           raw | AutoDis | KAN            simbolica        symbolic_results
                           3 semillas, MLflow          + fidelidad
```

Orquestado con un pipeline de Data Factory cuyas dependencias son las reales de datos.
Detalles en [`fabric/PIPELINE_README.md`](fabric/PIPELINE_README.md).

---

## Instalación y uso

```bash
git clone https://github.com/bdm-lab-cap/kanrec.git && cd kanrec
python -m venv .venv && source .venv/bin/activate

# efficient-kan esta vendorizado (kanrec/vendor/) porque no se publica en PyPI
pip install -e ".[dev,train]"

pytest tests/ -q                       # 107 tests
```

El extra `[train]` añade mlflow, necesario solo para entrenar en local o Colab.
**No instalarlo dentro de un notebook de Fabric:** sobreescribe el mlflow del runtime y
rompe el plugin `synapse.ml.mlflow`.

### Reproducir los experimentos

```bash
bash experiments/run_all.sh                                  # 3 encoders x 3 semillas
python experiments/symbolic_extraction.py                    # extraccion y fidelidad
PYTHONPATH=. python experiments/figura_validacion_simbolica.py
```

El dataset no se incluye por tamaño; se descarga con `data/download_criteo.sh`.

### Uso del paquete

```python
import torch
from kanrec.baselines import build_model
from kanrec.symbolic import SymbolicExtractor
from kanrec.vectorized import vectorize_model

model = build_model("kan-bspline", num_numerical=13,
                    cat_cardinalities=[...], embedding_dim=16, kan_grid_size=10)
model.calibrate(x_num_sample)          # adapta el grid a la distribucion real

# ... entrenamiento con BCEWithLogitsLoss y model.parameter_groups() ...

resultados = SymbolicExtractor(model).fit_all()   # ajuste sobre las 16 dimensiones
rapido = vectorize_model(model)                   # inferencia 3,89 veces mas rapida
```

---

## Estructura

```
kanrec/                  Paquete instalable
  encoder.py             Encoder KAN (calibracion del grid, winsorizado)
  model.py               Backbone compartido y grupos de parametros
  baselines.py           raw, AutoDis y KAN-REC desde una unica factoria
  symbolic.py            Extraccion simbolica sobre las 16 dimensiones
  ablation.py            Fidelidad por sustitucion
  vectorized.py          Encoder vectorizado (equivalencia verificada)
  latency.py             Medicion de latencia de inferencia
  drift.py               Deteccion de deriva (cobertura y PSI)
  faithfulness.py        Auditoria de monotonia
  schema.py              Esquema unico de los documentos de MongoDB
  spark_utils.py         Utilidades de Spark (muestreo aleatorio real)
  config.py              Secretos: Key Vault, entorno, .env
  vendor/                efficient-kan vendorizado (licencia MIT)

fabric/                  Notebooks de Microsoft Fabric (01, 03-07, 09, 10)
colab/                   Comparativa completa y analisis de cierre
experiments/             Scripts reproducibles
tests/                   107 tests, cada uno ligado a un fallo concreto
powerbi/                 Documentacion del panel
docs/                    Memoria, anexos y guion del video
```

---

## Seguridad

Los secretos se resuelven por `kanrec.config` (Azure Key Vault, variables de entorno,
`.env`), nunca en el código. La CI ejecuta `gitleaks` sobre el historial completo antes de
los tests, y un test de regresión falla si aparece una credencial en el repositorio.
El proyecto sufrió una fuga real de credenciales; su remediación completa está documentada
en [`SECURITY.md`](SECURITY.md).

---

## Licencia

Código bajo licencia MIT. `kanrec/vendor/efficient_kan.py` conserva la licencia MIT
original de [blealtan/efficient-kan](https://github.com/blealtan/efficient-kan).
