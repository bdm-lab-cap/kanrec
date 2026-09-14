# KAN-REC — Power BI Dashboard Setup
# Archivo: powerbi/README.md
#
# Este archivo documenta cómo conectar Power BI a OneLake y
# las medidas DAX necesarias para el dashboard de KAN-REC.

> **Origen de las tablas.** Todas las que consume este panel las crean los
> notebooks de Fabric, ninguna se construye a mano:
> `experiment_results` ← `04`; `streaming_kfk` ← Eventstream (Confluent);
> `streaming_processed`, `streaming_predictions`, `streaming_metrics`, `streaming_drift` ← `06`;
> `symbolic_results`, `spline_curves` y `baseline_metrics` ← `05` (celda final).

## Estructura del dashboard (4 páginas)

### Página 1: Curvas φ — Encoding por campo numérico
### Página 2: Fórmula de scoring simbólica
### Página 3: Comparativa AUC — KAN vs AutoDis vs Raw
### Página 4: Real-Time CTR — stream Kafka en vivo

---

## Conexión Power BI → OneLake

### Desde Fabric Workspace:
1. Abre el Lakehouse `kanrec_lakehouse`
2. Haz clic en "Open in Power BI" (botón en la esquina superior derecha)
3. Power BI Desktop se abre con la conexión a OneLake ya configurada
4. Selecciona las tablas: experiment_results, symbolic_results, baseline_metrics

### Tablas necesarias en OneLake:
- `kanrec_lakehouse.experiment_results` → métricas de entrenamiento
- `kanrec_lakehouse.symbolic_results`   → fórmulas extraídas por campo/seed
- `kanrec_lakehouse.baseline_metrics`   → thresholds para alertas
- `kanrec_lakehouse.streaming_processed`   → impresiones normalizadas del stream (Página 4)
- `kanrec_lakehouse.streaming_predictions` → P(click) por impresión, modelo y lote (Página 4)
- `kanrec_lakehouse.streaming_metrics`     → por lote: CTR observado vs predicho, log-loss, AUC, PSI máx, cobertura mín (Página 4)
- `kanrec_lakehouse.streaming_drift`       → por lote y campo: PSI, nivel, cobertura del rango calibrado (Página 4)
- `kanrec_lakehouse.spline_curves`      → curvas φ evaluadas en grid (Página 1)

---

## Medidas DAX

### Página 3: Comparativa AUC

```dax
-- AUC medio por encoder
AUC_Medio =
AVERAGEX(
    FILTER(experiment_results, experiment_results[dataset] = "criteo"),
    experiment_results[test_auc]
)

-- Diferencia vs AutoDis
Delta_vs_AutoDis =
VAR auc_kan =
    CALCULATE(
        AVERAGE(experiment_results[test_auc]),
        experiment_results[encoder] = "kan-bspline"
    )
VAR auc_autodis =
    CALCULATE(
        AVERAGE(experiment_results[test_auc]),
        experiment_results[encoder] = "autodis"
    )
RETURN auc_kan - auc_autodis

-- Estabilidad simbólica por campo
Estabilidad_Simbolica =
DIVIDE(
    COUNTROWS(
        FILTER(
            symbolic_results,
            symbolic_results[is_accepted] = TRUE()
        )
    ),
    COUNTROWS(symbolic_results)
)
```

### Página 2: Fórmula de scoring

```dax
-- Operador dominante por campo
Operador_Dominante =
TOPN(
    1,
    SUMMARIZE(
        FILTER(symbolic_results, symbolic_results[is_accepted] = TRUE()),
        symbolic_results[field_name],
        symbolic_results[operator],
        "n_seeds", COUNTROWS(symbolic_results)
    ),
    [n_seeds], DESC
)

-- R² medio por campo
R2_Medio_Campo =
AVERAGEX(
    FILTER(symbolic_results, symbolic_results[is_accepted] = TRUE()),
    symbolic_results[r2]
)
```

---

## Tabla spline_curves (generada por fabric/05_symbolic_extraction.py)

Esta tabla se genera en Colab tras el entrenamiento y se sube a OneLake.
Contiene las curvas φ evaluadas en 300 puntos por campo numérico.

Schema:
```
field_name  STRING    -- e.g. "I1", "I3"
field_idx   INT
x_value     FLOAT     -- valor de entrada normalizado [-3, 3]
y_value     FLOAT     -- salida del encoder (dimensión 0 del embedding)
seed        INT
dataset     STRING
```

Generación en Colab (añadir al notebook de extracción):
```python
import pandas as pd

curves_records = []
for j, field_name in enumerate(field_names):
    x_grid, y_curves = model.numerical_encoder.get_spline_curves(j, n_points=300)
    for i in range(len(x_grid)):
        curves_records.append({
            "field_name": field_name,
            "field_idx":  j,
            "x_value":    float(x_grid[i]),
            "y_value":    float(y_curves[i, 0]),
            "seed":       seed,
            "dataset":    dataset,
        })

curves_df = pd.DataFrame(curves_records)
curves_df.to_parquet("spline_curves.parquet", index=False)
# Subir a OneLake via Azure SDK (ver script upload_to_fabric.py)
```

---

## Visuales recomendados por página

### Página 1: Curvas φ
- **Gráfico de líneas** por campo (eje X = x_value, eje Y = y_value)
- Filtro por field_name y seed
- Anotación del operador simbólico dominante

### Página 2: Fórmula simbólica
- **Tabla** con campo, operador, R², estabilidad (%)
- **KPI cards** para gap RMSE y n_fields_accepted
- **Gráfico de barras** comparando R² por campo

### Página 3: Comparativa AUC
- **Gráfico de columnas agrupadas**: AUC por encoder (raw, autodis, kan-bspline, kan-rbf)
- **Gráfico de dispersión**: latencia vs AUC por encoder
- **Tabla de ablaciones**: grid_size vs AUC

### Página 4: Real-Time CTR — el modelo en servicio
- **Líneas, eje X = batch_id** (streaming_metrics): `ctr_observed` y `ctr_predicted`
  superpuestas. Es la imagen del modelo sirviendo: si las dos líneas se separan,
  el modelo está descalibrado respecto al tráfico.
- **KPI cards** (último lote): AUC online, log-loss, impresiones, modelo (`model`,
  checkpoint@sha) — el mismo hash que verifica `check_manifest` en 06.
- **Histograma** (streaming_predictions): distribución de `p_click` del último lote.
- **Matriz campo × lote** (streaming_drift): `psi` con formato condicional por
  `psi_level` (ok / warning / alert) y `coverage` con umbral 0,99. La cobertura
  es la señal propia de KAN-REC: fracción del lote que cae dentro del rango de
  nudos calibrado de cada φⱼ, es decir, dentro de la curva que se auditó.

Medidas DAX de la página 4:

```dax
-- Desviación de calibración del último lote (predicho - observado, en puntos de CTR)
Calib_Gap_pp =
VAR ultimo = MAX(streaming_metrics[batch_id])
RETURN 100 * (
    CALCULATE(MAX(streaming_metrics[ctr_predicted]), streaming_metrics[batch_id] = ultimo)
  - CALCULATE(MAX(streaming_metrics[ctr_observed]),  streaming_metrics[batch_id] = ultimo)
)

-- Campos en alerta de deriva en el último lote
Campos_En_Alerta =
VAR ultimo = MAX(streaming_drift[batch_id])
RETURN CALCULATE(
    COUNTROWS(streaming_drift),
    streaming_drift[batch_id] = ultimo,
    streaming_drift[psi_level] = "alert" || streaming_drift[coverage_alert] = TRUE()
)
```

### Reglas de Data Activator (sobre `streaming_metrics` y `streaming_drift`)

| Regla | Condición | Por qué |
|---|---|---|
| Descalibración | `ABS(ctr_predicted - ctr_observed) > 0.03` en 2 lotes seguidos | El modelo sigue ordenando bien (AUC) pero sus probabilidades ya no son fiables para pujar. |
| Deriva de entrada | `psi_level = "alert"` en cualquier campo | La distribución de un campo ha cambiado respecto a train: candidato a reentrenar. |
| Fuera de la curva auditada | `coverage < 0.99` en cualquier campo | Parte del tráfico se evalúa fuera del grid calibrado, donde φⱼ ya no es la curva extraída ni la fórmula persistida en MongoDB. Señal exclusiva de un encoder interpretable. |
| Modelo cambiado | `model` del último lote ≠ `model` del anterior | Trazabilidad: quién sirvió qué. |
