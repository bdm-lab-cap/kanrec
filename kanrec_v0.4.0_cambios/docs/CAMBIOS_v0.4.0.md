# kanrec 0.4.0 — el modelo se sirve, se vigila y se empaqueta

Cuatro cambios, todos en el paquete y en los notebooks de Fabric. Ninguno
altera un resultado ya publicado en la memoria (AUC, extracción simbólica,
fidelidad): añaden la capa de servicio que faltaba y corrigen dos
afirmaciones que el código no respaldaba.

## 1. Scoring online del stream (`kanrec.serving`, `fabric/06`)

| Antes | Ahora |
|---|---|
| 06 normalizaba el stream y agregaba el CTR observado. Ningún checkpoint puntuaba una impresión. | Cada micro-batch se puntúa con el checkpoint KAN-REC vectorizado (`score_spark`, `mapInPandas`). Se escriben `streaming_predictions` (P(click) por impresión) y `streaming_metrics` (CTR observado vs predicho, log-loss, AUC online por lote). |
| `spark.read.table` + `overwrite`: reprocesaba todo el histórico en cada ejecución. | Structured Streaming sobre la tabla Delta con `trigger(availableNow=True)` y checkpoint de offsets: sólo lo nuevo desde la última ejecución, y termina (encaja en Data Factory). |
| Checkpoint = `state_dict` desnudo; cargarlo exigía recalcular cardinalidades leyendo `train`. | `load_model` infiere encoder e hiperparámetros de las formas del `state_dict`. 04 escribe `<ckpt>.manifest.json` (versión, commit, columnas, cardinalidades, SHA-256 del checkpoint y de `scaler_stats.json`). 06 llama a `check_manifest` y **aborta** si los estadísticos de normalización no son los del entrenamiento. |
| Normalización duplicada a mano en 01 y 06 (el comentario de 06 que decía "misma función" era falso). | `kanrec.spark_utils.normalise_like_train` (imputación, log1p, escalado, 26 joins). 01 y 06 la llaman con los mismos artefactos. |
| Índices categóricos fuera de rango → `device-side assert`. | `Scorer` los recorta y los cuenta (`n_clamped`). |

El pipeline `kanrec_end_to_end.json` añade la dependencia 06 → 04 (antes
06 sólo dependía de 01; ahora necesita el checkpoint y su manifiesto).

## 2. Deriva (`kanrec.drift`)

Por lote y por campo, escrito en `streaming_drift`:

- **Cobertura del rango calibrado**: fracción del lote dentro de los nudos
  calibrados de φⱼ. Fuera de ese rango la spline es nula y φⱼ degenera en la
  ruta base; el modelo responde, pero no con la curva auditada ni con la
  fórmula persistida en MongoDB. Es la señal que sólo un encoder
  interpretable puede dar.
- **PSI** frente a train sobre bins de cuantiles (umbrales 0,10 / 0,25).
  La referencia se ajusta una vez y se persiste en
  `Files/config/drift_reference.json`.

Reglas de Data Activator y visuales de Power BI: `powerbi/README.md`,
página 4. La tercera señal que la memoria dice haber descartado (deriva de
la forma funcional) no está implementada, coherente con el texto.

## 3. Baseline raw vectorizada (`VectorizedRawEncoder`)

`RawNumericalEncoder` era un bucle de 13 `Linear(1, d)`; el sobrecoste
1,15× comparaba un encoder optimizado contra uno sin optimizar.
`vectorize_model` ahora vectoriza también raw. `experiments/latency_report.py`
regenera la tabla con los cinco modelos y **mide `torch.equal`** sobre el
checkpoint real, que es lo único que autoriza a escribir "bit a bit".

## 4. Wheel en CI

Job `build`: `python -m build`, `twine check`, instalación en un venv limpio
fuera del árbol de fuentes, artefacto de la ejecución y Release en tags `v*`.
Es la wheel que se sube al entorno de Fabric.

## Otras correcciones encontradas por el camino

- **04 sobreescribía el checkpoint principal.** La ablación de `grid_size`
  reentrenaba `kan-bspline gs10 s42` con 15 épocas en el **mismo fichero**
  que la comparativa (30 épocas), así que 05 y 06 cargaban la versión corta.
  Los checkpoints de ablación llevan ahora sufijo `_abl`.
- `kanrec/__init__.py` decía 0.3.0 y `setup.py` 0.3.2. Ambos: 0.4.0.
- El docstring de `vectorized.py` tenía cifras de una corrida anterior
  (81,7 %, 4,3×) distintas de la memoria (84,5 %, 4,66×) y decía "diferencia
  medida 2,4e-7" donde la memoria dice "bit a bit". Se ha reescrito para
  remitir a la tabla de la memoria y distinguir tolerancia de tests de
  equivalencia medida.
- `MANIFEST.in` para que la wheel incluya la licencia de efficient-kan.

## Tests nuevos

`tests/test_drift.py` (18), `tests/test_serving.py` (15 funciones, 18 casos),
`tests/test_vectorized.py` (+3). Los de drift son numpy puro y se han
ejecutado; los de serving y vectorized requieren torch y se ejecutan en CI.
**Actualizar la cifra de tests en memoria, README y guión con la que
imprima `pytest` en CI, una sola cifra en los tres sitios.**

## Orden de ejecución para regenerar todo

1. `git tag v0.4.0 && git push --tags` → CI construye la wheel. Subirla al
   entorno de Fabric (sustituye a la 0.3.2) y **asignar el entorno también a
   los ejecutores**: `score_spark` corre en `mapInPandas`.
2. Fabric, pipeline completo: 01 → 04 (escribe manifiestos en la celda 6b) →
   05 → 06 (primera ejecución: crea `drift_reference.json` y las cuatro
   tablas de streaming). Con el productor de Kafka activo, ejecutar 06 dos o
   tres veces separadas unos minutos para tener varios `batch_id` en el
   panel.
3. Colab (T4): `python experiments/latency_report.py --ckpt-dir ... --out
   resultados/latency.json`. Copiar a la memoria: sobrecoste
   `overhead_kan_vs_raw (ambos vectorizados)` y la frase `claim` de
   `equivalence["kan-vec"]`.
4. Power BI: añadir los visuales de la página 4 (`powerbi/README.md`).
5. Capturas para el Anexo D: salida de 06 (líneas `[batch N] ...`), tabla
   `streaming_metrics`, matriz de deriva.

## Qué tocar en la memoria (párrafos concretos)

- **5.1 Diagrama**: en la rama de streaming, tras "06 Stream processing",
  añadir "→ scoring con checkpoint KAN-REC → streaming_predictions /
  streaming_metrics / streaming_drift".
- **5.5 DevOps, Artefacto**: "wheel construida y validada por CI en un
  entorno limpio; publicada como artefacto de la ejecución y Release".
- **6.3 Flujos, streaming**: sustituir "procesamiento con los mismos
  estadísticos que el batch → streaming_processed y streaming_metrics" por:
  Structured Streaming incremental (`availableNow`, checkpoint de offsets);
  misma función de normalización (`normalise_like_train`) que 01; scoring
  online con el checkpoint verificado por manifiesto; métricas de
  calibración y deriva por lote.
- **6.4 Streaming**: añadir el manifiesto (`check_manifest` aborta si
  `scaler_stats.json` no es el del entrenamiento) y el recorte de índices.
- **6.5 Orquestación**: 06 depende ahora de 01 (estadísticos) y de 04
  (checkpoint + manifiesto). El párrafo del "punto de encuentro" gana un
  argumento: 06 consume el plano continuo y sirve el modelo del plano por
  lotes.
- **6.6 Almacenamiento**: fila OneLake/Delta añade
  `streaming_predictions`, `streaming_drift`; Files añade
  `*.manifest.json`, `drift_reference.json`.
- **6.7 Explotación, operacional**: el panel en tiempo real muestra CTR
  predicho vs observado, AUC online y deriva por campo; reglas de Data
  Activator sobre descalibración, PSI y cobertura del rango calibrado.
- **7.1 Resultado 5**: sustituir la fila "Sobrecoste vs. normalización" por
  el valor con raw vectorizada; sustituir "exactamente cero, bit a bit" por
  la frase que devuelva `latency_report.py`.
- **7.1 Resultado 6**: añadir "modelo servido sobre el stream con
  verificación de consistencia por manifiesto; deriva vigilada por lote".
- **7.5 Monitorización continua**: pasa de "futura línea" a hecho; dejar el
  párrafo sobre la señal descartada, que sigue siendo cierto.
- **Palabras clave**: añadir "Structured Streaming", "Model serving",
  "Drift detection".

## Qué tocar en el guión

- Bloque 2, tras "escribe simultáneamente en Delta Lake y en una base KQL":
  "y un consumidor incremental puntúa cada lote con el modelo entrenado,
  verificando por hash que sirve con los mismos estadísticos con los que se
  entrenó". En pantalla: el panel con las dos líneas (CTR observado y
  predicho) y la matriz de deriva. Es la única imagen del vídeo de un
  sistema vivo.
- Bloque 5: la cifra de sobrecoste y la frase de equivalencia salen de
  `latency_report.py`, no de la versión anterior.
- Cierre: "…una descripción auditable de cada variable, un monitor que avisa
  cuando el tráfico sale de la curva auditada, y cuesta un X % más".
