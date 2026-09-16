# Resultados

Datos en bruto de los que salen las cifras de la memoria. Cada fichero se
corresponde con un resultado concreto, de modo que las tablas del documento
pueden comprobarse sin ejecutar nada.

Todos provienen de la comparativa en Colab (GPU T4, 1,5 M de filas de Criteo,
semillas 42, 123 y 256) y del análisis de cierre sobre esos mismos
checkpoints. La ejecución en Fabric es la del pipeline, a escala reducida, y
sus tablas viven en el lakehouse.

| Fichero | Sección | Contiene |
|---|---|---|
| `experiment_results_colab.csv` | 7.1 · Resultado 1 | Una fila por encoder y semilla: AUC y log-loss de test, épocas, tiempo y número de parámetros |
| `experiment_summary_colab.csv` | 7.1 · Resultado 1 | Media y desviación por encoder sobre las tres semillas |
| `gridsize_ablation_colab.csv` | 7.1 · Resultado 2 | Ablación de `grid_size` ∈ {5, 10, 20} sobre KAN-REC |
| `symbolic_results.json` | 6.5 · pipeline | Salida del notebook 05 en Fabric sobre el checkpoint entrenado en el pipeline: fórmula, operador, R² y acuerdo entre dimensiones por campo y semilla |
| `estabilidad_semillas.csv` | 7.1 · Resultado 4 | Operador dominante por campo y en cuántas semillas se repite |
| `fidelidad.csv` | 7.1 · Resultado 4 | Ablación por sustitución: error de curva y degradación de AUC, por semilla |
| `monotonia.csv` | 7.2 | Tasa de violación de monotonía por campo, sobre las 16 dimensiones |
| `latency_v040.json` | 7.1 · Resultado 5 | Latencia de los cinco modelos y equivalencia numérica original/vectorizado |
| `comparison_figure.png` | — | Figura de la comparativa con barras de error |

## Cómo leer las cifras principales

**Paridad predictiva.** En `experiment_summary_colab.csv`: KAN-REC 0,7851 ± 0,0015
frente a 0,7841 ± 0,0010 de la normalización directa. La diferencia (0,0010) es
del orden de la desviación entre semillas, así que no es significativa. El
detalle por semilla está en `experiment_results_colab.csv`, donde se ve que la
normalización directa gana en una de las tres.

**Capacidad del spline.** En `gridsize_ablation_colab.csv`: 10 es el mejor valor
(0,7843), 5 pierde 0,0078 y 20 diverge (0,4277, por debajo del azar, deteniéndose
a las 3 épocas).

**Operadores.** Las cifras del Resultado 3 (R² entre 0,919 y 0,998, acuerdo
del 88 % al 100 %) provienen de la extracción sobre los checkpoints de Colab;
la tabla por campo está en el Anexo D.3 y los datos por semilla en
`estabilidad_semillas.csv`. `symbolic_results.json` es la extracción que el
pipeline de Fabric ejecuta sobre su propio checkpoint (200.000 filas): otro
modelo, mismo resultado cualitativo, `linear` en todos los campos y en las tres
semillas.

**Estabilidad.** En `estabilidad_semillas.csv`: 8 de 11 campos recuperan el
operador idéntico en las tres semillas; la media ponderada es 90,9 %. Los tres
campos con 2/3 (I3, I7, I12) son los de mayor R², donde varios operadores son
numéricamente equivalentes sobre una curva cuasi-lineal.

**Fidelidad.** En `fidelidad.csv`: error relativo de curva 0,066 ± 0,003 y
degradación de AUC 0,033 ± 0,004. El AUC base de este análisis (0,72) es menor
que el de la comparativa porque se ejecuta sobre una submuestra de 300.000
filas; lo relevante es la diferencia entre el modelo original y el sustituido,
no su valor absoluto.

**Monotonía.** En `monotonia.csv`: 11 de 13 campos por debajo del 5 % de
violación (media 2,84 %). I9 e I11 quedan en 6,1 % y 5,2 %.

**Latencia y equivalencia.** En `latency_v040.json`, medido sobre GPU T4 con
lotes de 4.096 y 100 repeticiones. El sobrecoste que reporta la memoria es
`sobrecoste_kan_vs_raw_ambos_vectorizados` (1,48×): compara los dos encoders
optimizados, porque la baseline recorre los campos en el mismo bucle de Python
que recorría KAN antes de vectorizarlo. El bloque `equivalencia` recoge la
comprobación que respalda la afirmación de que la optimización no cambia el
modelo: para `kan-vec`, `torch_equal` es cierto y la diferencia máxima es
exactamente 0. Para `raw-vec` no lo es (3,8·10⁻⁶), porque sustituir trece
productos matriciales por una operación elemento a elemento cambia el orden de
las operaciones en coma flotante.

## Nota sobre el número de campos

La selección por norma L1 de los coeficientes spline retiene diez u once campos
según el checkpoint: I3 e I12 quedan cerca del umbral del percentil 20 y entran
o no dependiendo del modelo. La conclusión no cambia, porque el operador es
lineal en todos los campos de todas las ejecuciones.

## Lo que no está aquí

Los checkpoints `.pt` (108 MB cada uno, nueve en total) y el TSV de Criteo
(2,26 GB) no se versionan. El dataset se obtiene con `data/download_criteo.sh` y
los checkpoints se regeneran con `fabric/04_model_comparison.py` o con el
cuaderno de Colab.
